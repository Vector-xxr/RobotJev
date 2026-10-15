# PlayJev 本机部署（5070Ti，2026-09-24）

上游：https://github.com/OmniJev/PlayJev （Apache-2.0）
0.8B 多模态模型，基座 Qwen3.5-0.8B-Base。**吃一帧画面 + 一组候选动作，吐动作上的概率分布**，
不生成文本。十个浏览器小游戏上用模仿学习 + DAgger 训出来的。

## 环境

```bash
conda activate playjev        # /home/crp-5070ti-01/miniforge3/envs/playjev
cd 算法/Jev/RobotJev/PlayJev
```

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| torch | 2.10.0+cu128（arch_list 含 sm_120，5070Ti 可用） |
| transformers | 5.17.0 |
| triton | **3.6.0**（不是 requirements 写的 3.7.1，见下） |
| playwright | 1.63.0 + chromium-headless-shell 153 |

重建：

```bash
conda create -n playjev python=3.12 -y && conda activate playjev
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
grep -v '^triton==' requirements.txt > /tmp/req.txt      # ← 见下面「requirements 自相矛盾」
pip install -r /tmp/req.txt socksio
playwright install chromium
```

## 两个坑

**1. requirements.txt 自己和自己打架。**
它 pin `triton==3.7.1`，而 `torch==2.10.0` 硬依赖 `triton==3.6.0`，
pip 直接 `ResolutionImpossible`。3.7.1 在 PyPI 上确实存在，所以不是「版本没了」，
就是这个 pin 写错了。按 torch 的要求走（3.6.0），跑起来没问题。
顺带：requirements 注释里指的 `hpc/env_setup.sh` 在仓库里**不存在**。

**2. HF 下载要 `socksio`。**
本机 `.bashrc` 设了 `ALL_PROXY=socks5://127.0.0.1:7897`，huggingface_hub 1.x 走 httpx，
走 socks5 必须有 socksio，不装的话权重一个字节都下不来。
（和 laya 那个环境同一个坑。）

另外 `causal_conv1d` 没装，transformers 会提示回退到 PyTorch 参考实现「much slower」。
实测 22 ms/次已经够快（比 README 标的 H200 43 ms 还快），没装。要更快可以补。

## 跑

```bash
# 1) 直接玩一局
python -m playjev.play snake --policy local --ckpt weights/PlayJev-0.8B --episodes 1

# 2) 起决策服务（OpenJev 的 /v1/systemone 请求形状，只是 state 里放帧）
python -m playjev.serve --ckpt weights/PlayJev-0.8B --port 18732

# 3) demo 页，把每个格子切到本地服务
cd demo && python -m http.server 18080 --bind 127.0.0.1
# 浏览器开 http://127.0.0.1:18080/index.html?server=http://127.0.0.1:18732
```

请求形状（`playjev/serve.py`）：

```json
{
  "state": {"frames": ["data:image/jpeg;base64,..."]},
  "questions": {"move": {"type": "choice",
                         "instructions": "Which move should the player make next?",
                         "criteria": {"up": "...", "down": "...", "left": "...", "right": "..."}}}
}
```

**文本 state 会被 400 拒掉** —— 那是 OpenJev/Laya 的活，这个模型只吃图。

## 实测

| | |
|---|---|
| 加载 | 15.1 s |
| 显存 | **3636 MiB**（README 说 ~3 GB，对得上） |
| 单次决策（稳态） | **21.9 ms** forward / 23.7 ms total |
| 首次请求 | 1071 ms —— 纯预热，别拿它当延迟 |

闭环里和 random 基线对比（各 1 局）：

| 游戏 | local | random |
|---|---|---|
| snake | 66 | — |
| breakout | **625** | 120 |
| mario | **399** | 242 |

CORS 已验证：serve 返回 `Access-Control-Allow-Origin: *`，:18080 的 demo 跨源调 :18732 不会被挡。
`curl` 打本机服务记得 `--noproxy '*'`，否则被 http_proxy 劫持。

## 和 RobotJev / Laya 什么关系

同一个 System One 形状（state + typed questions → 带概率的 typed answers），
但**输入是像素不是文本**。RobotJev 那条链现在是靠 YOLO 把图像压成一句话再喂给纯文本的
Laya；PlayJev 这种形态是直接从像素出动作分布，正好是那条链绕开的东西。
两者的 `/v1/systemone` 请求形状是兼容的，只是 state 一个放 `frames` 一个放 `body`。
