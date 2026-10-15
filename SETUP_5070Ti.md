# Laya 本机环境（5070Ti，2026-09-23）

上游：https://github.com/NandhaKishorM/laya （Apache-2.0，Convai Innovations）
本地 clone 在 `算法/Jev/RoboJev/`，以 editable 方式装进 conda 环境 `laya`。

## 环境

```bash
conda activate laya            # /home/crp-5070ti-01/miniforge3/envs/laya
python run_inference_demo.py
```

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.14.0+cu130（自带 sm_120，5070Ti 可用） |
| transformers | 5.17.0 |
| huggingface_hub | 1.32.0 |
| laya | 0.3.7（`pip install -e ".[serve]"`） |

重建步骤：

```bash
conda create -n laya python=3.11 -y
conda activate laya
pip install torch                      # PyPI 默认轮子就是 cu130，arch_list 含 sm_120
pip install -e ".[serve]"
pip install socksio                    # 见下面「两个本机坑」
```

## 两个本机坑

1. **`ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`**
   `.bashrc` 里 `ALL_PROXY=socks5://127.0.0.1:7897`，而 huggingface_hub 1.x 换成了 httpx，
   httpx 走 socks5 需要 `socksio`。不装就下不了权重。

2. **ROS jazzy 污染 `PYTHONPATH`**
   `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`，pytest 自动加载 ROS 的
   `launch_testing` 插件后炸在 `No module named 'lark'`。跑测试前加 `PYTHONPATH=`。
   跑 laya 本身不受影响（laya 不 import 冲突的包），但起 `laya-serve` 时也顺手清掉更稳。

## 权重

首次运行从 HF 下载，缓存在 `~/.cache/huggingface`。三个 checkpoint：

| 名字 | repo | 用途 |
|---|---|---|
| english | `convaiinnovations/laya` | ModernBERT-large 421M，拉丁字母英文 |
| multilingual | `convaiinnovations/laya` subfolder `multilingual` | mmBERT-base 322M，100+ 语言（中文走这个） |
| typed-decisions | 同上 subfolder `typed-decisions` | 微调版 |

`Router(preload=True)` 三个全常驻，冷启动约 94 s（含下载），之后进程内重建约 6 s。

## 实测（4 个问题，一次前向）

| | 延迟 |
|---|---|
| 单模型热路径中位 | **11.2 ms**（min 10.3 / p95 12.2） |
| Router 英文 | 约 445 ms 首次 → 之后同上 |
| Router 中文/印地语 | 55～76 ms（含切 checkpoint） |

README 给的 T4 数字是 32.8 ms，5070Ti 上快 3 倍。

## 仓库测试

CI 是把每个 test 当脚本跑，不是 pytest（好几个文件在模块级 `sys.exit`，pytest 收集会 INTERNALERROR）：

```bash
PYTHONPATH= python tests/test_router.py     # 17 个文件全过
```

## Jev 兼容 HTTP 接口

```bash
PYTHONPATH= LAYA_MODELS=english LAYA_PORT=8077 LAYA_HOST=127.0.0.1 laya-serve
curl --noproxy '*' http://127.0.0.1:8077/v1/systemone -H 'Content-Type: application/json' -d '{...}'
```

`curl` 必须带 `--noproxy '*'`，否则 127.0.0.1 也会被 http_proxy 劫持。
返回体带 `answers` / `usage` / `routing`，字段与 TypeSafe 的 System One API 对齐。

## 已知毛刺（上游的，不是装坏了）

- 每次 load 都会 warn：`this checkpoint ships invalid temperatures ... choice:11+=0.1006 -> 0.5`。
  选项数 ≥11 的 choice 题，confidence 未标定，别拿那个数做阈值门控。
- `noul` 对措辞很敏感：同一封信整封给进去 churn=0.823，只留一句
  "Refund the duplicate today or we cancel" 反而 churn=0.068。中文那条威胁取消的也只有 0.103。
  要用它做流失判断得先在自己的数据上量一遍。
