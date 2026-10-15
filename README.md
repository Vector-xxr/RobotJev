# RobotJev

在 Laya 和 PlayJev 两个开源决策模型上做 AR5 的「万物可抓」。位置、高度、宽度、力度全部由俯视 Gemini 335、腕部 Gemini 305 的深度，加上 YOLO-World、CLIP、MobileSAM 现场量出来；PlayJev 只在腕部伺服时选上下左右。抓取流程的完整说明见 [JEV_GRASP_SUMMARY.md](JEV_GRASP_SUMMARY.md)。

> [!WARNING]
> **按现在的硬件，真机流程跑不了。** 9/30 的标定全部作废：
> - 末端从 TG9801 夹爪换成了因时灵巧手（10-12），而 robojev 的满开宽度、力度档位、手指扫掠盒、`gripper.json` 都是按夹爪定的；
> - 腕部 305 换了装法（10-12 重新装到手上），`wrist_J.json` 和夹持中心作废；
> - 俯视 335 换了机位（10-13 改成正上方俯视），`top335_extrinsic.json` 作废。
>
> 10-12 以后，决策方向改在 [`../RobotDecision`](../RobotDecision) 里做。

> [!CAUTION]
> **RobotDecision 在用相机时，不要启动 robojev 的任何程序。** robojev 自带的相机代码（`robojev/camera.py`）打开 Orbbec 前会先复位它的 USB，然后只走 OBSENSOR。腕部 305 走 OBSENSOR 一帧不出，还会把 V4L2 也弄得不出帧，只能物理拔插救回来。RobotDecision 用 `v4l2:` 前缀走 V4L2，见 [工作区说明](../../../工作区说明.md) 第 3 节。

## 三部分的关系

| 目录 | 是什么 | 怎么用 |
|---|---|---|
| `laya/` 等根目录上游文件 | 上游 Laya（文本决策模型），说明见 [README_laya.md](README_laya.md) | 以 editable 方式装进 `laya` 环境。robojev 只有旧实验用它（`policy.py`，经 `bench_policy`、`run_loop`），抓取流程不用 |
| `PlayJev/` | 上游 PlayJev（吃一帧画面、出动作分布），说明见 [PlayJev/README.md](PlayJev/README.md) | 在 `playjev` 环境里起服务，监听 `127.0.0.1:18732`；robojev 通过 `probe_playjev.ask` 调用 |
| `robojev/`、`robojev_agent.py` | 自己的抓取流程和网页代理 | 跑在 `laya` 环境，经 `数据_硬件/SO101/atom_robot` 驱动 AR5 和夹爪 |

上游代码从哪来、改过什么，见 [上游来源.md](上游来源.md) 和 [PlayJev/上游来源.md](PlayJev/上游来源.md)。

## 时间线

| 日期 | 事 |
|---|---|
| 9-23 | `run_loop`：Laya 判方向 87.5% 可用，判幅度和门控不可用（见下面「Laya 在机器人上的边界」） |
| 9-24 | PlayJev 零样本看真实图不行；把状态画成贪吃蛇画面后，方向 100% |
| 9-28 | `run_playjev` 真机抓起蓝色套筒 |
| 9-28~29 | `run_composite` 抓起乒乓球、蓝色套筒、黄胶带、水瓶瓶盖；托盘里的球走 335 直线接近；鸡蛋用 5 档力；点选抓球约 31s |
| 9-29 | 俯视相机从 UGREEN 换成 Gemini 335 |
| 9-30 | 换 335 机位和长手指后重标（留一误差 1.9mm），蓝色套筒成功；同一天夹爪过载卡滞 |
| 10-09 | 腕部 305 走 OBSENSOR 出 0 帧 |
| 10-12 | 末端换成灵巧手，305 重新装到手上；本流程真机暂停，决策方向改走 `../RobotDecision` |
| 10-13 | 335 改成正上方俯视 |
| 10-15 | 目录改名为 RobotJev，PlayJev 并入，模型放进项目的 `weights/` |

## 入口

都在本仓库根目录跑。**必须在根目录**：YOLO-World 和 MobileSAM 的权重是按当前目录加载的。下表省略了前缀，完整写法是：

```bash
cd ~/yuhang_workspace/算法/Jev/RobotJev
PYTHONPATH= ~/miniforge3/envs/laya/bin/python <下表的命令>
```

抓取前要先起 PlayJev 服务、给臂上电，完整步骤见 [SUMMARY 的「命令」](JEV_GRASP_SUMMARY.md#命令)。下表里除了最后两行，跑起来都可能开相机或连臂。

| 做什么 | 命令 |
|---|---|
| 网页代理 | `robojev_agent.py --token <令牌>`。只绑 `127.0.0.1:8766`；网页上点「开始」时由它拉起 `click_grasp_web`，「恢复机械臂」时拉起 `atom_robot.ar5.unseize` |
| 点选抓取 | `-m robojev.click_grasp`；加 `--dry` 只点选、看判定，不连臂 |
| 按指令抓 | `-m robojev.run_composite --instruction "pick up the blue cylinder" --stage grasp`；常用参数见 [SUMMARY](JEV_GRASP_SUMMARY.md#命令) |
| 放回并回 0 位 | `-m robojev.place_and_home --down 0.06`；手里没东西用 `--no-place` |
| 依次抓几样 | 不用前缀，直接 `bash robojev/grasp_sequence.sh "pick up the X" "pick up the Y"`；不给参数时依次抓乒乓球、水瓶、蓝色圆柱。结果在 `logs/seq_<时间>/` |
| 标定 | `-m robojev.calibrate_top335`（`--dry-run` 只算路径不动臂，`--refit` 用上次存的点重新拟合）；`-m robojev.calibrate_gripper`（`--centre` 量夹持中心）。什么时候要重做见 [SUMMARY](JEV_GRASP_SUMMARY.md#标定换了硬件要重做哪些) |
| 读 Orbbec 出厂参数 | 换成 `lerobot` 环境（pyorbbecsdk 只装在那）：`PYTHONPATH= ~/miniforge3/envs/lerobot/bin/python -m robojev.read_orbbec_params` |
| 看相机 | `-m robojev.view_top`，加 `--wrist` 同时看腕部 |
| 只点选、不开相机 | `-m robojev.click_grasp --dry --frame f.npz`（用存下的一帧，npz 里要有 `color`、`depth`） |
| 看各入口的参数 | `-m robojev.<模块> --help` |

## 代码地图（`robojev/`）

**主流程**（各文件做什么见 [SUMMARY 的代码地图](JEV_GRASP_SUMMARY.md#代码地图robotjevrobojev)）

| 类别 | 文件 |
|---|---|
| 入口 | `run_composite`、`click_grasp`、`click_grasp_web`、`place_and_home`、`grasp_sequence.sh`，根目录的 `robojev_agent.py` |
| 感知、规划、判定 | `top335`、`objects`、`perception`、`grip_force`、`clearance` |
| PlayJev 伺服 | `snake_render`（把偏差画成贪吃蛇画面）、`probe_playjev`（`ask` 调服务） |
| 相机、录像 | `camera`、`rec` |
| 标定、查看 | `calibrate_top335`、`calibrate_gripper`、`read_orbbec_params`、`view_top`；结果在 `calibration/` |
| 离线测试 | `bench_snake`（PlayJev 在贪吃蛇画面上判方向） |

**旧实验，但主流程还 import 它们，不能删**

| 文件 | 主流程用的是 |
|---|---|
| `run_playjev` | `settle`、`adaptive_board`、`GripperXYZ` |
| `move_pregrasp` | `slow_goto` |
| `calibrate_top` | `path_min_z`（规划平移高度用） |
| `composite` | `run_composite` 在模块顶层 import；只有 `--locate demo` 真用到 |
| `top_servo` | 只有 `--locate top`（旧 UGREEN 俯视，相机已拆）用到 |
| `bench_select` | CLIP 类 `ClipSelect` |
| `actions` | 步长 `COARSE_M`、`FINE_M` 和 `hold` |

**纯旧代码**（主流程不 import）：`run_loop`、`bench_policy`、`parser`、`state`、`policy`、`localizer`、`detector`、`color_detector`、`proprio`、`calibrate`、`depth_probe`、`select_target`、`fit_demo_homography`、`wrist_refine`、`solve_down`。删之前注意：
- `bench_policy` 和 `run_loop --sim` 是下面的离线测试，它们 import 了 `parser`、`state`、`policy`，`run_loop` 还 import 了 `localizer`、`detector`、`color_detector`；
- `run_playjev`、`probe_playjev` 自己的命令行还会 import `color_detector`（它又 import `detector`）。

## 离线测试

不连硬件。在仓库根目录、`laya` 环境下跑，命令前都加 `PYTHONPATH=`（下表省略 `PYTHONPATH= ~/miniforge3/envs/laya/bin/python`）。

| 命令 | 正常结果 |
|---|---|
| `for t in tests/test_*.py; do PYTHONPATH= ~/miniforge3/envs/laya/bin/python $t; done` | 18 个文件 17 个过。`test_local_e2e` 要手动导出到 `~/laya_models` 的模型，不算 |
| `-m robojev.bench_policy` | 方向一致 21/24 = 87.5% |
| `-m robojev.run_loop --instruction "把苹果放到盘子里" --sim --compare` | 54 步跑完；和解析解方向一致约 57%（`--policy geometric` 是 48 步） |
| `run_inference_demo.py` | 跑完；中文、印地语路由到多语言模型（延迟见 [SETUP_5070Ti.md](SETUP_5070Ti.md#实测4-个问题一次前向)） |
| 先起 PlayJev 服务，再 `-m robojev.bench_snake --n 20` | 方向全对，左右、上下镜像后 20/20 |
| 先起 PlayJev 服务，再 `-m robojev.run_playjev --sim-target 0.50 0.05` | 跑完，方向 8/8 |

PlayJev 自己的测试见 [PlayJev/SETUP_5070Ti.md](PlayJev/SETUP_5070Ti.md#测试)。

## 两个环境

| 环境 | 用途 | 说明 |
|---|---|---|
| `laya`（Python 3.11） | Laya、robojev、RobotDecision 共用 | 版本和装法见 [SETUP_5070Ti.md](SETUP_5070Ti.md) |
| `playjev`（Python 3.12） | 只跑 PlayJev 服务 | 版本和装法见 [PlayJev/SETUP_5070Ti.md](PlayJev/SETUP_5070Ti.md)。服务常驻约占 3.6G 显存 |

## 硬件依赖

| 硬件 | robojev 怎么用 |
|---|---|
| AR5 | 经 `atom_robot` 驱动，网络配置见 [工作区说明](../../../工作区说明.md) 第 6.4 节；上电命令见 [SUMMARY 的「命令」](JEV_GRASP_SUMMARY.md#命令) |
| 俯视 Gemini 335 | 序列号 `CP0BB53000FC`，`robojev/camera.py` 按序列号找、走 OBSENSOR |
| 腕部 Gemini 305 | 序列号 `CV27864000V2`，同上（现在走 OBSENSOR 出 0 帧，见开头） |
| 夹爪 HKVR-TG9801 | robojev 没有显式指定末端，靠 `atom_robot/ar5/config.py` 里 `end_effector` 的默认值 `"hkvr"` |
| 标定文件 | `robojev/calibration/`：`top335_extrinsic.json`、`gripper.json`、`wrist_J.json` 是 9/30 的；`orbbec335_params.json` 是 9/29 的；`orbbec_params.json`、`pregrasp_pose.json`（固定抓取朝向）是 9/28 的。旧版本在 `backup_*` |
| 模型 | 根目录的 `yolov8x-worldv2.pt`、`mobile_sam.pt`；`weights/clip/`（CLIP）；`weights/laya/`（Laya）；`PlayJev/weights/PlayJev-0.8B/`。都不进 git |

## 已知问题和提示

- **真机跑不了**、**别和 RobotDecision 抢相机**：见开头两条。
- 抓取流程本身的问题（固定朝向够不着的区域、外参和末端耦合、走廊里的假障碍、力度类别分差小等）见 [SUMMARY 的已知问题](JEV_GRASP_SUMMARY.md#当前状态与已知问题10-15)。
- `run_composite.py` 开头的模块文档（`--help` 打出来的那段）还是 9-28 早上的流程，比如示范单应性、夹持中心 [15,32,88]mm，以 SUMMARY 为准。
- **本机请求别走代理**：`.bashrc` 设了 `http_proxy`，连 `127.0.0.1` 也会走代理；PlayJev 服务没起时，代理会回 502，把「连接被拒」遮住。`probe_playjev.ask` 已经绕开代理；自己用 `curl` 测本机服务时加 `--noproxy '*'`。
- 机器重启后 PlayJev 服务不会自己起来。内核升级后 GPU 驱动可能掉；GPU 用不了时，可以临时用 `PlayJev/serve_cpu.py` 在 CPU 上起服务，很慢，只能应急，用法见 [PlayJev/SETUP_5070Ti.md](PlayJev/SETUP_5070Ti.md#跑)。

## Laya 在机器人上的边界

9-23 用 `run_loop` 把 Laya 接进 AR5 闭环：一句指令 → YOLO 感知 → 把偏差写成话 → Laya 从 12 个离散动作（6 个方向 × 2 档幅度）里选 → 臂执行。结论：**Laya 只能用来判方向。**

`bench_policy` 的实测（24 道方向题，判据是解析解）：

| 问的东西 | 表现 | 随机基线 | 用不用 |
|---|---|---|---|
| 方向（6 选 1） | **87.5%** | 16.7% | 交给 Laya |
| 幅度（2 选 1） | 0% | 50% | 按距离硬判 |
| 该不该合爪（`grasp_now`，noul） | 和距离无关 | — | 按距离硬判 |

- **口语化是成败的关键。** 第一版把偏差写成 `dx=+0.180 m` 喂给它，8 道方向题只对 1 道（12.5%）；改成「The apple is 18 centimetres to the LEFT of the gripper」后到 79%；再把幅度从问题里拿掉，到 87.5%。原因见 `state.py` 的模块文档。
- **幅度和门控是常数。** 扫一遍距离（9-23 实测，记在 `policy.py` 的注释里）：

  ```
  距离      30cm   20cm   12cm   8cm    5cm    3cm    1cm
  P(big)   0.212  0.232  0.221  0.191  0.114  0.200  0.204
  grasp    0.655  0.540  0.644  0.638  0.625  0.656  0.646
  ```

  距离变了 30 倍，输出不动。Laya 是非自回归的分类头，读不出数字大小。
- **闭环照样能完成。** 三个轴同时有偏差时，和解析解方向一致只有约 57%，但只要选的方向能减小某个轴的偏差，闭环就会收敛，只是路径不是最短（Laya 54 步，解析解 48 步）。
- **选目标也不行。** 「8 条换了说法的指令里选对哪个候选」，Laya 只对 3~4 条，而且是按选项位置在猜（`bench_select.py` 的模块文档，9-28）。现在选目标交给 CLIP。
- **判力度、判抓取位置也不行。** 回答和物体无关：把选项倒序，Laya 就跟着换答案，是按选项位置选的（`grip_force.py` 的模块文档，9-29）。现在这两件事交给 CLIP。

`run_loop` 那条链踩过的坑（细节在各文件的模块文档里）：

1. **相机后端互相污染。** 同一台 Orbbec 上 OBSENSOR 和 V4L2 不能交替用，切换之间要复位 USB 再等 8 秒（等 3 秒不够），`camera.py` 打开前会自动做。见 `depth_probe.py`。腕部 305 现在的情况更糟，见开头。
2. **粗档步长要小于 `max_lead`（15mm），不是小于 `max_step_translation`（20mm）。** 取 18mm 时每一步都被缰绳限速，不报错，只在 `env.stats['leashed']` 里看得见。现在取 12mm（`actions.py` 的 `COARSE_M`）。
3. **yolo11n 在俯视桌面上认不出东西**（水瓶 conf 0.07，还误检成 cup），默认改成 yolo11s。见 `detector.py`。
4. **COCO 没有 plate。** 说「盘子」会被映射成 `bowl`，`Task.notes` 里会写明。见 `parser.py`。
5. **SVD 求刚体变换必须查行列式**，否则噪声下会解出镜像矩阵。见 `calibrate.py` 的 `kabsch()`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [JEV_GRASP_SUMMARY.md](JEV_GRASP_SUMMARY.md) | 抓取流程（`run_composite`、`click_grasp`）的完整说明：每个数从哪来、标定、命令、实测记录、已知问题；附录是设计时踩过的坑 |
| [SETUP_5070Ti.md](SETUP_5070Ti.md) | `laya` 环境：版本、重建、权重、测试 |
| [PlayJev/SETUP_5070Ti.md](PlayJev/SETUP_5070Ti.md) | `playjev` 环境：版本、重建、起服务、测试 |
| [上游来源.md](上游来源.md)、[PlayJev/上游来源.md](PlayJev/上游来源.md) | 上游仓库、提交号、改过什么 |
| [README_laya.md](README_laya.md) | 上游 Laya 的说明 |
| [PlayJev/README.md](PlayJev/README.md) | 上游 PlayJev 的说明 |
| [../RobotDecision](../RobotDecision) | 10-12 以后的决策方向（自动抓取闭环） |
| [../../../工作区说明.md](../../../工作区说明.md) | 整个工作区：结构、环境、约定、离线测试、迁移 |
