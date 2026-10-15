# RobotJev：用 Laya 做机器人闭环决策

一句指令 → YOLO 感知 → 特权信息 → Laya 从 12 个离散动作里选 → AR5 执行 → 再看一帧。

```
指令 "把苹果放到盘子里"
   │
   ├─ parser.py      → target='apple'  container='bowl'
   │
   ├─ camera.py      → Orbbec RGBD（OBSENSOR 后端，彩色+深度同源 30 Hz）
   ├─ detector.py    → YOLO 框
   ├─ localizer.py   → 像素+深度 →(内参)→ 相机系 →(外参)→ 基座系 xyz
   │
   ├─ state.py       → 特权信息，口语化成 Laya 能读的一段话
   ├─ policy.py      → Laya 选方向 / 距离定幅度 / 距离定抓放
   ├─ actions.py     → 12 个动作 → env.step() 的 7 维 delta
   │
   └─ run_loop.py    → 状态机：接近 → 抓 → 抬 → 搬 → 放
```

## 跑

```bash
conda activate laya
cd 算法/Jev/RobotJev

# 0) 深度流验证（已通过，留着复跑用）
python -m robojev.depth_probe

# 1) 不碰硬件，验证状态机+策略+执行链
python -m robojev.run_loop --instruction "把苹果放到盘子里" --sim --policy geometric
python -m robojev.run_loop --instruction "把苹果放到盘子里" --sim --policy laya --compare

# 2) 量 Laya 在这个动作空间上到底行不行
python -m robojev.bench_policy

# 3) 标定（外参必须动真臂，不标的话闭环坐标是错的）
python -m robojev.calibrate --intrinsics
python -m robojev.calibrate --extrinsics

# 4) 真相机 + mock 臂
python -m robojev.run_loop --instruction "把苹果放到盘子里" --compare

# 5) 真机（先做完 3）
python -m robojev.run_loop --instruction "把苹果放到盘子里" --real
```

## Laya 在这里能干什么、不能干什么

`bench_policy.py` 的实测（24 道方向题，判据是解析解）：

| 问的东西 | 表现 | 随机基线 | 用不用 |
|---|---|---|---|
| **方向**（6 选 1） | **87.5%** | 16.7% | ✅ 交给 Laya |
| 幅度（2 选 1） | 0% | 50% | ❌ 按距离硬判 |
| grasp_now（noul） | 与距离无关 | — | ❌ 按距离硬判 |

第一版用 `dx=+0.180 m` 这种写法喂它，8 道方向题只对 1 道（12.5%）。改成
「The apple is 18 centimetres to the LEFT of the gripper」之后跳到 79%，
再把幅度从问题里摘掉后到 87.5%。**这个改写是整件事能成立的关键**，
原因见 `state.py` 的模块文档。

幅度和门控为什么不能交给它 —— 扫一遍距离就明白，它们是常数：

```
距离      30cm   20cm   12cm   8cm    5cm    3cm    1cm
P(big)   0.212  0.232  0.221  0.191  0.114  0.200  0.204
grasp    0.655  0.540  0.644  0.638  0.625  0.656  0.646
```

距离变了 30 倍，输出没动。Laya 是非自回归分类头，读不了数字大小。

闭环里（三轴偏差同时存在，比 bench 的单轴难）与解析解符合率约 **58%**，
但任务照样完成 —— 只要选的方向能减小某个轴的偏差，闭环就会收敛，
只是路径不是最优的（laya 55 步 vs 解析解 48 步）。

## 踩过的坑

1. **相机后端互相污染**。OBSENSOR 和 v4l2 在同一台 Orbbec 上不能交替用，
   切换之间要 USB 复位 + **等 8 秒**（等 3 秒不够）。`camera.py` 里已经自动做了。
   详见 `depth_probe.py` 模块文档。

2. **粗档步长必须 < `max_lead`(15mm)**，不是 < `max_step_translation`(20mm)。
   取 18mm 时 `env.stats['leashed']` 是 46/50 —— 每一步都被缰绳限速，不报错，
   只在统计里看得见。现在取 12mm，leashed=0。

3. **yolo11n 在俯视桌面上认不出东西**（水瓶 conf 0.07，还误检成 cup）。
   默认已改成 yolo11s。见 `detector.py`。

4. **COCO 没有 plate**。说「盘子」会被映射到 `bowl`，`Task.notes` 里会写明。

5. **SVD 求刚体变换必须查 det**，否则噪声下会解出镜像矩阵。`calibrate.py:kabsch()`。

## 还没做

- **外参标定**（`calibrate.py --extrinsics` 写好了但没跑过，要动真臂）。
  没标定之前 `localizer` 出来的坐标是错的，闭环会朝固定方向走。
- **相机朝向**：Orbbec 原来对着零件架，需要重新对准工作台。
- 真机闭环（要先做完上面两条）。
- 拿 `--log` 存下来的轨迹微调 Laya，让它接管幅度和门控（仓库根目录有
  `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`）。
