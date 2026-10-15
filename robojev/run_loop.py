#!/usr/bin/env python3
"""RoboJev 闭环：一句指令 -> 感知 -> 特权信息 -> Laya 选动作 -> 机械臂执行 -> 再看。

    # mock 臂 + 真相机，最安全的那档
    python -m robojev.run_loop --instruction "把苹果放到盘子里"

    # 换成解析策略跑同样的流程，用来判断问题出在模型还是在模型下面
    python -m robojev.run_loop --instruction "把苹果放到盘子里" --policy geometric

    # 两个都算，执行 laya 的，逐步记符合率
    python -m robojev.run_loop --instruction "..." --policy laya --compare

⚠️ 默认 `--mock`。要驱动真臂必须显式 `--real`，而且先把外参标定做完
（`python -m robojev.calibrate`），否则臂会朝一个和目标无关的方向稳定地走。

**相机后端冲突**：本闭环用 OBSENSOR 独占 Orbbec（同时取彩色+深度），所以
mock 环境里不要再让 ar5 的 OpenCVCameras 去开 Orbbec 的 :1.4 —— 两个后端
互相污染，见 `robojev/depth_probe.py`。这里已经把 env 的相机关掉了。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))       # 让 atom_robot 可导入

from . import actions as A                                          # noqa: E402
from .camera import OrbbecRGBD                                      # noqa: E402
from .color_detector import ColorDetector                            # noqa: E402
from .detector import Detector                                      # noqa: E402
from .localizer import (DepthLocalizer, Extrinsics, PlaneLocalizer,  # noqa: E402
                        load_intrinsics)
from .parser import ParseError, parse                               # noqa: E402
from .policy import GeometricPolicy, LayaPolicy, compare            # noqa: E402
from .state import (PHASE_APPROACH, PHASE_CARRY, PHASE_DONE,        # noqa: E402
                    PHASE_LIFT, Percept, WorldState)

logger = logging.getLogger("robojev")

# 抓住之后先直上抬多少步。这一段不交给模型：抓起来立刻横向移动会把物体
# 在桌面上拖行甚至带翻，而「抓完先抬起来」没有任何需要判断的地方。
LIFT_STEPS = 6


def build_env(mock: bool):
    """起 AR5 环境。相机交给我们自己的 OrbbecRGBD，这里把 env 的相机关掉。"""
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    cfg = AR5Config()
    cfg.mock = mock
    cfg.real_cameras_in_mock = False        # 关键：别让它去抢 Orbbec 的 v4l2 节点
    cfg.cameras.sources = {}                # 观测里的图像我们不用，省一份拷贝
    env = AR5Env(cfg)
    env.connect()
    return env


class RealPerception:
    """真相机 + 检测器 + 3D 定位。

    检测器可以是 YOLO（`detector.Detector`）也可以是颜色分割
    （`color_detector.ColorDetector`），两者的 `detect()` 都返回同构的 Detection，
    但**挑「哪一个」的规则不同**：YOLO 按类名过滤再取置信度最高的，颜色分割只有
    一个类、要按面积取最大的。所以这里不写死 `Detector.best`，让每个检测器
    自己给。写死过一次，结果是颜色那条路永远返回 None —— 因为 `cls_name`
    是 'blue' 而 task.target 也是 'blue' 时碰巧能过，换个颜色就静默失效。
    """

    def __init__(self, cam, localizer, task, target_det, container_det=None):
        self.cam, self.localizer, self.task = cam, localizer, task
        self.target_det = target_det
        self.container_det = container_det

    def _one(self, det_obj, name, frame):
        if det_obj is None:
            return Percept(seen=False, cls_name=name)
        dets = det_obj.detect(frame.color, keep=[name])
        d = det_obj.best(dets, name)
        if d is None:
            return Percept(seen=False, cls_name=name)
        xyz = self.localizer.locate(d, frame)
        return Percept(seen=True, cls_name=name, conf=d.conf, xyz_base=xyz, pixel=d.centre)

    def look(self):
        frame = self.cam.read()
        if not frame.ok:
            return None, None
        target = self._one(self.target_det, self.task.target, frame)
        container = (self._one(self.container_det, self.task.container, frame)
                     if self.task.needs_place else Percept())
        return target, container

    def close(self):
        self.cam.disconnect()


class SimPerception:
    """不碰相机，直接给两个固定的基座系坐标。

    存在的理由是：状态机、策略、`env.step` 这条链必须能在没有相机、没有标定、
    臂也不上电的情况下整体跑一遍。否则「闭环不动」会同时有五个嫌疑人，
    而这条路能一次性把感知那三层摘出去。位置取在实测可达范围内。
    """

    def __init__(self, task, target_xyz, container_xyz):
        self.task = task
        self.t = np.asarray(target_xyz, dtype=np.float64)
        self.c = np.asarray(container_xyz, dtype=np.float64)
        self.grabbed = False

    def look(self):
        target = Percept(seen=True, cls_name=self.task.target, conf=0.99, xyz_base=self.t)
        container = (Percept(seen=True, cls_name=self.task.container, conf=0.99,
                             xyz_base=self.c) if self.task.needs_place else Percept())
        return target, container

    def close(self):
        pass


def run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(message)s")

    # ---------- 1. 指令 + 检测器 ----------
    # 颜色目标绕过 COCO：「蓝色圆柱体」这类东西 YOLO 里根本没有对应的类，
    # 走 parse() 只会报「类别表里没有」。给了 --target-color 就直接构造 Task。
    target_det = container_det = None
    if args.target_color:
        from .parser import Task

        task = Task(instruction=args.instruction, target=args.target_color,
                    container=args.container_color, target_said=args.target_color,
                    container_said=args.container_color or "",
                    notes=["目标按颜色分割，不走 YOLO"])
        if not args.sim:
            target_det = ColorDetector(args.target_color, min_area_px=args.min_area)
            if args.container_color:
                container_det = ColorDetector(args.container_color, min_area_px=args.min_area)
    else:
        # sim 模式不起 YOLO，所以也拿不到类别表，跳过类别校验。
        yolo = None if args.sim else Detector(conf=args.conf)
        try:
            task = parse(args.instruction,
                         coco_names=None if yolo is None else yolo.class_names)
        except ParseError as exc:
            print(f"\n指令解析失败：\n{exc}")
            return 2
        target_det = container_det = yolo
    print(f"\n指令 : {task.instruction}")
    how = "颜色分割" if args.target_color else "YOLO"
    print(f"目标 : {task.target_said} -> {how} '{task.target}'")
    if task.needs_place:
        print(f"容器 : {task.container_said} -> {how} '{task.container}'")
    for n in task.notes:
        print(f"  ⚠ {n}")

    # ---------- 2. 感知（真相机 or 合成） ----------
    if args.sim:
        percep = SimPerception(task, args.sim_target, args.sim_container)
        print(f"感知 : SIM  目标@{args.sim_target}  容器@{args.sim_container}（不碰相机）")
    else:
        intr = load_intrinsics()
        extr = Extrinsics.load()
        if args.localizer == "plane":
            localizer = PlaneLocalizer(intr=intr, extr=extr, table_z=args.table_z)
        else:
            localizer = DepthLocalizer(intr=intr, extr=extr)
        cam = OrbbecRGBD()
        cam.connect()
        percep = RealPerception(cam, localizer, task, target_det, container_det)
        print(f"定位 : {type(localizer).__name__}  "
              f"外参{'已标定' if extr.calibrated else ' **未标定（坐标是错的）**'}")
        print(f"相机 : Orbbec {cam.fps:.1f} Hz")

    # ---------- 3. 策略 ----------
    geo = GeometricPolicy()
    if args.policy == "laya":
        policy = LayaPolicy(single_question=args.single_question,
                            trust_laya_scale=args.trust_laya_scale,
                            trust_laya_gates=args.trust_laya_gates)
    else:
        policy = geo
    print(f"策略 : {policy.name}" + ("  (同时算 geometric 做对照)" if args.compare else ""))

    # ---------- 4. 机械臂 ----------
    env = build_env(mock=not args.real)
    print(f"机械臂: {'**真机**' if args.real else 'mock'}\n")

    phase = PHASE_APPROACH
    holding = False
    gripper = 0.0
    lift_left = 0
    records = []
    t_start = time.perf_counter()

    try:
        for step in range(args.max_steps):
            target, container = percep.look()
            if target is None:
                logger.warning("相机没给出成对的帧，跳过这一步")
                time.sleep(0.05)
                continue

            obs = env.get_observation(t=step)
            ee = np.asarray(obs.observation["ee_pose"], dtype=np.float64)[:3]

            ws = WorldState(task=task, phase=phase, ee_xyz=ee, gripper=gripper,
                            holding=holding, target=target, container=container, step=step)

            # --- 抬起阶段是写死的，不问模型 ---
            if phase == PHASE_LIFT:
                # 用和粗档同一个步长，别另写一个数 —— 它同样受 max_lead 约束，
                # 写大了这一段就会被缰绳限速。
                act = A.lookup("up", "big").to_delta(gripper=1.0)
                env.step(act)
                lift_left -= 1
                print(f"[{step:3d}] LIFT  剩 {lift_left} 步")
                if lift_left <= 0:
                    phase = PHASE_CARRY if task.needs_place else PHASE_DONE
                continue

            # --- 决策 ---
            d = policy.decide(ws)
            cmp_info = None
            if args.compare and policy is not geo:
                cmp_info = compare(d, geo.decide(ws))

            dist = ws.distance
            dist_s = "  ?  " if dist is None else f"{dist * 100:5.1f}cm"
            line = (f"[{step:3d}] {phase:17s} d={dist_s} "
                    f"{d.kind:7s} {d.action.name if d.action else '-':14s} {d.why}")
            if cmp_info:
                line += f"   [geo={cmp_info['geometric']} {'=' if cmp_info['agreed'] else 'X'}]"
            print(line)

            records.append({
                "step": step, "phase": phase, "numeric": ws.to_laya_state()["numeric"],
                "body": ws.to_laya_state()["body"],
                "decision": {"kind": d.kind, "action": d.action.name if d.action else None,
                             "why": d.why, "confidence": d.confidence},
                "compare": cmp_info,
            })

            # --- 执行 ---
            env.step(d.to_env_action())

            # --- 状态机 ---
            if d.kind == "grasp":
                gripper, holding = 1.0, True
                phase, lift_left = PHASE_LIFT, LIFT_STEPS
                print(f"      -> 抓住了，进入 LIFT（直上 {LIFT_STEPS} 步）")
            elif d.kind == "release":
                gripper, holding = 0.0, False
                phase = PHASE_DONE
                print("      -> 松爪，任务结束")

            if phase == PHASE_DONE:
                break
            time.sleep(args.period)

    finally:
        percep.close()
        env.close()

    el = time.perf_counter() - t_start
    print(f"\n跑了 {len(records)} 步 / {el:.1f}s，结束阶段 = {phase}")
    print(f"env 统计: {env.stats}")

    if args.compare:
        c = [r["compare"] for r in records if r["compare"]]
        if c:
            agree = sum(x["agreed"] for x in c)
            samedir = sum(x["same_direction"] for x in c)
            print(f"与解析解符合率: 完全一致 {agree}/{len(c)} ({100 * agree / len(c):.0f}%)，"
                  f"方向一致 {samedir}/{len(c)} ({100 * samedir / len(c):.0f}%)")

    if args.log:
        Path(args.log).write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"轨迹写到 {args.log}（以后微调 Laya 就用这个）")
    return 0 if phase == PHASE_DONE else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instruction", required=True, help="例如「把苹果放到盘子里」")
    p.add_argument("--policy", choices=["laya", "geometric"], default="laya")
    p.add_argument("--localizer", choices=["depth", "plane"], default="depth")
    p.add_argument("--table-z", type=float, default=0.10,
                   help="plane 方案下桌面在基座系的高度（米）")
    p.add_argument("--single-question", action="store_true",
                   help="一问 12 选 1（confidence 未标定，只作对照）")
    p.add_argument("--trust-laya-scale", action="store_true",
                   help="幅度也交给 Laya。实测它与距离无关，会一直选 small")
    p.add_argument("--trust-laya-gates", action="store_true",
                   help="抓/放门控也交给 Laya。实测 noul 与距离无关，会在半空合爪")
    p.add_argument("--compare", action="store_true", help="同时算 geometric 并统计符合率")
    p.add_argument("--real", action="store_true", help="驱动真臂。默认 mock")
    p.add_argument("--sim", action="store_true",
                   help="不碰相机，用合成的目标位置跑整条状态机+策略+执行链")
    p.add_argument("--sim-target", type=float, nargs=3, default=[0.50, 0.12, 0.18],
                   metavar=("X", "Y", "Z"), help="--sim 时目标物在基座系的位置")
    p.add_argument("--sim-container", type=float, nargs=3, default=[0.45, -0.15, 0.22],
                   metavar=("X", "Y", "Z"), help="--sim 时容器在基座系的位置")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--period", type=float, default=0.10, help="控制周期，秒")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度门限")
    p.add_argument("--target-color", default="",
                   help="按颜色找目标而不是用 YOLO：blue/yellow/green/red/orange。"
                        "COCO 里没有的东西（胶带卷、积木、圆柱块）只能走这条")
    p.add_argument("--container-color", default="", help="容器也按颜色找")
    p.add_argument("--min-area", type=int, default=300, help="颜色连通块最小面积（像素）")
    p.add_argument("--log", default="", help="把轨迹写成 json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
