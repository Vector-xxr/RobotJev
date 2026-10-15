#!/usr/bin/env python3
"""把手里的东西原地放回桌面，张爪，抬起，再慢速回 0 位。

    python -m robojev.place_and_home --place-z 0.18
    python -m robojev.place_and_home --no-place          # 手里没东西，直接张爪回 0 位

「0 位」= collect 的映射中心 `centre_joints`（atom_robot/ar5/joint_calibration.json），
主臂全零时从臂该在的姿势，也是遥操/数采每条起步的地方。**不是** `config.arm.home_joints`
（那是工作空间盒中心，两者 j4 差 45°）。

为什么先放下再张爪：抓完停在离桌面约 7cm 的空中，直接张爪套筒会掉下去；带着回 0 位再张
就是从 30 多厘米高处掉。所以先竖直降回抓取点上方 5mm（`--place-z`，指尖高度），再张。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))

from . import actions as A                               # noqa: E402
from .move_pregrasp import slow_goto                     # noqa: E402
from .run_playjev import GripperXYZ, settle              # noqa: E402

CENTRE = Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101" / "atom_robot" / "ar5" / "joint_calibration.json"


def hold_gripper(env, value: float, seconds: float) -> None:
    """原地保持，同时把夹爪指令稳定发出去；每 50ms 一帧，RT 不会断粮。"""
    t_end = time.time() + seconds
    while time.time() < t_end:
        env.step(A.hold(value))
        time.sleep(0.05)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place-z", type=float, default=0.18, help="放下时的指尖高度（米）")
    ap.add_argument("--no-place", action="store_true", help="不下降，直接张爪回 0 位")
    ap.add_argument("--down", type=float, default=None,
                    help="相对当前位置竖直下降多少（米）再张爪。抬了多少就降多少，不需要知道桌面/指尖高度"
                         "（指尖偏移示范定不住，用绝对高度 --place-z 不可靠）")
    ap.add_argument("--lift", type=float, default=0.07, help="张爪后抬起多少（米）")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    centre = np.asarray(json.loads(CENTRE.read_text())["centre_joints"], float)

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True
    env = AR5Env(cfg)
    env.connect()
    try:
        grip0 = float(env.hand.get_opening())
        settle(env, gripper=grip0)            # 连上先稳住，夹爪保持现状（别在空中松开）
        tip = GripperXYZ(env, mock=False)
        g = tip()
        print(f"当前指尖 {np.round(g, 3).tolist()}   夹爪 {grip0:.2f}（1=合拢）")

        if a.down is not None:
            steps = int(round(a.down / A.COARSE_M))
            for _ in range(steps):
                env.step(np.array([0, 0, -A.COARSE_M, 0, 0, 0, grip0]))
                settle(env, gripper=grip0)
            print(f"相对下降 {steps * A.COARSE_M * 100:.1f}cm")
        elif not a.no_place:
            n = 0
            while g[2] > a.place_z + 0.002 and n < 30:
                dz = max(-A.COARSE_M, a.place_z - g[2])
                env.step(np.array([0, 0, dz, 0, 0, 0, grip0]))
                settle(env, gripper=grip0)
                g = tip()
                n += 1
            print(f"放下到指尖 z={g[2]:.3f}（{n} 步）")

        hold_gripper(env, 0.0, 1.5)           # 张开要 0.5s，多等一点
        print(f"张爪，夹爪 {env.hand.get_opening():.2f}")

        steps = int(round(a.lift / A.COARSE_M))
        for _ in range(steps):
            env.step(A.lookup("up", "big").to_delta(gripper=0.0))
            settle(env, gripper=0.0)
        print(f"抬起 {steps * A.COARSE_M * 100:.0f}cm，指尖 {np.round(tip(), 3).tolist()}")

        reached, el = slow_goto(env, centre)
        q = np.asarray(env.arm.get_joint_positions(), float)
        print(f"回 0 位: 到位={reached} 用时 {el:.1f}s  关节(度) {np.round(np.rad2deg(q), 2).tolist()}  "
              f"和 0 位最大差 {np.rad2deg(np.max(np.abs(q - centre))):.2f}°")
        print(f"运动错误: {env.arm.has_motion_error()}  env 统计 unreachable={env.stats['unreachable']}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
