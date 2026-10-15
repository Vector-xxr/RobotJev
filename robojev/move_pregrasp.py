#!/usr/bin/env python3
"""把臂慢慢移到预抓取姿态（来自示范），到位后报指尖位置。

    python -m robojev.move_pregrasp                   # 用 calibration/pregrasp_pose.json
    python -m robojev.move_pregrasp --dry-run         # 只打印要走的关节差，不动

**姿态从哪来**：示范里真正抓起过乒乓球的片段，挑抓取点离当前目标最近的那条，
取它合爪前 2 秒、在抓取点正上方的那一帧关节角。人实际到过的姿态，比现解 IK 可靠 ——
这台臂竖直朝下根本解不出来（j6/j7 只有 ±50°），而示范里所有抓取都是斜 50~60° 的。

**为什么不直接 goto_joints**：它每 50ms 走 `joint_jump_max*0.5 = 0.075 rad`，约 86°/s
的指令速度，对「第一次去一个新姿态」太快。这里自己插值，每 50ms 走 STEP_RAD，
最后剩零点几度再交给 goto_joints 做保持和锚点同步（它绕过 step_joints 会让限速器
的锚点失步，必须由它自己来同步，见 ar5_env.goto_joints 里的注释）。
不改 joint_jump_max：它和 joint_goal_step 有联动校验，改小了会让 RT 发送线程拒指令。

碰撞检测**只在这次运行里**打开（config 里是 suppleStop + 最不灵敏 0.01，触发不掉电）。
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

POSE = Path(__file__).resolve().parent / "calibration" / "pregrasp_pose.json"
HOMO = Path(__file__).resolve().parent / "calibration" / "homography.json"
STEP_RAD = 0.01          # 每 50ms，约 11°/s（第一次去新姿态，宁慢）


def tip_of(kin, q, t):
    T = kin.fk(q)
    return T[:3, 3] + T[:3, :3] @ t, T


def slow_goto(env, q_goal, step_rad: float = STEP_RAD, timeout_s: float = 60.0):
    """慢速关节插值到 q_goal，最后交给 goto_joints 保持并同步锚点。返回 (到位, 用时)。

    控制器报运动错误就停下并抛 RuntimeError。每 50ms 发一次目标，不会让 RT 锁存。
    """
    q_goal = np.asarray(q_goal, float)
    if hasattr(env.arm, "clear_latch"):
        env.arm.clear_latch()
    t_start = time.time()
    while time.time() - t_start < timeout_s:
        cur = np.asarray(env.arm.get_joint_positions(), float)
        err = q_goal - cur
        if np.max(np.abs(err)) < np.deg2rad(0.5):
            break
        scale = min(1.0, step_rad / max(np.max(np.abs(err)), 1e-9))
        env.arm.servo_to_joints(cur + err * scale)
        if env.arm.has_motion_error():
            raise RuntimeError("控制器报运动错误，已停下")
        time.sleep(0.05)
    reached = env.goto_joints(q_goal, timeout_s=10.0)
    return reached, time.time() - t_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default=str(POSE))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.kinematics import AR5Kinematics

    pose = json.loads(Path(a.pose).read_text())
    q_goal = np.asarray(pose["q"], float)
    t = np.asarray(json.loads(HOMO.read_text())["tip_offset_flange"], float)

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True
    env = AR5Env(cfg)
    env.connect()
    try:
        kin = AR5Kinematics.from_robot(env.arm)
        q0 = np.asarray(env.arm.get_joint_positions(), float)
        tip0, _ = tip_of(kin, q0, t)
        tipg, _ = tip_of(kin, q_goal, t)
        d = np.rad2deg(q_goal - q0)
        print(f"来源: {pose['src']} 帧 {pose['frame']}")
        print(f"当前关节(度) {np.round(np.rad2deg(q0), 1).tolist()}   指尖 {np.round(tip0, 3).tolist()}")
        print(f"目标关节(度) {np.round(np.rad2deg(q_goal), 1).tolist()}   指尖 {np.round(tipg, 3).tolist()}")
        print(f"关节差(度)   {np.round(d, 1).tolist()}   最大 {np.abs(d).max():.1f}°  "
              f"预计 {np.abs(q_goal - q0).max() / STEP_RAD * 0.05:.1f}s")
        if a.dry_run:
            return 0

        try:
            reached, elapsed = slow_goto(env, q_goal)
        except RuntimeError as exc:
            print(exc)
            return 3
        q1 = np.asarray(env.arm.get_joint_positions(), float)
        tip1, T1 = tip_of(kin, q1, t)
        print(f"\n到位={reached}  用时 {elapsed:.1f}s")
        print(f"实际关节(度) {np.round(np.rad2deg(q1), 1).tolist()}  和目标最大差 "
              f"{np.rad2deg(np.max(np.abs(q1 - q_goal))):.2f}°")
        print(f"指尖 {np.round(tip1, 4).tolist()}   法兰 {np.round(T1[:3, 3], 4).tolist()}   "
              f"夹爪轴 {np.round(T1[:3, 2], 3).tolist()}")
        print(f"运动错误: {env.arm.has_motion_error()}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
