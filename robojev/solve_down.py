#!/usr/bin/env python3
"""求一个「夹爪竖直朝下」的关节姿态。只算，不动臂。

    python -m robojev.solve_down --xyz 0.45 0.0 0.30
    python -m robojev.solve_down --xyz 0.45 0.0 0.30 --tilt 5   # 允许偏竖直 5°

第一次用 DLS 数值 IK 从当前姿态出发，12 个 yaw × 2 个种子全军覆没，而且没说
是哪条过滤卡的。这里每个候选都报原因，并且额外把**种子扫开**：7 轴臂同一个末端
位姿有一整族解，DLS 只会收敛到离种子最近的那个，种子离得远（当前姿态夹爪是横着的，
要转约 90°）就可能掉进限位里出不来。

判据（全部要过）：
  - IK 收敛
  - 位置误差 < 1 mm，z 轴偏竖直 < --tilt
  - 每个关节离软限位 >= joint_limit_margin
  - 最低点高于 joint_floor_z（`AR5Env._lowest_point`，和 goto_joints 用的是同一个检查）
选中的解写到 calibration/q_down.json，给 move_down 用。
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "算法" / "DexGraspVLA-" / "SO101"))

OUT = Path(__file__).resolve().parent / "calibration" / "q_down.json"


def target_T(xyz, yaw_rad, tilt_rad=0.0, tilt_dir=0.0):
    z = np.array([np.sin(tilt_rad) * np.cos(tilt_dir), np.sin(tilt_rad) * np.sin(tilt_dir),
                  -np.cos(tilt_rad)])
    x0 = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    x = x0 - z * (x0 @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, xyz
    return T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xyz", type=float, nargs=3, default=[0.45, 0.0, 0.30])
    ap.add_argument("--tilt", type=float, default=0.5, help="允许偏离竖直多少度")
    ap.add_argument("--seeds", type=int, default=40, help="随机种子个数（外加当前姿态和 home）")
    a = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.kinematics import AR5Kinematics

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    env.connect()
    try:
        kin = AR5Kinematics.from_robot(env.arm)
        q0 = np.asarray(env.arm.get_joint_positions(), float)
        jmin, jmax = np.asarray(cfg.arm.joint_min), np.asarray(cfg.arm.joint_max)
        lo, hi = jmin + cfg.arm.joint_limit_margin, jmax - cfg.arm.joint_limit_margin
        P = np.asarray(a.xyz, float)
        rng = np.random.default_rng(0)
        seeds = [("当前", q0), ("home", np.asarray(cfg.arm.home_joints, float))]
        for i in range(a.seeds):
            seeds.append((f"随机{i}", rng.uniform(lo * 0.8, hi * 0.8)))

        why = Counter()
        limit_joint = Counter()
        cands = []
        tilt_max = np.deg2rad(a.tilt)
        tilts = [0.0] if a.tilt <= 0.5 else [0.0, tilt_max * 0.5, tilt_max]
        for yaw_deg, (sname, seed), tilt in itertools.product(range(0, 360, 30), seeds, tilts):
            for tdir in ([0.0] if tilt == 0 else np.deg2rad([0, 90, 180, 270])):
                T = target_T(P, np.deg2rad(yaw_deg), tilt, tdir)
                q, info = kin.ik(T, seed, max_iters=800)
                if q is None:
                    why["IK 不收敛"] += 1
                    continue
                Tq = kin.fk(q)
                perr = np.linalg.norm(Tq[:3, 3] - P) * 1000
                zerr = np.rad2deg(np.arccos(np.clip(-Tq[2, 2], -1, 1)))
                if perr > 1.0 or zerr > a.tilt + 0.1:
                    why[f"精度不够"] += 1
                    continue
                bad = [j for j in range(7) if q[j] < lo[j] or q[j] > hi[j]]
                if bad:
                    why["超关节限位"] += 1
                    for j in bad:
                        limit_joint[f"j{j + 1}"] += 1
                    continue
                low = env._lowest_point(q)
                if low is not None and low < cfg.arm.joint_floor_z:
                    why["最低点低于地板"] += 1
                    continue
                margin = float(np.min(np.minimum(q - lo, hi - q)))
                cands.append(dict(q=q, yaw=yaw_deg, seed=sname, tilt=float(zerr), perr=float(perr),
                                  margin=margin, dmax=float(np.max(np.abs(q - q0))),
                                  low=low, man=float(kin.manipulability(q))))
                why["可行"] += 1
    finally:
        env.close()

    print(f"目标 xyz={P.tolist()}  允许偏竖直 {a.tilt}°  种子 {len(seeds)} 个")
    print("过滤统计:", dict(why))
    if limit_joint:
        print("  超限的关节:", dict(limit_joint))
    if not cands:
        print("\n没有可行解。")
        return 1

    # 偏好：离限位远 > 关节总改变量小 > 远离奇异
    cands.sort(key=lambda c: (-min(c["margin"], np.deg2rad(15)), c["dmax"], -c["man"]))
    b = cands[0]
    print(f"\n可行解 {len(cands)} 个，选中：yaw={b['yaw']}° 种子={b['seed']} 偏竖直 {b['tilt']:.2f}°")
    print(f"  关节(度)   {np.round(np.rad2deg(b['q']), 1).tolist()}")
    print(f"  当前(度)   {np.round(np.rad2deg(q0), 1).tolist()}")
    print(f"  最大改变 {np.rad2deg(b['dmax']):.1f}°  离限位最近 {np.rad2deg(b['margin']):.1f}°  "
          f"最低点 z={b['low']:.3f} m  可操作度 {b['man']:.4f}  位置误差 {b['perr']:.3f} mm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"q": b["q"].tolist(), "q_from": q0.tolist(), "xyz": P.tolist(),
                               "yaw_deg": b["yaw"], "tilt_deg": b["tilt"]}, indent=2))
    print(f"写到 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
