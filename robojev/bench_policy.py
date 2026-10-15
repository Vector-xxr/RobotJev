#!/usr/bin/env python3
"""不碰硬件，用造出来的偏差量 Laya 的方向决策对不对。

    python -m robojev.bench_policy
    python -m robojev.bench_policy --single-question   # 一问 12 选 1 的对照

判据是解析解（`GeometricPolicy`）：偏差最大的轴往减小它的方向走，这是闭式的、
不会错的。Laya 和它一致的比例就是「Laya 在这个动作空间上能不能用」的直接答案。

分两级看：
  方向一致 —— 轴和正负都对（12 选 1 里的 6 选 1 部分）
  完全一致 —— 连粗细档也对

方向一致率的随机基线是 1/6 = 17%。低于这个说明模型是在帮倒忙。
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from .parser import parse
from .policy import GeometricPolicy, LayaPolicy
from .state import PHASE_APPROACH, PHASE_CARRY, Percept, WorldState

EE = np.array([0.40, 0.0, 0.30])       # 夹爪固定在工作空间中间，只动目标


def cases():
    """覆盖单轴、双轴、远近的偏差。"""
    out = []
    for axis, sign in itertools.product([0, 1, 2], [+1, -1]):
        for mag in (0.18, 0.06, 0.02):       # 远 / 中 / 近
            off = np.zeros(3)
            off[axis] = sign * mag
            out.append(off)
    # 双轴混合：主轴明显更大，解析解仍然是确定的
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        for sa, sb in itertools.product([+1, -1], repeat=2):
            off = np.zeros(3)
            off[a] = sa * 0.15
            off[b] = sb * 0.04
            out.append(off)
    return out


def make_ws(task, offset, phase=PHASE_APPROACH, holding=False):
    goal = Percept(seen=True, cls_name=task.target, conf=0.9, xyz_base=EE + offset)
    ws = WorldState(task=task, phase=phase, ee_xyz=EE, gripper=0.0, holding=holding,
                    target=goal if phase == PHASE_APPROACH else Percept(),
                    container=goal if phase == PHASE_CARRY else Percept())
    return ws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-question", action="store_true")
    ap.add_argument("--show", type=int, default=6, help="打印前几条的详情")
    args = ap.parse_args()

    task = parse("把苹果放到盘子里")
    geo = GeometricPolicy()
    laya = LayaPolicy(single_question=args.single_question)

    n = same_dir = same_all = 0
    confs = []
    print(f"提问方式: {'一问 12 选 1' if args.single_question else '两问 6+2'}\n")
    for i, off in enumerate(cases()):
        ws = make_ws(task, off)
        g = geo.decide(ws)
        l = laya.decide(ws)
        if g.action is None:          # 已在抓取半径内，不算方向题
            continue
        n += 1
        dir_ok = (l.action is not None and l.action.axis == g.action.axis
                  and l.action.sign == g.action.sign)
        all_ok = (l.action is not None and l.action.name == g.action.name)
        same_dir += dir_ok
        same_all += all_ok
        confs.append(l.confidence)
        if i < args.show:
            print(f"  off={np.round(off, 3)}  期望 {g.action.name:14s} "
                  f"得到 {(l.action.name if l.action else l.kind):14s} "
                  f"{'方向✓' if dir_ok else '方向✗'}{'/全✓' if all_ok else ''}  conf={l.confidence:.2f}")

    print(f"\n{n} 道方向题")
    print(f"  方向一致  {same_dir:3d}/{n}  {100 * same_dir / n:5.1f}%   (随机基线 16.7%)")
    print(f"  完全一致  {same_all:3d}/{n}  {100 * same_all / n:5.1f}%   (随机基线  8.3%)")
    print(f"  平均 confidence {np.mean(confs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
