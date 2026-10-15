#!/usr/bin/env python3
"""不碰硬件，量 PlayJev 在「渲染成 Snake 的抓取状态」上能不能选对方向。

    python -m robojev.bench_snake --n 80

三个判据：
  1. 正确率：选的方向让曼哈顿距离变小。4 选 1、目标不在同一行/列时
     有 2 个方向是对的，所以随机基线是 50%（同行同列时是 25%），下面分开报。
  2. 左右镜像 / 3. 上下镜像：翻转后的画面上，选择仍然要在「翻转后的正确集合」里。
     不能要求「left 必须变成 right」—— 目标在斜对角时两个方向都对，模型镜像后
     换成另一个也对的方向是完全合理的。第一版就是这么写的，把正确行为判成了不自洽。
"""

from __future__ import annotations

import argparse

import numpy as np

from .probe_playjev import ask
from .snake_render import COLS, INSTRUCTIONS, OPTIONS, ROWS, draw

H_SWAP = {"left": "right", "right": "left", "up": "up", "down": "down"}
V_SWAP = {"up": "down", "down": "up", "left": "left", "right": "right"}
STEP = {"up": (0, -1), "down": (0, +1), "left": (-1, 0), "right": (+1, 0)}   # (dcol, drow)


def choose(img) -> tuple:
    r = ask(img, INSTRUCTIONS, OPTIONS)["answers"]["move"]
    return r["choice"], r["probabilities"]


def good_moves(head, food):
    dc, dr = food[0] - head[0], food[1] - head[1]
    out = set()
    if dc > 0: out.add("right")
    if dc < 0: out.add("left")
    if dr > 0: out.add("down")
    if dr < 0: out.add("up")
    return out


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    stats = {"diag": [0, 0], "line": [0, 0]}
    h_ok = v_ok = 0
    n = 0
    confs = []
    while n < a.n:
        head = (int(rng.integers(0, COLS)), int(rng.integers(0, ROWS)))
        food = (int(rng.integers(0, COLS)), int(rng.integers(0, ROWS)))
        if head == food:
            continue
        n += 1
        img = draw(head, food)
        c, p = choose(img)
        confs.append(max(p.values()))
        good = good_moves(head, food)
        kind = "diag" if len(good) == 2 else "line"
        stats[kind][0] += c in good
        stats[kind][1] += 1

        ch, _ = choose(cv2.flip(img, 1))
        cv, _ = choose(cv2.flip(img, 0))
        h_ok += ch in {H_SWAP[g] for g in good}
        v_ok += cv in {V_SWAP[g] for g in good}
        if n <= 6:
            print(f"  蛇头{head} 食物{food}  对的={sorted(good)}  选={c:5s} "
                  f"左右镜像->{ch:5s} 上下镜像->{cv:5s}  max_p={max(p.values()):.2f}")

    d_ok, d_n = stats["diag"]
    l_ok, l_n = stats["line"]
    print(f"\n{n} 个状态")
    print(f"  不同行不同列  {d_ok:3d}/{d_n:<3d} {100 * d_ok / max(d_n, 1):5.1f}%   (随机 50%)")
    print(f"  同行或同列    {l_ok:3d}/{l_n:<3d} {100 * l_ok / max(l_n, 1):5.1f}%   (随机 25%)")
    print(f"  左右镜像后仍对  {h_ok:3d}/{n}  {100 * h_ok / n:5.1f}%")
    print(f"  上下镜像后仍对  {v_ok:3d}/{n}  {100 * v_ok / n:5.1f}%")
    print(f"  平均最大概率  {np.mean(confs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
