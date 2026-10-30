#!/usr/bin/env python3
"""⚠️ 实测这个判据在预抓取高度是**错的**，别在那个高度用（2026-09-28）。

在指尖上方 7cm 处，套筒偏两指中线 -52px，按它修正把臂推偏了 34mm。对照成功示范
（pingpong/episode_0028）同一时刻，套筒在腕部画面是 (311,422)，而单应性对准后是
(304,417) —— 本来就是对的。原因是透视：目标在指尖下方，投影不在手指那个深度上，
「目标在两指中线上」只在**抓取高度**才成立。示范统计也显示预抓取时这个偏差本来就
在 -40~+95px 之间。要用的话只能在下探到抓取高度之后做。

下探前用腕部相机做横向微调：把目标挪到两根手指的中线上。

    python -m robojev.wrist_refine --color blue            # 标定 + 微调
    python -m robojev.wrist_refine --color blue --measure  # 只看不动

为什么要这一步：俯视单应性的留一法误差中位 5.9mm、p90 12mm，而这次对准完在腕部
画面里套筒偏了两指中线约 40px（套筒直径的 1/3，1.5~2cm）。夹爪开口 98mm、套筒外径
估计 55mm，每边余量只有约 2cm —— 偏差到了边缘上，下探时一根手指可能压到套筒壁。

腕部相机跟着夹爪走，所以不需要任何标定就能说「目标在不在两指之间」。
缺的只是「像素往哪边动 <-> 臂往哪边动」，这里用两次 10mm 点动现场量出来（图像雅可比），
再用它把目标像素的 x 挪到两指中线上。只修横向（图像 x）：手指是左右合拢的，
图像 y（沿接近方向）的误差不影响夹不夹得住，而且下探时它本来就会变。

每一步都走 env.step + settle，受全部安全钳位保护；单次修正上限 15mm。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "算法" / "DexGraspVLA-" / "SO101"))

from .run_playjev import settle  # noqa: E402

JOG_M = 0.010
MAX_FIX_M = 0.015


def finger_midline(color_bgr: np.ndarray, rows=(400, 470)) -> float | None:
    """画面底部两根黑色手指的内侧边缘中点（x）。手指是深色大块，贴着左右下角。"""
    import cv2

    g = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    band = g[rows[0]:rows[1]]
    dark = ((band < 45) * 255).astype(np.uint8)
    # 只认**贴着画面左/右边缘**的暗块。第一版按列统计暗像素，套筒中间那个孔也是暗的，
    # 被当成了右手指，中线算成 234（肉眼约 345）。
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    w = band.shape[1]
    left_edge = right_edge = None
    for k in range(1, n):
        x, _y, ww, _h, area = st[k]
        if area < 200:
            continue
        if x == 0:
            left_edge = max(left_edge or 0, x + ww - 1)
        if x + ww >= w:
            right_edge = min(right_edge or w, x)
    if left_edge is None or right_edge is None:
        return None
    return 0.5 * (left_edge + right_edge)


def target_px(det, color_bgr):
    """最大的那块目标颜色。深蓝色的手指也会被 blue 命中，所以排除贴着画面左右边缘的块。"""
    w = color_bgr.shape[1]
    ds = [d for d in det.detect(color_bgr) if d.xyxy[0] > 20 and d.xyxy[2] < w - 20]
    return ds[0] if ds else None


def measure(cam, det, n=8):
    us, vs, mids = [], [], []
    for _ in range(n * 3):
        f = cam.read()
        if f.ok:
            d = target_px(det, f.color)
            m = finger_midline(f.color)
            if d is not None:
                us.append(d.centre[0]); vs.append(d.centre[1])
            if m is not None:
                mids.append(m)
        if len(us) >= n:
            break
        time.sleep(0.04)
    if len(us) < 3:
        return None
    return float(np.median(us)), float(np.median(vs)), (float(np.median(mids)) if mids else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", default="blue")
    ap.add_argument("--measure", action="store_true", help="只测量不动")
    ap.add_argument("--tol-px", type=float, default=8.0)
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    from .camera import OrbbecRGBD
    from .color_detector import ColorDetector

    det = ColorDetector(a.color)
    cam = OrbbecRGBD()
    cam.connect()                     # 先开相机（含 USB 复位），再连臂：连臂后不能长时间不发目标
    time.sleep(1.0)
    m0 = measure(cam, det)
    if m0 is None:
        print("腕部画面里没看到目标")
        cam.disconnect()
        return 1
    print(f"目标像素 ({m0[0]:.0f},{m0[1]:.0f})   两指中线 x={m0[2]}   横向偏差 "
          f"{(m0[0] - m0[2]) if m0[2] else float('nan'):+.0f}px")
    if a.measure:
        cam.disconnect()
        return 0

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True
    env = AR5Env(cfg)
    env.connect()
    settle(env, gripper=0.0)
    try:
        # --- 图像雅可比：两次点动 ---
        J = np.zeros((2, 2))
        for k, (axis_name, d) in enumerate([("y", np.array([0, -JOG_M, 0])), ("x", np.array([JOG_M, 0, 0]))]):
            env.step(np.r_[d, 0, 0, 0, 0.0]); settle(env, 0.0); time.sleep(0.3)
            mk = measure(cam, det)
            env.step(np.r_[-d, 0, 0, 0, 0.0]); settle(env, 0.0); time.sleep(0.3)
            if mk is None:
                print(f"点动 {axis_name} 后看不到目标，停")
                return 2
            J[:, k] = (np.array(mk[:2]) - np.array(m0[:2])) / JOG_M
            print(f"  点动 {'-y' if axis_name == 'y' else '+x'} {JOG_M * 1000:.0f}mm -> 像素移动 "
                  f"({mk[0] - m0[0]:+.1f},{mk[1] - m0[1]:+.1f})")
        print(f"图像雅可比(px/m) 列=[-y, +x]:\n{np.round(J, 0)}")

        # --- 横向修正：只修图像 x，用对图像 x 最敏感的那个基座方向 ---
        for it in range(4):
            m = measure(cam, det)
            if m is None or m[2] is None:
                print("测不到目标或手指中线，停")
                return 3
            err = m[0] - m[2]
            print(f"[{it}] 目标 x={m[0]:.0f} 中线={m[2]:.0f} 偏差 {err:+.0f}px")
            if abs(err) <= a.tol_px:
                print("横向已对准")
                return 0
            # 最小范数解：只要求图像 x 变化 -err，图像 y 不管
            jx = J[0]                          # d(px_x)/d[-y, +x]
            step = -err * jx / float(jx @ jx)  # 在 [-y, +x] 坐标里
            delta = np.array([step[1], -step[0], 0.0])
            nrm = np.linalg.norm(delta)
            if nrm > MAX_FIX_M:
                delta *= MAX_FIX_M / nrm
            print(f"     修正 基座 dx={delta[0] * 1000:+.1f}mm dy={delta[1] * 1000:+.1f}mm")
            env.step(np.r_[delta, 0, 0, 0, 0.0]); settle(env, 0.0); time.sleep(0.3)
        print("4 次修正后仍未对准")
        return 4
    finally:
        env.close()
        cam.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
