#!/usr/bin/env python3
"""夹爪几何自标定：用腕部 RGB-D 看自己的两根手指，量出抓取要用的夹爪参数。约 10s，只动夹爪不动臂。

    python -m robojev.calibrate_gripper              # 在当前姿态量（要求手指后面的背景比 13cm 远，0 位即可）
    python -m robojev.run_composite ... --calib-gripper   # 抓取开始时在 0 位量一次

量什么（结果写 calibration/gripper.json）：
  full_open_m   夹爪读数 r 时两指内缘间距 w，拟合 w = full_open·(1−r)。手册 98mm，实测 99mm
  root_cam      手指内侧面最靠近相机那端（相机系）。夹持中心往上到这里的竖直距离 = 能容纳物体顶部多高，
                即抓顶部时伸进去多深（GRASP_BAND）。竖直方向要用固定朝向下的 J，所以存点、运行时再算
  tip_cam       指尖（相机系）。实测指尖基本就在示范给的夹持中心高度（低 0~3mm）
  grasp_centre_cam  夹持中心 G（相机系）：夹着乒乓球时球心在哪（--centre，球面拟合）。伺服把物体送到这里。
                没量过就用示范测的 [15,32,88]mm（旧夹爪）

9/30 换夹爪（同型号）后：根部和旧的差 <1.3mm，指尖远了 ~12mm（手指长了），夹住的球心
[9,29,93]mm（旧 G 差 7.7mm）；指尖在 G 下方 24mm（旧 11.7mm）—— 所以「离支撑面下限」也要跟着标定走（finger_floor）。
⚠️ 腕部 305 断过电后深度会和彩色错开约 (100, 30) 像素（硬件对齐丢了，OpenCV 后端开不回来，
   每次报 setXu ioctl error）：先跑一次 read_orbbec_params（pyorbbecsdk 开一次流）恢复，再标定。

    python -m robojev.calibrate_gripper --centre     # 夹着乒乓球（抓起来之后）量夹持中心

手指是黑的，表面大多没有深度，只有边缘有；够量位置。根部再往上超出深度最近距离（~53mm），
所以 root_cam 是下限：真实根部只会更靠相机，按它算的 GRASP_BAND 偏保守。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "算法" / "DexGraspVLA-" / "SO101"))

from . import actions as A                               # noqa: E402

CALIB = Path(__file__).resolve().parent / "calibration"
OUT = CALIB / "gripper.json"
FINGER_NEAR_M = 0.13                 # 比这近的点算手指（背景要比它远）
LEVELS = (0.0, 0.5, 0.75)            # 0.0 量满开；0.5/0.75 手指整根都在画面里，量根部和指尖


def _depth_median(env, cam, target, n=8):
    ds = []
    while len(ds) < n:
        f = cam.read()
        if f.ok and f.depth is not None:
            ds.append(f.depth.astype(float))
        env.step(A.hold(target)); time.sleep(0.05)
    D = np.stack(ds); D[D == 0] = np.nan
    return np.nan_to_num(np.nanmedian(D, axis=0)).astype(np.uint16)


def _fingers(cm, depth, split_x):
    """两根手指的点，按相机 x 分左右。"""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    P = cm.points_keep(depth, xs.ravel().astype(float), ys.ravel().astype(float))
    Q = P[np.isfinite(P[:, 2]) & (P[:, 2] < FINGER_NEAR_M)]
    return Q[Q[:, 0] < split_x], Q[Q[:, 0] >= split_x]


def measure(env, cam, cm, grasp_centre):
    rows, roots, tips = [], [], []
    for target in LEVELS:
        t0 = time.time()
        while time.time() - t0 < 2.2:                     # 合到位要 1.6s
            env.step(A.hold(target)); time.sleep(0.05)
        r = float(env.hand.get_opening())
        L, R = _fingers(cm, _depth_median(env, cam, target), grasp_centre[0])
        if len(L) < 50 or len(R) < 50:
            print(f"   读数 {r:.2f}: 手指点太少（左 {len(L)} 右 {len(R)}），跳过")
            continue
        gap = float(np.percentile(R[:, 0], 2) - np.percentile(L[:, 0], 98))
        rows.append((r, gap))
        line = f"   读数 {r:.2f}: 两指内缘间距 {gap * 1000:.0f}mm"
        if target > 0:
            for S in (L, R):
                c = S.mean(0)
                a = np.linalg.svd(S - c, full_matrices=False)[2][0]
                a = a if a[2] > 0 else -a                  # 指向指尖（远离相机）
                s = (S - c) @ a
                roots.append(c + np.percentile(s, 2) * a)
                tips.append(c + np.percentile(s, 98) * a)
        print(line)
    hold_open(env)
    if len(rows) < 2 or not roots:
        raise RuntimeError("手指点不够，量不出来（背景是否比 13cm 远？）")
    r, w = np.array(rows).T
    k = 1.0 - r
    full_open = float((k @ w) / (k @ k))                  # 过原点最小二乘 w = full_open·(1−r)
    resid = float(np.max(np.abs(w - full_open * k)))
    if resid > 0.006:
        # 手指是黑的，只有边缘有深度，某一档边缘缺一块时间距会偏（9/28 一次 0.75 档量成 5mm，应 ~20mm）
        print(f"   满开拟合残差 {resid * 1000:.0f}mm 太大，不采用（{full_open * 1000:.0f}mm），保留手册值")
        full_open = None
    res = dict(load() or {})                                  # 保留夹持中心等别的标定项
    res.update(full_open_m=full_open, root_cam=np.mean(roots, 0).tolist(), tip_cam=np.mean(tips, 0).tolist(),
               samples=[dict(reading=float(a), gap_m=float(b)) for a, b in rows],
               date=time.strftime("%Y-%m-%d %H:%M"))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"   满开 {'手册值' if full_open is None else f'{full_open * 1000:.0f}mm'}；手指根部(相机系) {np.round(np.array(res['root_cam']) * 1000).astype(int).tolist()}"
          f" 指尖 {np.round(np.array(res['tip_cam']) * 1000).astype(int).tolist()} mm -> {OUT.name}")
    return res


def hold_open(env, s=1.5):
    t0 = time.time()
    while time.time() - t0 < s:
        env.step(A.hold(0.0)); time.sleep(0.05)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else None


DEFAULT_CENTRE = np.array([0.0148, 0.0323, 0.0882])    # 旧夹爪：示范里抓住球那一刻球心的位置


def grasp_centre():
    """夹持中心 G（相机系，米）：标定过（--centre）就用标定值，否则用示范测的旧值。"""
    g = load()
    return np.asarray(g["grasp_centre_cam"], float) if g and g.get("grasp_centre_cam") else DEFAULT_CENTRE.copy()


def finger_floor(J, G=None, margin=0.002, lo=0.012):
    """两指中间最低能离支撑面多高：指尖在夹持中心下方多深（沿竖直，竖直方向 = −J[:,2]）+ 余量。
    旧夹爪 11.7mm -> 12mm；9/30 换的手指长了，指尖在 G 下方 24mm -> 26mm。没标定就用 lo。"""
    g = load()
    if not g or not g.get("tip_cam"):
        return lo
    G = grasp_centre() if G is None else np.asarray(G, float)
    up = -np.asarray(J, float)[:, 2]
    up = up / np.linalg.norm(up)
    depth = -float((np.asarray(g["tip_cam"], float) - G) @ up)
    return max(lo, depth + margin)


def apply_calibration(J):
    """运行前把夹爪标定落实到各处：返回 G，并设 grip_force.FINGER_FLOOR_M。"""
    from . import grip_force

    G = grasp_centre()
    grip_force.FINGER_FLOOR_M = finger_floor(J, G)
    return G


def measure_centre(cam, n=8):
    """夹着乒乓球（抓起来、悬在空中）时：腕部看球，球面拟合（r=20mm）求球心 = 夹持中心 G，写 gripper.json。"""
    from .calibrate_top335 import sphere_fit
    from .objects import TargetFinder

    tf = TargetFinder("pick up the orange ping pong ball")
    cs, rs = [], []
    for _ in range(5 * n):
        f = cam.read()
        if not (f.ok and f.depth is not None):
            continue
        cands = tf._candidates(f.color)
        if not cands:
            continue
        i, _, _ = tf._pick(f.color, cands, view="wrist")
        if i is None:
            continue
        ys, xs = np.nonzero(tf._segment(f.color, cands[i].xyxy))
        P = tf.cam.points(f.depth, xs.astype(float), ys.astype(float))
        if len(P) < 50:
            continue
        P = P[np.abs(P[:, 2] - np.median(P[:, 2])) < 0.02]      # 漏进掩码的手指 / 背景不要
        cs.append(sphere_fit(P, r=0.020)[0]); rs.append(sphere_fit(P)[1])
        if len(cs) >= n:
            break
    if len(cs) < 3:
        raise RuntimeError("腕部认不出夹着的乒乓球（先把球抓起来，再量）")
    G = np.median(cs, axis=0)
    res = dict(load() or {})
    res.update(grasp_centre_cam=G.tolist(), grasp_centre_spread_m=np.std(cs, axis=0).tolist(),
               grasp_centre_how=f"夹着乒乓球，腕部球面拟合 r=20mm，{len(cs)} 帧（自由半径 {np.median(rs) * 1000:.1f}mm）",
               grasp_centre_date=time.strftime("%Y-%m-%d %H:%M"))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"   夹持中心 G（相机系）{np.round(G * 1000, 1).tolist()}mm，帧间散布 {np.round(np.std(cs, axis=0) * 1000, 1).tolist()}mm"
          f"（原来 {np.round(DEFAULT_CENTRE * 1000, 1).tolist()}）-> {OUT.name}")
    return G


def band_from(res, grasp_centre, J):
    """夹持中心往上到手指根部的竖直距离（米）。基座 +z 在相机系里是 −J[:,2]。"""
    return float((np.asarray(res["root_cam"]) - grasp_centre) @ (-J[:, 2]))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--centre", action="store_true",
                    help="量夹持中心：要先把乒乓球抓起来悬在空中。只开腕部相机、不连臂（连臂会先张爪，球就掉了）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    from .camera import OrbbecRGBD
    from .objects import CameraModel
    from .run_composite import GRASP_CENTER_CAM, connect_arm

    cam = OrbbecRGBD(); cam.connect()
    if a.centre:
        try:
            measure_centre(cam)
        finally:
            cam.disconnect()
        return 0
    env = connect_arm()
    try:
        measure(env, cam, CameraModel(), GRASP_CENTER_CAM)
    finally:
        env.close(); cam.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
