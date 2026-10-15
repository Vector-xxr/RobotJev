"""手指扫掠体碰撞检查：规划好抓取点之后，两根手指伸进来、合拢的路上有没有别的东西（盘沿、碗壁、旁边的物体）。

原来只查「抓取点离支撑面 ≥ 12mm」，只防指尖戳到正下方；高出来的东西挡在手指路上完全看不见。

在腕部这一帧里做，不用先动臂：朝向固定时臂只平移，把夹持中心 G 挪到抓取点 p，场景点在相机系里
整体平移 (G − p)。手指在相机系里的位置是固定的（calibrate_gripper 量的根部/指尖），所以：
  1. 取这一帧所有深度点，去掉物体自己的（掩码膨胀几像素）和比 13cm 近的（手指自己）
  2. 平移 (G − p)
  3. 两根手指各一个盒子：开合方向 x 在预张开位置外侧、厚度 FINGER_T；y、z 覆盖根部到指尖；
     z 再往相机方向延长 APPROACH_M —— 手指是沿光轴从后面伸进来的，扫过的是一条长盒子
  4. 落进盒子的点 = 会碰到的东西

**支撑面本身不算障碍**（给了 support 时）：离支撑面 SUPPORT_BAND_M 以内、以及更低的点都去掉。
指尖在夹持中心下方 11.8mm，抓取高度下限 12mm 时指尖正好贴着支撑面 —— 碰不碰桌由那条下限管，这里只查
支撑面以上的东西。9/29 桌上乒乓球：「中」「下」两个位姿报 232/653 个碰撞点，全在球两侧的泡沫板上
（离支撑面 −1.6~+1.1mm），于是「每个位姿都会碰，不抓」。
"""

from __future__ import annotations

import numpy as np

FINGER_T = 0.017       # 手指厚度（沿开合方向）：9/29 夹爪自标定，开度 0.5 时每根手指的点云在相机 x 上跨 ~17mm
APPROACH_M = 0.10      # 手指沿光轴伸进来的距离（= 预抓取点后退的距离）
PRE_GAP_M = 0.012      # 合爪前预张开：比抓取处宽这么多（和 grip_force.PRE_GAP_M 一致）
MIN_HITS = 30          # 这么多点落进手指扫掠体才算会碰（深度噪声会有零星几个）


def finger_boxes(width, gripper):
    """两根手指的扫掠盒（相机系，米）：[(lo, hi), (lo, hi)]。"""
    root, tip = np.asarray(gripper["root_cam"]), np.asarray(gripper["tip_cam"])
    xc = 0.5 * (root[0] + tip[0])
    half = 0.5 * (width + PRE_GAP_M)
    y0, y1 = min(root[1], tip[1]) - 0.012, max(root[1], tip[1]) + 0.012
    z0, z1 = root[2] - APPROACH_M, tip[2] + 0.005
    left = (np.array([xc - half - FINGER_T, y0, z0]), np.array([xc - half, y1, z1]))
    right = (np.array([xc + half, y0, z0]), np.array([xc + half + FINGER_T, y1, z1]))
    return [left, right]


def check(cm, depth, mask, grasp_cam, G, width, gripper, up=None, support=None):
    """抓取点 grasp_cam（腕部相机系）会不会让手指碰到别的东西。返回 dict(hits, where, ok)。
    support = (支撑面上一点, 竖直向上单位向量)，相机系。"""
    import cv2

    from .objects import SUPPORT_BAND_M

    h, w = depth.shape
    ys, xs = np.mgrid[0:h:2, 0:w:2]
    keep = ~(cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[ys, xs]
    P = cm.points_keep(depth, xs[keep].astype(float), ys[keep].astype(float))
    P = P[np.isfinite(P[:, 0]) & (P[:, 2] > 0.13)]
    if support is not None:
        c0, n = np.asarray(support[0], float), np.asarray(support[1], float)
        P = P[(P - c0) @ n > SUPPORT_BAND_M]
    Pm = P + (np.asarray(G) - np.asarray(grasp_cam))
    hits, where = 0, []
    for name, (lo, hi) in zip(("左指", "右指"), finger_boxes(width, gripper)):
        inside = np.all((Pm >= lo) & (Pm <= hi), axis=1)
        k = int(inside.sum())
        hits += k
        if k:
            q = P[inside]
            desc = f"{name} {k} 点"
            if up is not None:
                desc += f"（障碍高出抓取点 {float(np.percentile((q - grasp_cam) @ up, 90)) * 1000:+.0f}mm）"
            where.append(desc)
    return dict(hits=hits, where=where, ok=hits < MIN_HITS)
