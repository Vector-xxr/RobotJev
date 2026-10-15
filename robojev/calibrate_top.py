#!/usr/bin/env python3
"""俯视相机 ↔ 臂 自标定。不用示范数据，全自动，约 3 分钟。

    python -m robojev.calibrate_top --dry-run     # 只算逆解、打印计划，不动臂
    python -m robojev.calibrate_top               # 真跑，结果写 calibration/top_calib.json

俯视相机没有深度，一个像素只对应一条视线，所以必须知道「相机 ↔ 基座」的关系。做法：

1. 臂保持固定朝向 R0，法兰走 2 层 × 3×3 共 18 个点。
2. 每个点让夹爪开、合各一次；俯视画面里只有手指在动，**变化像素的质心 = 手指在画面里的位置**。
   不用识别模型，也不用贴标记。
3. 18 对（法兰位置, 手指像素）拟合 3×4 投影矩阵 P（DLT）。
   朝向固定时，「手指点 = 法兰 + 常量偏移」，偏移被吸收进 P，不需要知道手指离法兰多远。
4. 桌面高度：腕部 RGB-D 在上下两层中心点各测一次「夹持中心到桌面的距离 d」，
   z_touch = 法兰 z − d（夹持中心贴到桌面时法兰的 z）。两层应一致，是自检。
5. 腕部相机运动映射 J：固定朝向下是常量，这里测一次存下，用来算「沿光轴后退」。

固定朝向 R0 取自 pregrasp_pose.json 那组关节角（之前几次成功抓取都在这个朝向，腕部相机斜向下看桌面）。
只在标定时读一次，存进结果文件；抓取流程运行时不再读任何示范。
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

CALIB = Path(__file__).resolve().parent / "calibration"
OUT = CALIB / "top_calib.json"

# 网格：法兰位置。覆盖泡沫板上 x 0.31~0.51、y -0.50~-0.26 那片（指尖位置）。
# 第一版 x 小 5cm、上层 0.41：18 点里 10 点的变化区域顶到画面上沿，质心被切偏（重投影中位 11.6px）。
# 下层手指离泡沫板约 8cm，高于桌上的物体（套筒 4.7cm）。
GRID_X = (0.283, 0.383, 0.483)
GRID_Y = (-0.377, -0.257, -0.137)
GRID_Z = (0.33, 0.39)
GRIP_WAIT_S = 1.8                      # 合爪 1.6s
DIFF_THRESH = 25                       # 灰度差阈值
MIN_AREA, MAX_AREA = 300, 40000        # 手指变化区域面积（像素）


def grid():
    """蛇形走：同层相邻点只差一格；上层按下层倒序走，换层是原地竖直抬起。
    （按行扫描时，从一行末到下一行首关节变化大，关节插值的路径会偏离直线。）"""
    layer = []
    for j, y in enumerate(GRID_Y):
        xs = GRID_X if j % 2 == 0 else GRID_X[::-1]
        layer += [(x, y) for x in xs]
    pts = [np.array([x, y, GRID_Z[0]]) for x, y in layer]
    pts += [np.array([x, y, GRID_Z[1]]) for x, y in layer[::-1]]
    return pts


def path_min_z(kin, qa, qb, n=20):
    """关节线性插值 qa->qb 途中法兰的最低 z（slow_goto 走的就是关节插值）。"""
    return min(kin.fk(qa + (qb - qa) * s)[2, 3] for s in np.linspace(0, 1, n))


def plan(kin, R0, q_seed):
    """逐点解逆解（上一个点的解当种子，关节走得连续）。解不出返回 None。"""
    qs, q = [], np.asarray(q_seed, float)
    for p in grid():
        T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = p
        sol, info = kin.ik(T, q)
        qs.append(sol)
        if sol is not None:
            q = sol
    return qs


def median_frame(cam, n=5):
    fs = []
    for _ in range(n):
        fs.append(cam.read().color.astype(np.float32))
        time.sleep(0.04)
    return np.median(np.stack(fs), axis=0).astype(np.uint8)


def finger_pixel(img_open, img_closed):
    """开/合两帧的差：只有手指在动。返回 (质心像素, 面积, 掩码)。"""
    import cv2

    d = cv2.absdiff(cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_closed, cv2.COLOR_BGR2GRAY))
    d = cv2.GaussianBlur(d, (5, 5), 0)
    m = (d > DIFF_THRESH).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 80:
            keep[lab == i] = 1
    area = int(keep.sum())
    if area == 0:
        return None, 0, keep
    ys, xs = np.nonzero(keep)
    return np.array([xs.mean(), ys.mean()]), area, keep


def hold_grip(env, value, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        env.step(A.hold(value))
        time.sleep(0.05)


def dlt(X, u):
    """X (N,3) 基座点，u (N,2) 像素 -> 3x4 P。Hartley 归一化。"""
    def norm(pts):
        c = pts.mean(0); s = np.sqrt(pts.shape[1]) / np.mean(np.linalg.norm(pts - c, axis=1))
        T = np.eye(pts.shape[1] + 1); T[:-1, :-1] *= s; T[:-1, -1] = -s * c
        return T
    Tx, Tu = norm(X), norm(u)
    Xh = (Tx @ np.c_[X, np.ones(len(X))].T).T
    uh = (Tu @ np.c_[u, np.ones(len(u))].T).T
    rows = []
    for (x, y, z, w), (a, b, c) in zip(Xh, uh):
        p = np.array([x, y, z, w])
        rows.append(np.r_[np.zeros(4), -c * p, b * p])
        rows.append(np.r_[c * p, np.zeros(4), -a * p])
    P = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 4)
    P = np.linalg.inv(Tu) @ P @ Tx
    return P / np.linalg.norm(P[2, :3])


def project(P, X):
    h = (P @ np.c_[X, np.ones(len(X))].T).T
    return h[:, :2] / h[:, 2:3]


def table_distance(cam, cm):
    """腕部深度整幅 RANSAC 拟合桌面，返回夹持中心到桌面的距离（米）和内点比例。"""
    from .run_composite import FINGER_NEAR_M, GRASP_CENTER_CAM, _frame

    _, depth = _frame(cam)
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    Q = cm.points(depth, xs.ravel(), ys.ravel())
    Q = Q[Q[:, 2] > FINGER_NEAR_M]
    rng = np.random.default_rng(0)
    best_k, best = 0, None
    for _ in range(400):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        k = int((np.abs((Q - a) @ n) < 0.004).sum())
        if k > best_k:
            best_k, best = k, (a, n)
    a, n = best
    inl = Q[np.abs((Q - a) @ n) < 0.004]
    c = inl.mean(0); n = np.linalg.svd(inl - c, full_matrices=False)[2][2]
    return float(abs((GRASP_CENTER_CAM - c) @ n)), best_k / len(Q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="只连控制器读 DH、解逆解，不动臂")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    import warnings
    warnings.filterwarnings("ignore")
    from atom_robot.ar5.kinematics import AR5Kinematics

    q_ref = np.asarray(json.loads((CALIB / "pregrasp_pose.json").read_text())["q"], float)

    if a.dry_run:
        from atom_robot.ar5.config import AR5Config
        from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
        arm = RokaeAR5(AR5Config().arm)
        arm.connect(power_on=False)
        kin = AR5Kinematics.from_robot(arm)
        R0 = kin.fk(q_ref)[:3, :3]
        qs = plan(kin, R0, q_ref)
        q_prev = np.asarray(arm.get_joint_positions(), float)
        worst = np.inf
        for p, q in zip(grid(), qs):
            if q is None:
                print(f"法兰 {np.round(p, 3).tolist()}  逆解失败"); continue
            zmin = path_min_z(kin, q_prev, q)
            worst = min(worst, zmin - p[2])
            print(f"法兰 {np.round(p, 3).tolist()}  逆解 OK  途中法兰最低 z {zmin:.3f}（比目标低 {max(0, p[2] - zmin) * 1000:.0f}mm）")
            q_prev = q
        print(f"可达 {sum(q is not None for q in qs)}/{len(qs)}；途中最多比目标层低 {max(0, -worst) * 1000:.0f}mm")
        return 0

    import cv2
    from .camera import OrbbecRGBD, TopCamera
    from .move_pregrasp import slow_goto
    from .objects import CameraModel
    from .run_composite import KeepAlive, connect_arm, measure_jacobian

    wrist = OrbbecRGBD(); wrist.connect()           # 先开腕部相机（含 USB 复位），再连臂
    top = TopCamera(); top.connect()
    env = connect_arm()
    ka = KeepAlive(env).__enter__()
    kin = AR5Kinematics.from_robot(env.arm)
    R0 = kin.fk(q_ref)[:3, :3]
    qs = plan(kin, R0, q_ref)
    dbg = CALIB / "top_calib_debug"
    dbg.mkdir(exist_ok=True)
    X, U, rec = [], [], []
    try:
        time.sleep(1.0)
        for k, (p, q) in enumerate(zip(grid(), qs)):
            if q is None:
                print(f"[{k:2d}] 法兰 {np.round(p, 3).tolist()} 逆解失败，跳过")
                continue
            reached, _ = slow_goto(env, q)
            time.sleep(0.4)
            img_o = median_frame(top)
            hold_grip(env, 1.0, GRIP_WAIT_S); time.sleep(0.2)
            img_c = median_frame(top)
            hold_grip(env, 0.0, GRIP_WAIT_S)
            px, area, m = finger_pixel(img_o, img_c)
            f = kin.fk(np.asarray(env.arm.get_joint_positions(), float))[:3, 3]
            vis = img_c.copy(); vis[m > 0] = (0, 0, 255)
            if px is not None:
                cv2.circle(vis, tuple(int(v) for v in px), 6, (0, 255, 0), 2)
            cv2.imwrite(str(dbg / f"p{k:02d}.png"), vis)
            h, w = m.shape
            border = bool(m[:4].any() or m[-4:].any() or m[:, :4].any() or m[:, -4:].any())
            ok = px is not None and MIN_AREA <= area <= MAX_AREA and not border   # 碰边 = 被切掉一块，质心偏
            print(f"[{k:2d}] 法兰 {np.round(f, 3).tolist()}  手指像素 "
                  f"{'无' if px is None else np.round(px).astype(int).tolist()}  面积 {area}{'  碰边' if border else ''}  {'✓' if ok else '✗'}")
            if ok:
                X.append(f); U.append(px); rec.append(dict(flange=f.tolist(), px=px.tolist(), area=area))

        if len(X) < 8:
            raise SystemExit(f"有效点只有 {len(X)} 个，不够拟合（至少 8 个，要跨两层）")
        X, U = np.array(X), np.array(U)
        P = dlt(X, U)
        err = np.linalg.norm(project(P, X) - U, axis=1)
        # 剔一次大残差（手指被臂挡住半边时质心会偏）
        good = err < max(3.0, 3 * np.median(err))
        if good.sum() < len(X):
            P = dlt(X[good], U[good])
            err = np.linalg.norm(project(P, X) - U, axis=1)
        print(f"投影矩阵重投影误差：中位 {np.median(err[good]):.1f}px 最大 {err[good].max():.1f}px（用 {good.sum()}/{len(X)} 点）")

        # 桌面高度 + J：上下两层中心点
        cm = CameraModel()
        z_touch = []
        for zi, z in enumerate(GRID_Z):
            p = np.array([GRID_X[1], GRID_Y[1], z])
            q = qs[GRID_Z.index(z) * 9 + 4]
            slow_goto(env, q); time.sleep(0.6)
            d, frac = table_distance(wrist, cm)
            fz = kin.fk(np.asarray(env.arm.get_joint_positions(), float))[2, 3]
            z_touch.append(fz - d)
            print(f"   层 {zi}: 法兰 z {fz:.3f}  夹持中心离桌面 {d * 1000:.0f}mm（内点 {frac:.0%}） -> z_touch {fz - d:.3f}")
        if abs(z_touch[0] - z_touch[1]) > 0.01:
            print(f"   ⚠ 两层 z_touch 差 {abs(z_touch[0] - z_touch[1]) * 1000:.0f}mm（>10mm），桌面测量可疑")
        print("测腕部相机运动映射 J（固定朝向下是常量）:")
        J = measure_jacobian(env, wrist, cm)

        OUT.write_text(json.dumps(dict(
            P=P.tolist(), q_ref=q_ref.tolist(), R0=R0.tolist(), z_touch=float(np.mean(z_touch)),
            z_touch_layers=z_touch, J=J.tolist(), reproj_px_median=float(np.median(err[good])),
            points=rec, date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
        print(f"已写 {OUT}")
        slow_goto(env, np.asarray(json.loads((Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101" / "atom_robot" / "ar5"
                                              / "joint_calibration.json").read_text())["centre_joints"], float))
        print("已回 0 位")
    finally:
        ka.__exit__(None, None, None)
        env.close(); top.disconnect(); wrist.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
