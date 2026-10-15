#!/usr/bin/env python3
"""标定：内参（棋盘格）+ 手眼外参（点对点）。

    # 1) 内参：拿张棋盘格在相机前换姿势，按空格拍，拍够 15 张
    python -m robojev.calibrate --intrinsics

    # 2) 外参：臂带着夹爪走到若干位姿，每个位姿在画面上点一下夹爪尖
    python -m robojev.calibrate --extrinsics

外参标定**必须动真臂**，而且是唯一必须动真臂的一步。没有它，`localizer` 出来的
基座系坐标是错的，闭环会朝一个和目标无关的方向稳定地走。

**为什么用点对点而不是 AX=XB。**
相机是固定在架子上的（eye-to-hand），而且这条链路只需要**位置**：YOLO 给的是
物体中心，我们要的是它在基座系的 xyz，姿态全程不变。位置对位置就是一个刚体
配准问题，三点就能定，N 点用 SVD 求最小二乘解。AX=XB 那一套是给 eye-in-hand
或者需要完整位姿的场合的，在这里是多余的复杂度和多余的误差源。

⚠️ **SVD 解出来的 R 必须查行列式。** `U @ Vt` 在点有噪声时会解出 det=-1 的
**反射**矩阵而不是旋转矩阵。反射矩阵拿去用，表现是「某一个轴的方向是反的」，
而且怎么调别的参数都调不好。下面的 `kabsch()` 里有那行修正，别删。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .camera import Intrinsics, OrbbecRGBD
from .localizer import CALIB_DIR

logger = logging.getLogger(__name__)


def kabsch(cam_pts: np.ndarray, base_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """求 R,t 使 base ≈ R @ cam + t。返回 (R, t, RMS 残差)。"""
    P = np.asarray(cam_pts, dtype=np.float64).reshape(-1, 3)
    Q = np.asarray(base_pts, dtype=np.float64).reshape(-1, 3)
    if len(P) < 3:
        raise ValueError(f"至少要 3 个点才能定一个刚体变换，只给了 {len(P)} 个")

    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    # ↓ 这一行是防反射的。去掉它在有噪声时会解出 det=-1 的镜像矩阵，
    #   表现是某个轴整体反向，而且调别的参数永远调不回来。
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cQ - R @ cP
    resid = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    return R, t, float(np.sqrt((resid ** 2).mean()))


# ----------------------------------------------------------------- 内参

def calibrate_intrinsics(rows: int, cols: int, square_m: float, want: int) -> int:
    import cv2

    pattern = (cols, rows)                       # cv2 要 (宽, 高) 的内角点数
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    print(f"棋盘格内角点 {cols}x{rows}，格子 {square_m * 1000:.0f} mm")
    print("换姿势、换远近、让格子占满画面不同区域。空格=拍，q=结束\n")

    with OrbbecRGBD() as cam:
        while len(objpoints) < want:
            f = cam.read()
            if not f.ok:
                continue
            gray = cv2.cvtColor(f.color, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            vis = f.color.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)
            cv2.putText(vis, f"{len(objpoints)}/{want}  {'FOUND' if found else 'no board'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("intrinsics", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" ") and found:
                # 亚像素细化。不做的话角点只有整像素精度，fx 能差出百分之几。
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp.copy())
                imgpoints.append(corners)
                print(f"  拍到第 {len(objpoints)} 张")
        cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print(f"只有 {len(objpoints)} 张，太少，标不出来（至少 5 张，建议 15）")
        return 1

    h, w = 480, 640
    rms, K, dist, _rv, _tv = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    intr = Intrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]),
                      cx=float(K[0, 2]), cy=float(K[1, 2]), width=w, height=h)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIB_DIR / "intrinsics.json"
    out.write_text(json.dumps(intr.__dict__, indent=2))
    print(f"\n重投影 RMS = {rms:.3f} px  （> 1.0 就该重标）")
    print(f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
    print(f"写到 {out}")
    return 0


# ----------------------------------------------------------------- 外参

def calibrate_extrinsics(n_points: int) -> int:
    import cv2

    from .localizer import load_intrinsics

    intr = load_intrinsics()
    print("外参标定。流程：")
    print("  1. 你用示教/遥操把臂移到一个位姿，夹爪尖要在相机画面里看得见")
    print("  2. 在画面上**点一下夹爪尖**，程序读该点深度并记下当前臂位姿")
    print(f"  3. 换个位姿重复，至少 {n_points} 个点，尽量散开（别都在一个平面上）")
    print("  q = 结束并求解\n")

    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    env.connect()

    cam_pts: List[np.ndarray] = []
    base_pts: List[np.ndarray] = []
    click: dict = {}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (x, y)

    cv2.namedWindow("extrinsics")
    cv2.setMouseCallback("extrinsics", on_mouse)

    try:
        with OrbbecRGBD() as cam:
            while True:
                f = cam.read()
                if not f.ok:
                    continue
                vis = f.color.copy()
                cv2.putText(vis, f"{len(cam_pts)}/{n_points}  click the gripper tip",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for p in click.get("history", []):
                    cv2.circle(vis, p, 5, (0, 0, 255), -1)
                cv2.imshow("extrinsics", vis)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

                if "xy" in click:
                    u, v = click.pop("xy")
                    z = f.depth_at(u, v, patch=7)
                    if z is None:
                        print(f"  ({u},{v}) 没有有效深度，换个点再点一次")
                        continue
                    p_cam = intr.deproject(u, v, z)
                    p_base = np.asarray(env.arm.get_ee_pose(), dtype=np.float64)[:3]
                    cam_pts.append(p_cam)
                    base_pts.append(p_base)
                    click.setdefault("history", []).append((u, v))
                    print(f"  #{len(cam_pts)}  cam={np.round(p_cam, 3)}  base={np.round(p_base, 3)}")
        cv2.destroyAllWindows()
    finally:
        env.close()

    if len(cam_pts) < 3:
        print(f"只采到 {len(cam_pts)} 个点，至少要 3 个")
        return 1

    R, t, rms = kabsch(np.array(cam_pts), np.array(base_pts))
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIB_DIR / "extrinsics.json"
    out.write_text(json.dumps({"R": R.tolist(), "t": t.tolist(),
                               "n_points": len(cam_pts), "rms_m": rms}, indent=2))
    print(f"\n残差 RMS = {rms * 1000:.1f} mm   （> 10 mm 就别信，多采几个点重标）")
    print(f"det(R) = {np.linalg.det(R):.4f}   （必须 ≈ +1；是 -1 说明解成了镜像）")
    print(f"写到 {out}")
    return 0


def calibrate_homography(points_xy: List[Tuple[float, float]], z: float, color: str) -> int:
    """俯视相机像素 -> 桌面平面上的基座 xy。给 run_playjev 用。

    相机固定、目标都在桌面上，所以像素和桌面 xy 之间就是一个单应性（3x3），
    4 个点就能定，多几个点更稳。比 3D 外参简单一个量级，也不需要深度。

    流程：臂**逐点**移到 `points_xy`（每一步都要你按键确认），夹爪指尖进入俯视画面，
    然后你在画面上点一下指尖。记录（像素, FK 算的法兰 xy）。

    前提：**夹爪必须竖直朝下**。这样指尖正好在法兰正下方，点指尖 = 点法兰的 xy。
    夹爪斜着的话，指尖和法兰的 xy 差出一个夹爪长度，整张单应性都是偏的。
    `z` 取指尖离桌面 2~3 cm 的法兰高度：离桌面越高，视差越大。
    """
    import cv2

    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    from . import actions as A
    from .camera import TopCamera
    from .proprio import ProprioReader

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    env.connect()
    reader = ProprioReader(env)

    p0 = reader.read()
    if p0.z_axis[2] > -0.9:
        env.close()
        print(f"夹爪没有竖直朝下（z 轴 = {np.round(p0.z_axis, 3).tolist()}），先把它转成朝下再标。")
        return 2

    px, xy = [], []
    click: dict = {}

    def on_mouse(event, x, y, _f, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["uv"] = (x, y)

    cv2.namedWindow("homography")
    cv2.setMouseCallback("homography", on_mouse)
    try:
        with TopCamera() as cam:
            for i, (tx, ty) in enumerate(points_xy):
                print(f"\n点 {i + 1}/{len(points_xy)}：下一步把法兰移到 x={tx:.3f} y={ty:.3f} z={z:.3f}")
                if input("  回车 = 走，s = 跳过这个点，q = 结束：").strip().lower() == "q":
                    break
                # 小步走过去，每步不超过粗档，受 env.step 的全部安全钳位保护
                for _ in range(400):
                    g = reader.read().xyz
                    d = np.array([tx, ty, z]) - g
                    if np.linalg.norm(d) < 0.002:
                        break
                    d = d / max(np.linalg.norm(d) / A.COARSE_M, 1.0)
                    env.step(np.array([d[0], d[1], d[2], 0, 0, 0, 0.0]))
                time.sleep(0.5)
                g = reader.read().xyz
                print(f"  到位：FK xy = ({g[0]:.4f}, {g[1]:.4f})，现在在画面上点指尖")
                click.clear()
                while "uv" not in click:
                    vis = cam.read().color.copy()
                    for (u, v) in px:
                        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
                    cv2.imshow("homography", vis)
                    if (cv2.waitKey(20) & 0xFF) == ord("q"):
                        raise KeyboardInterrupt
                px.append(click["uv"])
                xy.append((g[0], g[1]))
                print(f"  记下 像素{click['uv']} <-> xy({g[0]:.4f}, {g[1]:.4f})")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()

    if len(px) < 4:
        print(f"只有 {len(px)} 个点，单应性至少要 4 个")
        return 1
    H, _ = cv2.findHomography(np.array(px, np.float64), np.array(xy, np.float64), 0)
    pred = cv2.perspectiveTransform(np.array(px, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pred - np.array(xy), axis=1) * 1000
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIB_DIR / "homography.json"
    out.write_text(json.dumps({"H": H.tolist(), "pixels": px, "xy": xy, "z": z,
                               "err_mm": err.tolist()}, indent=2))
    print(f"\n重投影误差 中位 {np.median(err):.1f} mm  最大 {err.max():.1f} mm   （> 10 mm 就重标）")
    print(f"写到 {out}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsics", action="store_true", help="标内参（棋盘格）")
    p.add_argument("--extrinsics", action="store_true", help="标手眼外参（要动真臂）")
    p.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数")
    p.add_argument("--cols", type=int, default=9, help="棋盘格内角点列数")
    p.add_argument("--square", type=float, default=0.025, help="格子边长（米）")
    p.add_argument("--shots", type=int, default=15, help="内参拍几张")
    p.add_argument("--points", type=int, default=8, help="外参采几个点")
    p.add_argument("--homography", action="store_true", help="标俯视相机像素 -> 桌面 xy（给 run_playjev）")
    p.add_argument("--z", type=float, default=None, help="--homography 时的法兰高度（米），必须给")
    p.add_argument("--grid", default="0.35,-0.25;0.35,0.15;0.55,-0.25;0.55,0.15;0.45,-0.05",
                   help="--homography 采点的 xy，分号分隔。要落在泡沫板空白处")
    a = p.parse_args()

    if a.intrinsics:
        return calibrate_intrinsics(a.rows, a.cols, a.square, a.shots)
    if a.extrinsics:
        return calibrate_extrinsics(a.points)
    if a.homography:
        if a.z is None:
            print("--homography 必须给 --z（法兰高度）。受控点在法兰不在指尖，没有安全默认值。")
            return 2
        pts = [tuple(float(v) for v in s.split(",")) for s in a.grid.split(";")]
        return calibrate_homography(pts, a.z, "blue")
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
