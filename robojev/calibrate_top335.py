#!/usr/bin/env python3
"""俯视 Gemini 335 ↔ 臂 外参标定：夹着乒乓球走一组点，335 看球心，拟合刚体变换。只做一次。

    python -m robojev.calibrate_top335 --dry-run    # 只算逆解和路径，不动臂
    python -m robojev.run_composite --instruction "pick up the ping pong ball" --stage grasp   # 先把球抓起来
    python -m robojev.calibrate_top335              # 夹着球走点、拟合，结果写 calibration/top335_extrinsic.json
    python -m robojev.place_and_home --down 0.06    # 回到起始点后放回球

为什么夹着球：335 从桌对面约 1m 看，黑色手指深度不可靠；橙色的球 CV 好认、深度好测，
而且夹住后球心就在两指中间（≈ 夹持中心），位置相对夹爪固定。

拟合什么：朝向固定（和抓取时相同的 R0）时，球心 = 法兰 + 常量偏移，所以
    球心(335 系) = R · 法兰(基座系) + t
R 是纯旋转（Kabsch），t 吸收了相机位置和球相对法兰的偏移。
用法：335 看到物体在 p，臂法兰去 R^T(p − t)，两指中间就在物体上 —— 不需要知道偏移本身。

安全：所有点法兰 z ≥ 0.40m（球底离桌面 ≥ 20cm，高于桌上最高的水瓶 17.5cm）；
动臂前逐段检查关节插值路径不下沉；夹爪始终保持合拢（连臂时不张爪）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "算法" / "DexGraspVLA-" / "SO101"))

CALIB = Path(__file__).resolve().parent / "calibration"
OUT = CALIB / "top335_extrinsic.json"
GRID_X = (0.30, 0.38, 0.46)
GRID_Y = (-0.35, -0.25, -0.15)
GRID_Z = (0.40, 0.46)
Z_MIN = 0.40                 # 走点时法兰最低高度
DIP_TOL = 0.02               # 关节插值途中允许比两端低多少
BALL_R = 0.020               # 乒乓球半径（40mm 标准球）；换别的球用 --radius
MAX_LOO_M = 0.005            # 留一误差中位超过这个就不写外参（9/30 俯视时写进过留一 267mm 的外参）


def grid():
    """蛇形：同层相邻点只差一格；上层倒序，换层是原地竖直抬起。"""
    layer = []
    for j, y in enumerate(GRID_Y):
        xs = GRID_X if j % 2 == 0 else GRID_X[::-1]
        layer += [(x, y) for x in xs]
    pts = [np.array([x, y, GRID_Z[0]]) for x, y in layer]
    pts += [np.array([x, y, GRID_Z[1]]) for x, y in layer[::-1]]
    return pts


def plan(kin, R0, q_start):
    """逐点逆解（上一点当种子）+ 检查每段关节插值途中法兰不低于两端较低者 − DIP_TOL、不低于 Z_MIN − DIP_TOL。"""
    out, q_prev = [], np.asarray(q_start, float)
    for p in grid():
        T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = p
        q, info = kin.ik(T, q_prev, max_iters=300)
        if q is None:
            out.append((p, None, f"逆解失败 {info['reason']}"))
            continue
        z0, z1 = kin.fk(q_prev)[2, 3], p[2]
        zmin = min(kin.fk(q_prev + (q - q_prev) * s)[2, 3] for s in np.linspace(0, 1, 25))
        floor = max(min(z0, z1) - DIP_TOL, Z_MIN - DIP_TOL) if out else min(z0, z1) - DIP_TOL
        ok = zmin >= floor
        out.append((p, q if ok else None, f"途中最低 {zmin:.3f}" + ("" if ok else f" < {floor:.3f}，不走")))
        if ok:
            q_prev = q
    return out


def sphere_fit(P, r=None, c0=None, iters=30):
    """点云拟合球（Gauss-Newton）。r=None 时半径一起拟合。返回 (球心, 半径)。"""
    c = np.asarray(c0 if c0 is not None else P.mean(0), float).copy()
    rr = float(r) if r is not None else float(np.median(np.linalg.norm(P - c, axis=1)))
    for _ in range(iters):
        v = P - c
        dist = np.linalg.norm(v, axis=1) + 1e-9
        res = dist - rr
        Jc = -v / dist[:, None]
        Jm = np.c_[Jc, -np.ones(len(P))] if r is None else Jc
        step = np.linalg.lstsq(Jm, -res, rcond=None)[0]
        c += step[:3]
        if r is None:
            rr += step[3]
        if np.linalg.norm(step[:3]) < 1e-6:
            break
    return c, rr


def color_core(img, mask, min_px=30):
    """掩码里只留「和球本身同色」的像素：饱和度高（黑手指、白板都不饱和），色相接近掩码里饱和像素的中位色相。
    9/30 335 挪到正上方后，夹着的球一半被夹爪挡住，分割框里混进大片黑手指和白板，17 点里 11 点球面拟合发散
    （半径 100~680mm）。只按「饱和 + 同色相」筛，不写死橙色 —— 换别的彩色球也成立。"""
    import cv2

    if mask is None or mask.sum() < min_px:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = mask & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    if sat.sum() < min_px:
        return None
    h = hsv[..., 0].astype(int)
    h0 = int(np.median(h[sat]))
    dh = np.minimum(np.abs(h - h0), 180 - np.abs(h - h0))              # 色相是环形的
    core = sat & (dh < 12)
    return core if core.sum() >= min_px else None


def ball_points(tf, cam, n=5):
    """335 里球的可见表面点（相机系，米），连取 n 帧合并。"""
    Ps = []
    for _ in range(n * 3):
        f = cam.read()
        if not (f.ok and f.depth is not None):
            time.sleep(0.03)
            continue
        cands = tf._candidates(f.color, max_area_frac=0.25)
        if not cands:
            continue
        i, rank, m = tf._pick(f.color, cands, view="top")
        if i is None:
            continue
        mask = color_core(f.color, tf._segment(f.color, cands[i].xyxy))
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        P = tf.cam.points(f.depth, xs, ys)
        if len(P) < 30:
            continue
        P = P[np.abs(P[:, 2] - np.median(P[:, 2])) < 0.03]      # 去掉漏进掩码的背景/手指
        if len(P) >= 30:
            Ps.append(P)
        if len(Ps) >= n:
            break
        time.sleep(0.03)
    return np.concatenate(Ps) if len(Ps) >= 3 else None


def kabsch(F, C):
    """C ≈ R F + t。"""
    fm, cm = F.mean(0), C.mean(0)
    U, _, Vt = np.linalg.svd((F - fm).T @ (C - cm))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cm - R @ fm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refit", action="store_true", help="不动臂，用上次存的 top335_points.npz 重新拟合")
    ap.add_argument("--radius", type=float, default=BALL_R, help="标定球半径（米），乒乓球 0.020")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    import warnings
    warnings.filterwarnings("ignore")
    from atom_robot.ar5.kinematics import AR5Kinematics

    q_ref = np.asarray(json.loads((CALIB / "pregrasp_pose.json").read_text())["q"], float)

    if a.refit:
        d = np.load(CALIB / "top335_points.npz")
        return fit_and_save([(d["flange"][k], d[f"P{k}"]) for k in range(len(d["flange"]))], q_ref, a.radius)

    if a.dry_run:
        from atom_robot.ar5.config import AR5Config
        from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
        arm = RokaeAR5(AR5Config().arm); arm.connect(power_on=False)
        kin = AR5Kinematics.from_robot(arm)
        R0 = kin.fk(q_ref)[:3, :3]
        q0 = np.asarray(arm.get_joint_positions(), float)
        for p, q, msg in plan(kin, R0, q0):
            print(f"法兰 {np.round(p, 3).tolist()}  {'OK ' if q is not None else '跳过 '}{msg}")
        print("（dry-run 以当前姿态为起点；真跑时以抓完球后的姿态为起点重新检查）")
        return 0

    from .camera import TOP_SERIAL, OrbbecRGBD
    from .move_pregrasp import slow_goto
    from .objects import CameraModel, TargetFinder
    from .run_composite import KeepAlive
    from .run_playjev import settle
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    tf = TargetFinder("pick up the ping pong ball")
    tf.cam = CameraModel(CALIB / "orbbec335_params.json")
    cam = OrbbecRGBD(serial=TOP_SERIAL); cam.connect()
    cfg = AR5Config(); cfg.mock = False; cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True
    env = AR5Env(cfg); env.connect()
    grip = float(env.hand.get_opening())
    settle(env, gripper=grip)                         # 连上先稳住，夹爪保持现状（别把球放掉）
    ka = KeepAlive(env).__enter__()
    rec = []
    try:
        if grip < 0.2 or grip > 0.97:
            raise SystemExit(f"夹爪读数 {grip:.2f}，不像夹着球（应 ~0.6）。先抓球再标定")
        kin = AR5Kinematics.from_robot(env.arm)
        R0 = kin.fk(q_ref)[:3, :3]
        q_start = np.asarray(env.arm.get_joint_positions(), float)
        steps = plan(kin, R0, q_start)
        for k, (p, q, msg) in enumerate(steps):
            if q is None:
                print(f"[{k:2d}] 法兰 {np.round(p, 3).tolist()} 跳过：{msg}")
                continue
            slow_goto(env, q)
            time.sleep(0.6)
            f = kin.fk(np.asarray(env.arm.get_joint_positions(), float))[:3, 3]
            P = ball_points(tf, cam)
            if P is None:
                print(f"[{k:2d}] 法兰 {np.round(f, 3).tolist()}  335 里没认出球")
                continue
            c, r = sphere_fit(P)
            rec.append((f, P))
            print(f"[{k:2d}] 法兰 {np.round(f * 1000).astype(int).tolist()}mm  球心(335，半径自由) {np.round(c * 1000).astype(int).tolist()}mm"
                  f"  半径 {r * 1000:.1f}mm  {len(P)} 点  夹爪 {env.hand.get_opening():.2f}")
        print("回到起始姿态（球的上方）")
        slow_goto(env, q_start)
    finally:
        ka.__exit__(None, None, None)
        env.close(); cam.disconnect()

    if len(rec) < 6:
        raise SystemExit(f"有效点只有 {len(rec)} 个（至少 6 个），没写结果")
    return fit_and_save(rec, q_ref, a.radius)


def fit_and_save(rec, q_ref, r_ball=BALL_R):
    """rec = [(法兰, 球表面点云)]。标记点 = 球心，用已知半径拟合。

    9/29：各点自由半径 20.5~22.6mm（乒乓球 20mm），统一半径后留一误差中位 1.8mm、最大 3.0mm。
    9/30：335 挪到正上方，一半点分割坏了（自由半径 100~680mm），原来「取各点半径中位数当统一半径」
    被坏点带到 188mm，留一 267mm 还照样写进了外参。现在：
      * 半径用已知值（乒乓球 20mm，--radius 可改），自由拟合的半径只用来筛坏点：偏离 >30% 的点不用
      * Kabsch 之后逐个去掉误差最大的点，直到它 < max(5mm, 3×中位)，至少留 6 点
      * 留一误差中位 > MAX_LOO_M 就不写外参（写到 *_rejected.json 备查）
    """
    np.savez(CALIB / "top335_points.npz", flange=np.array([r[0] for r in rec]),
             **{f"P{k}": r[1] for k, r in enumerate(rec)})
    free = [sphere_fit(P) for _, P in rec]
    rf = np.array([r for _, r in free])
    good = np.abs(rf / r_ball - 1.0) < 0.3
    print(f"各点自由拟合半径 {np.round(rf * 1000, 1).tolist()} mm；已知半径 {r_ball * 1000:.1f}mm，"
          f"偏离 <30% 的 {good.sum()}/{len(rf)} 点可用")
    if good.sum() < 6:
        print(f"可用点只有 {good.sum()} 个（至少 6 个），不写外参")
        return 1
    F = np.array([f for f, _ in rec])
    C = np.array([sphere_fit(P, r=r_ball, c0=c)[0] for (_, P), (c, _) in zip(rec, free)])
    keep = good.copy()
    while True:
        R, t = kabsch(F[keep], C[keep])
        err = np.linalg.norm(C - (F @ R.T + t), axis=1)
        worst = int(np.argmax(np.where(keep, err, -1)))
        if keep.sum() <= 6 or err[worst] < max(0.005, 3 * np.median(err[keep])):
            break
        keep[worst] = False
    print(f"拟合残差：中位 {np.median(err[keep]) * 1000:.1f}mm  最大 {err[keep].max() * 1000:.1f}mm（用 {keep.sum()}/{len(err)} 点）")
    idx = np.nonzero(keep)[0]
    loo = []
    for i in idx:                                   # 留一：去掉这一点拟合，预测它差多少 ≈ 实际使用时的误差
        m = idx[idx != i]
        Ri, ti = kabsch(F[m], C[m])
        loo.append(np.linalg.norm(C[i] - (Ri @ F[i] + ti)))
    print(f"留一预测误差：中位 {np.median(loo) * 1000:.1f}mm  最大 {max(loo) * 1000:.1f}mm")
    print(f"335 在基座系的位置 {np.round(-R.T @ t, 3).tolist()} m（含球相对法兰的偏移，只作参考）")
    out = dict(R=R.tolist(), t=t.tolist(), q_ref=q_ref.tolist(), rms_mm=float(np.sqrt(np.mean(err[keep] ** 2)) * 1000),
               loo_median_mm=float(np.median(loo) * 1000), loo_max_mm=float(max(loo) * 1000), ball_radius_m=r_ball,
               points=[dict(flange=f.tolist(), ball=c.tolist(), used=bool(u)) for f, c, u in zip(F, C, keep)],
               note="球心(335 系) = R·法兰(基座系) + t；朝向固定为 pregrasp_pose.json 的 q；夹的是乒乓球",
               date=time.strftime("%Y-%m-%d %H:%M"))
    if np.median(loo) > MAX_LOO_M:
        bad = OUT.with_name(OUT.stem + "_rejected.json")
        bad.write_text(json.dumps(out, indent=1))
        print(f"留一误差中位 {np.median(loo) * 1000:.1f}mm > {MAX_LOO_M * 1000:.0f}mm，不采用；写到 {bad.name} 备查")
        return 1
    OUT.write_text(json.dumps(out, indent=1))
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
