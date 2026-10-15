"""俯视 Gemini 335 粗定位：看一眼就知道物体在基座系哪里，一步把臂送到腕部能接手的位置。

替代旧的「俯视 UGREEN 像素闭环」—— 那套每次都要开合夹爪在画面里认夹爪（怕灯光闪、怕人走动），
没有深度所以有视差、平移高度固定会刮高物体。这里：

  物体 3D（335 系）  YOLO-World -> CLIP -> SAM -> 深度（已对齐彩色）-> 支撑面 -> 抓取点/高度
  -> 臂法兰目标      外参（calibrate_top335）：球心(335) = R·法兰 + t，夹住时球心 ≈ 两指中间，
                     所以 法兰 = Rᵀ(p − t) 就是「两指中间到 p」的法兰位置
  -> 预抓取点        从抓取点沿腕部视线后退 PRE_BACK_M（腕部 J 给视线方向），腕部在这里接手
  -> 路径            先在安全高度平移到预抓取点正上方，再竖直下降

不需要每次标定、不看夹爪、不怕灯光（深度是主动红外）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import rec

CALIB = Path(__file__).resolve().parent / "calibration"
PRE_BACK_M = 0.10          # 预抓取点：抓取点沿腕部视线往回 10cm（之前成功交接时物体都在腕部视线前 ~10cm）
PRE_BACKS = (0.10, 0.08, 0.06)   # 到不了就缩短后退距离（腕部深度最近能看 ~6cm 外；夹持中心在相机前 88mm）
PREF_Z_M = 0.40            # 平移高度的**首选**（不是下限）：标定时整张网格在这个高度走过，没碰到桌上任何东西；
                           # 到不了时按 335 看到的走廊往下降（9/29 离线扫：固定朝向下 17 格桌面只差这一步）
CLEAR_M = 0.08             # 平移时两指中间比走廊里最高的东西（含目标本身）高这么多
CORRIDOR_M = 0.08          # 走廊半宽：夹爪全开 98mm + 两指各 17mm 的一半，再加余量
REACH_M = 1.0              # 环境边界「不限」用的数：比臂纯运动学包络（半径 ~0.70m）大，等于不限
BODY_M = 0.15              # 夹持中心往后（−视线方向）到机身末端：相机在夹持中心后 88mm（夹爪标定），再加转接件


def load_extrinsic():
    d = json.loads((CALIB / "top335_extrinsic.json").read_text())
    q_ref = json.loads((CALIB / "pregrasp_pose.json").read_text())["q"]
    if not np.allclose(d["q_ref"], q_ref):
        raise RuntimeError("top335_extrinsic.json 是在另一个朝向下标的，重新跑 calibrate_top335")
    return np.asarray(d["R"], float), np.asarray(d["t"], float), d


def to_flange(p_cam, R, t):
    """335 系里的点 -> 让两指中间到这个点时的法兰位置（基座系）。"""
    return R.T @ (np.asarray(p_cam, float) - t)


def to_flange_many(P, R, t):
    """N×3 版 to_flange。"""
    return (np.asarray(P, float) - t) @ R


def closing_dir(J, R):
    """夹爪开合方向在 335 系里的单位向量：开合沿腕部相机 x，J 的第 0 行 = −(相机 x 在基座系)。"""
    c = R @ (-np.asarray(J, float)[0, :])
    return c / np.linalg.norm(c)


def grip_width(pts, J, R):
    """抓取高度以上的物体点沿夹爪开合方向的宽度（米）。335 只看得见朝它那一面，所以是下限。"""
    if pts is None or len(pts) < 15:
        return None
    s = np.asarray(pts, float) @ closing_dir(J, R)
    return float(np.percentile(s, 98) - np.percentile(s, 2))


def locate(tf, cam335, cm335, R, t, n=5, tries=20, feasible=None, anchor_xy=None, width_fn=None):
    """在 335 里找物体；连看 n 帧取中位。借用 tf 的模型，换成 335 的相机模型和俯视文字，用完换回。

    **不设工作空间之类的专用限制**，335 看得见的都算。挑哪一个只看两件事：
      1. CLIP：名次 <= TOP_K 的候选，按余量从大到小
      2. 物理上抓不抓得了：feasible(obs, plane) 返回 None = 能抓，否则返回原因（逆解不出 / 比夹爪宽）。
         挑余量最大的**能抓的**那个；抓不了的不是悄悄丢掉，而是记下原因最后报出来。
    第一帧定下来以后，后面几帧只认基座系水平位置离它 < JUMP_M 的候选（同一个东西），不再重新比 CLIP ——
    否则两个都能抓的同类物体会在帧间来回换，中位数取到两者中间。

    anchor_xy：已知目标在基座系（法兰坐标）的水平位置时给（抬起后看物体跟没跟着走），只按位置认、不查可行性。
    只比水平位置：物体被抬起来，水平位置不变、高度变。

    9/29 为什么原来要工作空间盒：335 画面里远处另一张桌上也有瓶子（1628mm 外），CLIP 余量只差 0.01。
    那个瓶子本来就够不着 —— 现在由逆解排除，而不是由人画的盒子排除。
    """
    from .objects import JUMP_M, ObjectObs

    cam_wrist, last, up_wrist = tf.cam, tf.last, tf.up
    tf.cam = cm335
    tf.up = R[:, 2] / np.linalg.norm(R[:, 2])      # 基座 +z 在 335 系里的方向（球心 = R·法兰 + t）
    got, rejected, scene = [], {}, None
    anchor = None if anchor_xy is None else np.asarray(anchor_xy, float)[:2]
    try:
        for _ in range(tries):
            f = cam335.read()
            if not (f.ok and f.depth is not None):
                time.sleep(0.03)
                continue
            cands = tf._candidates(f.color)
            if not cands:
                continue
            rank, margin = tf._rank(f.color, cands, view="top")
            order = sorted((i for i in range(len(cands)) if rank[i] <= tf.TOP_K), key=lambda i: -margin[i])
            best = None
            for i in order:
                c, mg = cands[i], float(margin[i])
                mask = tf._segment(f.color, c.xyxy)
                if mask is None or not mask.any():
                    continue
                tf._last_z_g = None
                tf._grasp_pts = None
                geo = tf._geometry(f.depth, mask)
                if geo is None:
                    continue
                xy = to_flange(geo[0], R, t)[:2]
                item = (mg, c, geo, mask, tf._last_z_g, tf._support_plane(f.depth, mask), tf._grasp_pts, f.depth)
                if anchor is not None:
                    d = float(np.linalg.norm(xy - anchor))
                    if d <= JUMP_M and (best is None or d < best[0]):
                        best = (d, item)
                    continue
                if feasible is not None:
                    why = feasible(_obs(item), item[5])
                    if why:
                        key = tuple(np.round(xy / 0.05).astype(int))     # 同一个东西不同帧只记一次
                        rejected.setdefault(key, (c.phrase, mg, xy, why))
                        continue
                best = (0.0, item)
                break
            if best is not None:
                got.append(best[1])
                if anchor is None:
                    anchor = to_flange(best[1][2][0], R, t)[:2]
                scene = f.depth
            if len(got) >= n:
                break
            time.sleep(0.03)
    finally:
        tf.cam, tf.last, tf.up = cam_wrist, last, up_wrist
    rej = list(rejected.values())
    if len(got) < 3:
        print(f"   335 里没有稳定的、抓得了的目标（{len(got)} 帧）")
        for ph, mg, xy, why in rej:
            print(f"     看到了但抓不了：「{ph}」CLIP 余量 {mg:+.3f} 法兰水平位置 {np.round(xy * 1000).astype(int).tolist()}mm —— {why}")
        return None
    med = lambda k: np.median([g[2][k] for g in got], axis=0)
    mg, c, geo, mask, _, plane, _, _ = got[-1]
    ys, xs = np.nonzero(mask)
    hs = [g[2][3] for g in got if g[2][3] is not None]
    zs = [g[4] for g in got if g[4] is not None]
    obs = ObjectObs(phrase=c.phrase, xyxy=np.array([xs.min(), ys.min(), xs.max(), ys.max()], float),
                    mask_px=int(mask.sum()), centre=med(2), surface=med(0), extent=med(1), source="detect",
                    clip=mg, extra={"height": float(np.median(hs)) if hs else None,
                                     "z_g": float(np.median(zs)) if zs else None, "rejected": rej,
                                     "grasp_pts": got[-1][6], "scene_depth": scene,
                                     "width": _median_or_none([g[6] for g in got], width_fn)},
                    grasp=med(4), grasp_extent=med(5))
    return obs, plane


def _median_or_none(pts_list, fn):
    if fn is None:
        return None
    v = [fn(p) for p in pts_list]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def _obs(item):
    """locate 里单帧候选 -> ObjectObs（给可行性检查用）。"""
    from .objects import ObjectObs
    mg, c, geo, mask, z_g, plane, pts, depth = item
    return ObjectObs(phrase=c.phrase, xyxy=np.asarray(c.xyxy, float), mask_px=int(mask.sum()), centre=geo[2],
                     surface=geo[0], extent=geo[1], source="detect", clip=mg,
                     extra={"height": geo[3], "z_g": z_g, "grasp_pts": pts, "scene_depth": depth},
                     grasp=geo[4], grasp_extent=geo[5])


def support_flange_z(obs, plane, R, t):
    """物体下面支撑面的高度，换成「两指中间在那里」时的法兰 z。量不到支撑面就用物体最低的可见点（只会更高、更保守）。"""
    up = R[:, 2] / np.linalg.norm(R[:, 2])
    z_g = obs.extra.get("z_g")
    if plane is not None and z_g is not None:
        return float(to_flange(obs.grasp - z_g * up, R, t)[2])
    pts = obs.extra.get("grasp_pts")
    if pts is not None and len(pts):
        return float(np.percentile(to_flange_many(pts, R, t)[:, 2], 2))
    return None


def scene_obstacle_z(depth, cm335, R, t, f_from, f_to, radius=CORRIDOR_M, body=BODY_M, a=None):
    """335 深度图里，夹爪从 f_from 平移到 f_to（法兰坐标，水平）扫过的走廊中最高的东西（法兰坐标 z）。

    走廊 = 两指中间走过的水平线段，加宽 radius（夹爪半宽 + 余量），再往 −a 的水平方向拖 body 那么长
    （手指后面的相机和机身在夹持中心后上方）。臂自己也在 335 画面里：法兰坐标下夹爪最低的部分（指尖）
    只比起点低 ~1cm，臂的其余部分都在更上面，所以只有比起点低 3cm 以上的点才算障碍。
    返回 (最高点 z, 走廊里的点数)；没有点返回 (None, 0)。
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:3, 0:w:3]
    P = cm335.points_keep(depth, xs.ravel().astype(float), ys.ravel().astype(float))
    P = P[np.isfinite(P[:, 0])]
    F = to_flange_many(P, R, t)
    F = F[F[:, 2] < f_from[2] - 0.03]
    q = F[:, :2]
    p0, p1 = np.asarray(f_from, float)[:2], np.asarray(f_to, float)[:2]
    back = np.zeros(2) if a is None else -body * np.asarray(a, float)[:2]
    near = np.zeros(len(q), bool)
    for s in (0.0, 0.5, 1.0):                      # 夹持中心 -> 机身末端，几条平行线段的并
        o0, o1 = p0 + s * back, p1 + s * back
        d = o1 - o0
        u = np.clip(((q - o0) @ d) / max(d @ d, 1e-9), 0.0, 1.0)
        near |= np.linalg.norm(q - (o0 + np.outer(u, d)), axis=1) < radius
    if near.sum() < 20:
        return None, int(near.sum())
    return float(np.percentile(F[near, 2], 99.5)), int(near.sum())   # 99.5 百分位：飞点不算


def plan(obs, plane, R, t, J, kin, q_seed, scene=None, cm335=None, f_start=None):
    """算预抓取点、平移高度和三个关节目标。返回 dict；到不了就 raise（原因里说是哪一段）。

    没有工作空间盒：能不能到只看逆解。几个几何量到不了时按顺序退：
      * 预抓取点沿腕部视线后退 PRE_BACKS 里的距离，远的优先（腕部接手时离物体远一点看得全）
      * 平移高度：优先 PREF_Z_M（9/29 标定时整张网格在这个高度走过、没碰到任何东西），
        到不了就往下降，最低降到「335 看到的走廊里最高的东西 + CLEAR_M」和预抓取点高度中较高的那个
        （要 scene；没有 scene 就不降）
    """
    R0 = kin.fk(q_seed)[:3, :3]
    f_grasp = to_flange(obs.grasp, R, t)
    a = -J[2, :]                                   # 腕部光轴在基座系的方向（固定朝向下是常量）
    a = a / np.linalg.norm(a)
    h = obs.extra.get("height") or 0.0

    def ik(p, seed):
        T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = p
        return kin.ik(T, seed, max_iters=300)

    q_g, info = ik(f_grasp, q_seed)
    if q_g is None:
        raise RuntimeError(f"抓取点 法兰 {np.round(f_grasp, 3).tolist()} 逆解不出（{info['reason']}）—— 固定朝向下够不着")
    top = None
    if plane is not None:
        c0, n = plane
        top = to_flange(obs.grasp + (h - (obs.extra.get("z_g") or h / 2)) * (n if n @ (-c0) > 0 else -n), R, t)[2]
    why = []
    for back in PRE_BACKS:
        f_pre = f_grasp - back * a
        q_p, info = ik(f_pre, q_g)
        if q_p is None:
            why.append(f"后退 {back * 100:.0f}cm 的预抓取点逆解不出")
            continue
        need = f_pre[2] if top is None else max(f_pre[2], top + CLEAR_M)
        z_pref = max(PREF_Z_M, need)
        zs = [z_pref]
        if scene is not None and f_start is not None:
            # 走廊里 335 看得见东西（至少看得见桌面）才降；看不见 = 不知道有什么，不降
            z_obs, _ = scene_obstacle_z(scene, cm335, R, t, f_start, f_pre, a=a)
            if z_obs is not None:
                low = max(f_pre[2], z_obs + CLEAR_M)
                zs += [float(z) for z in np.arange(z_pref - 0.02, low, -0.02)] + ([low] if low < z_pref else [])
        for z in zs:
            f_above = np.r_[f_pre[:2], z]
            q_a, info = ik(f_above, q_p)
            if q_a is not None:
                note = "" if z == z_pref else f"（{z_pref * 1000:.0f}mm 够不着，按 335 看到的走廊降下来）"
                return dict(f_grasp=f_grasp, f_pre=f_pre, f_above=f_above, height=h, back=back, note=note,
                            q_grasp=q_g, q_pre=q_p, q_above=q_a)
        why.append(f"后退 {back * 100:.0f}cm 时平移高度 {zs[0] * 1000:.0f}~{zs[-1] * 1000:.0f}mm 都逆解不出")
    raise RuntimeError(f"抓取点够得着，但接近路径到不了：{'；'.join(why)}")


def feasibility(o, plane, R, t, J, kin, q_ref, cm335, f_start, full_open=0.098, need_height=False):
    """能不能抓 —— 只看物理条件，不画框。返回 (None, plan) = 能抓；(原因, None) = 抓不了。

      夹得下：抓取高度以上沿夹爪开合方向的宽度 ≤ 满开 − 预留（grip_width，只算看得见的一面，是下限）
      够得着：抓取点 / 预抓取点 / 平移点逆解都得出（plan）
      need_height（点选模式）：物体要高出它的支撑面 FINGER_FLOOR_M 以上 —— 夹爪能合住的最低高度。
        按名字找时 CLIP 先保证了选中的是那个物体；点选时人可以点到坑、平面、阴影上。9/29 白板上的凹槽被判「能抓」：
        测不出高度时 _geometry 退回「可见面 + 最窄维一半」，plan 照样通过。
    """
    from .grip_force import FINGER_FLOOR_M, PRE_GAP_M

    if need_height:
        h = o.extra.get("height")
        if h is None:
            return "量不到它下面的支撑面（不是放在平面上的东西，或者底部看不见），测不出高度", None
        if h <= FINGER_FLOOR_M:
            return (f"只高出支撑面 {h * 1000:.0f}mm，不到夹爪能合住的 {FINGER_FLOOR_M * 1000:.0f}mm"
                    f"（坑、平面、贴着桌面的片）"), None
    w = grip_width(o.extra.get("grasp_pts"), J, R)
    if w is not None and w > full_open - PRE_GAP_M:
        return (f"抓取处沿夹爪方向宽 {w * 1000:.0f}mm（只算看得见的一面），"
                f"夹爪全开 {full_open * 1000:.0f}mm 减预留 {PRE_GAP_M * 1000:.0f}mm 放不进"), None
    try:
        return None, plan(o, plane, R, t, J, kin, q_ref, scene=o.extra.get("scene_depth"), cm335=cm335, f_start=f_start)
    except RuntimeError as e:
        return str(e), None


def measure_mask(tf, depth, mask, cm335, R, t):
    """335 一帧里给定掩码（点选出来的）的几何 -> (ObjectObs, 支撑面) 或 None。
    借 tf 的几何算法（支撑面、高度、抓取点和 locate 完全一样），换成 335 的相机和竖直方向，用完换回。"""
    from .objects import ObjectObs

    cam_w, last, up_w = tf.cam, tf.last, tf.up
    tf.cam = cm335
    tf.up = R[:, 2] / np.linalg.norm(R[:, 2])
    try:
        tf._last_z_g = None
        tf._grasp_pts = None
        geo = tf._geometry(depth, mask)
        if geo is None:
            return None
        z_g, pts, sup = tf._last_z_g, tf._grasp_pts, getattr(tf, "_support_info", None)
        plane = tf._support_plane(depth, mask)
    finally:
        tf.cam, tf.last, tf.up = cam_w, last, up_w
    ys, xs = np.nonzero(mask)
    obs = ObjectObs(phrase=tf.phrase, xyxy=np.array([xs.min(), ys.min(), xs.max(), ys.max()], float),
                    mask_px=int(mask.sum()), centre=geo[2], surface=geo[0], extent=geo[1], source="point", clip=0.0,
                    extra={"height": geo[3], "z_g": z_g if geo[3] is not None else None, "grasp_pts": pts,
                           "scene_depth": depth, "support": sup, "rejected": []},
                    grasp=geo[4], grasp_extent=geo[5])
    return obs, plane


def locate_click(tf, cam335, cm335, R, t, sel, J, n=5, tries=20):
    """点选模式的 locate：拿人点的提示（点 / 框）在新的几帧上重新分割，取和当时选中的掩码最重合的那个粒度，
    测几何，n 帧取中位。重合度 < 0.5 的帧不要 —— 物体被挪了或被挡了。"""
    from .objects import _mask_iou

    got, best_iou = [], 0.0
    for _ in range(tries):
        f = cam335.read()
        if not (f.ok and f.depth is not None):
            time.sleep(0.03)
            continue
        ms = tf.segment_prompt(f.color, points=sel["points"], labels=sel["labels"], box=sel.get("box"), multimask=True)
        if not ms:
            continue
        ious = [_mask_iou(m, sel["mask"]) for m in ms]
        k = int(np.argmax(ious))
        best_iou = max(best_iou, ious[k])
        if ious[k] < 0.5:
            continue
        r = measure_mask(tf, f.depth, ms[k], cm335, R, t)
        if r is not None:
            got.append(r)
        if len(got) >= n:
            break
        time.sleep(0.03)
    if len(got) < 3:
        print(f"   335 里对不上点选的目标（{len(got)} 帧；和点选时的掩码最多重合 {best_iou:.2f}）—— 物体被挪动或被挡住了，重新点")
        return None
    obs_all = [g[0] for g in got]
    obs, plane = got[-1]
    for k in ("centre", "surface", "extent", "grasp", "grasp_extent"):
        setattr(obs, k, np.median([getattr(o, k) for o in obs_all], axis=0))
    hs = [o.extra["height"] for o in obs_all if o.extra["height"] is not None]
    zs = [o.extra["z_g"] for o in obs_all if o.extra["z_g"] is not None]
    obs.extra["height"] = float(np.median(hs)) if hs else None
    obs.extra["z_g"] = float(np.median(zs)) if zs else None
    obs.extra["width"] = _median_or_none([o.extra["grasp_pts"] for o in obs_all], lambda p: grip_width(p, J, R))
    return obs, plane


def open_workspace(env, floor_z):
    """环境层的工作空间盒（AR5Config，遥操用的）换成这次抓取自己的边界：水平和上方不限 —— 够不够得着由逆解说了算；
    下方 = 335 量的支撑面，两指中间不低于它上方 FINGER_FLOOR_M（指尖不碰桌）。只改这个进程里的 env.config。

    原来的 z 下界 0.08 / 关节地板 0.10 是按因时手推的（台面 −0.08 + 手长 0.15）；换成 TG9801 后，
    桌上物体抓取时法兰在 ~0.19（9/29 日志），法兰到 0.08 时手指早已插进桌面 9cm —— 那道闸既不限制也不保护。"""
    from .grip_force import FINGER_FLOOR_M

    arm = env.config.arm
    old = (tuple(arm.workspace_min), tuple(arm.workspace_max), arm.joint_floor_z)
    z = float(floor_z) + FINGER_FLOOR_M
    arm.workspace_min = (-REACH_M, -REACH_M, z)
    arm.workspace_max = (REACH_M, REACH_M, REACH_M)
    arm.joint_floor_z = z
    print(f"   环境边界：水平/上方不限（逆解决定）；法兰下限 = 335 量的支撑面 {floor_z * 1000:.0f}mm + {FINGER_FLOOR_M * 1000:.0f}mm "
          f"= {z * 1000:.0f}mm（原工作空间盒 {np.round(np.array(old[0]) * 1000).astype(int).tolist()}~"
          f"{np.round(np.array(old[1]) * 1000).astype(int).tolist()}mm、关节地板 {old[2] * 1000:.0f}mm）")
    return old


def approach(env, tf, cam335, cm335, J, slow_goto, step_and_settle, path_min_z, full_open=0.098, click=None):
    """0 位 -> 335 定位 -> 平移到预抓取点上方 -> 竖直降到预抓取点。
    click = click_grasp.select 返回的点选结果：不按名字找，按人点的提示重新分割（locate_click），并要求物体高出支撑面。
    成功返回 {obs, plane, plan, R, t}（交接后要拿 335 的测量和腕部交叉验证），失败返回 None。"""
    from atom_robot.ar5.kinematics import AR5Kinematics

    from . import actions as A

    rec.tag("top335_locate")
    R, t, meta = load_extrinsic()
    kin = AR5Kinematics.from_robot(env.arm)
    q_ref = np.asarray(meta["q_ref"], float)
    q_now = np.asarray(env.arm.get_joint_positions(), float)
    f_start = kin.fk(q_now)[:3, 3]
    width_fn = lambda pts: grip_width(pts, J, R)
    need_height = click is not None

    def feasible(o, plane):
        return feasibility(o, plane, R, t, J, kin, q_ref, cm335, f_start, full_open, need_height)[0]

    if click is None:
        r = locate(tf, cam335, cm335, R, t, feasible=feasible, width_fn=width_fn)
    else:
        r = locate_click(tf, cam335, cm335, R, t, click, J)
    if r is None:
        print("335 里没有能抓的目标")
        return None
    obs, plane = r
    h, w = obs.extra.get("height"), obs.extra.get("width")
    print(f"335 看到目标：抓取点(335 系) {np.round(obs.grasp * 1000).astype(int).tolist()}mm  "
          f"高度 {'—' if h is None else f'{h * 1000:.0f}mm'}  沿夹爪方向宽 {'—' if w is None else f'{w * 1000:.0f}mm'}"
          f"（外参 {meta['date']}，留一误差 ~2mm）")
    for ph, mg, xy, why in obs.extra["rejected"]:
        print(f"   另外看到但抓不了：「{ph}」CLIP 余量 {mg:+.3f} 法兰水平位置 {np.round(xy * 1000).astype(int).tolist()}mm —— {why}")
    why, p = feasibility(obs, plane, R, t, J, kin, q_ref, cm335, f_start, full_open, need_height)
    if why:
        print(f"   这个目标抓不了：{why}")
        return None
    print(f"   两指中间到抓取点时法兰 {np.round(p['f_grasp'] * 1000).astype(int).tolist()}mm；"
          f"预抓取（沿视线后退 {p['back'] * 100:.0f}cm）{np.round(p['f_pre'] * 1000).astype(int).tolist()}mm；"
          f"平移高度 {p['f_above'][2] * 1000:.0f}mm{p['note']}")
    floor = support_flange_z(obs, plane, R, t)
    if floor is not None:
        open_workspace(env, floor)
    else:
        print("   335 量不到支撑面，环境边界不改（仍用工作空间盒）")
    zmin = path_min_z(kin, q_now, p["q_above"])
    z_lo = min(f_start[2], p["f_above"][2]) - 0.02
    if zmin < z_lo:
        raise RuntimeError(f"到预抓取上方的关节插值途中法兰降到 {zmin:.3f}m（低于 {z_lo:.3f}），不走")
    rec.tag("top335_move")
    reached, el = slow_goto(env, p["q_above"])
    print(f"   到预抓取点上方: {reached} {el:.1f}s")
    dz = p["f_pre"][2] - p["f_above"][2]
    n = int(np.ceil(abs(dz) / A.COARSE_M - 1e-9))
    for _ in range(n):                              # 竖直直线下降（笛卡尔小步，不走关节插值）
        step_and_settle(env, [0.0, 0.0, dz / n])
    print(f"   竖直下降 {abs(dz) * 1000:.0f}mm 到预抓取点")
    time.sleep(0.3)
    return dict(obs=obs, plane=plane, plan=p, R=R, t=t)
