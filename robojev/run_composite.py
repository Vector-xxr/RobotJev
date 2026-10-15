#!/usr/bin/env python3
"""复合策略抓取：CV 模型 + 深度相机给数，PlayJev 选方向。**不含任何物体专属常数**，换物体只换指令（或 --click 点选）。

完整流程和每个数的来源见 JEV_GRASP_SUMMARY.md。**10-12 起末端换成灵巧手、两台相机都换了位置，
9/30 的标定全部作废，真机暂时跑不了**，见 README.md。

    python -m robojev.run_composite --instruction "pick up the blue cylinder" --stage grasp
    python -m robojev.run_composite --click --stage grasp        # 在 335 的一帧里用鼠标点要抓的物体

--stage 停在哪一步（默认 --locate top335 时；三步都会连臂、动臂）：
  locate    回 0 位 -> 335 定位（YOLO-World 只查指令派生的说法 -> CLIP 筛 -> MobileSAM + 深度）
            -> 能不能抓（逆解、夹得下、走廊）-> 平移到预抓取点上方、竖直降到预抓取点，停。
  pregrasp  再加腕部接手：现场测 J、认物体、和 335 交叉验证高度；CLIP 定力度和抓取位姿。不合爪。
  grasp     再加 PlayJev 伺服（偏差画成贪吃蛇棋盘，选上下左右）-> 合爪 -> 抬 60mm -> 判成败。
            腕部和 335 量的高度对不上（如透明物体）时，改走 335 直线接近。
--locate 的 demo（示范单应性）、top（旧 UGREEN 俯视，相机已拆）、wrist 是早期的定位方式。

所有数的来源：
  目标中心/尺寸     MobileSAM 掩码 + 深度 -> 点云（腕部 305 先去畸变）；335 连看 5 帧取中位
  335 -> 臂         calibrate_top335 的外参（calibration/top335_extrinsic.json），只在固定抓取朝向下成立
  夹持中心 G        calibrate_gripper --centre 标的（calibration/gripper.json）；没标过才用示范测的旧值 [15,32,88]mm
  相机运动映射 J    现场点动 ±1cm，静态场景光流 + 深度，投影到最近的正交矩阵；测不出时用 calibration/wrist_J.json
  离支撑面下限      指尖在 G 下方多深 + 2mm（9/30 长手指是 26mm）；--finger-floor 只这一次覆盖
  力度              CLIP 判易碎 / 软 / 硬 -> 查表；--force 只这一次覆盖
  合爪应有读数      1 - 宽度/满开（满开 98mm，TG9801 手册；calibrate_gripper 量过就用量的）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))

from . import actions as A                                            # noqa: E402
from . import rec                                                      # noqa: E402
from .composite import choose_pregrasp, homography, in_coverage, pixel_to_xy  # noqa: E402
from .probe_playjev import ask                                         # noqa: E402
from .run_playjev import adaptive_board, settle                        # noqa: E402
from .snake_render import INSTRUCTIONS, OPTIONS, render                # noqa: E402

def _grasp_centre():
    """夹持中心 G（腕部相机系）：calibrate_gripper --centre 标过就用标定值，否则示范测的旧值 [15,32,88]mm。"""
    import json as _j
    f = Path(__file__).resolve().parent / "calibration" / "gripper.json"
    g = _j.loads(f.read_text()) if f.exists() else {}
    return np.asarray(g.get("grasp_centre_cam") or [0.0148, 0.0323, 0.0882], float)


GRASP_CENTER_CAM = _grasp_centre()
GRIPPER_FULL_OPEN_M = 0.098            # TG9801 手册：行程 0~98mm
JOG_M = 0.010
ALIGN_TOL_M = 0.006
AXIS_DEADBAND_M = A.FINE_M / 2
FINGER_NEAR_M = 0.13                   # 比这近的点是手指（实测 0.075~0.087m），光流里排除
RUN = {}                               # 这次运行采用的力度 / 位姿 / 结果，收尾时打印（print_summary）
FORCE_OVERRIDE = None                  # --force：只这一次，覆盖 CLIP 判出的力度档位


# ---------------------------------------------------------------- RT 看门狗

class KeepAlive:
    """每 50ms 重置 RT 发送线程的「多久没收到目标」计时，只碰计时、不碰目标。

    发送线程的规矩是「臂在动时 0.25s 收不到新目标就刹车锁存」。这条闭环每一步都要跑
    YOLO-World + CLIP + SAM，settle 判停稳时臂常常还在极慢地蠕动 —— 第一次搜索就这样锁了。
    和 `RokaeAR5.hold_until_rest` 是同一个原语（`_goal_ticks = 0`），驱动注释写明它
    "touches only the staleness clock, never the goal"：目标不变，臂只会停在最后一个目标上。
    """

    def __init__(self, env, period: float = 0.05):
        import threading

        self.arm, self.period = env.arm, period
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="rt-keepalive", daemon=True)

    def _run(self):
        lock = getattr(self.arm, "_setpoint_lock", None)
        while lock is not None and not self._stop.wait(self.period):
            with lock:
                self.arm._goal_ticks = 0

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=1.0)


# ---------------------------------------------------------------- 臂

def connect_arm():
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True     # suppleStop + 最不灵敏，触发不掉电
    env = AR5Env(cfg)
    env.connect()
    settle(env, gripper=0.0)
    return env


def step_and_settle(env, delta_xyz, grip=0.0):
    d = np.asarray(delta_xyz, float)
    n = np.linalg.norm(d)
    if n > A.COARSE_M:                              # 不超过缰绳（max_lead 15mm）
        d = d * (A.COARSE_M / n)
    env.step(np.r_[d, 0, 0, 0, grip])
    settle(env, gripper=grip)


# ---------------------------------------------------------------- 相机运动映射

def _frame(cam):
    for _ in range(20):
        f = cam.read()
        if f.ok and f.depth is not None:
            return f.color.copy(), f.depth.copy()
        time.sleep(0.03)
    raise RuntimeError("腕部相机取不到帧")


def scene_shift(cm, a, b):
    """两帧之间**静态场景**在相机系里的 3D 平移（米），几百个特征点的中位数。"""
    import cv2

    ga, gb = cv2.cvtColor(a[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(b[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(ga, 600, 0.01, 7)
    if pts is None or len(pts) < 30:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(21, 21), maxLevel=3)
    st = st.ravel().astype(bool)
    p0, p1 = pts.reshape(-1, 2)[st], nxt.reshape(-1, 2)[st]
    P0 = cm.points_keep(a[1], p0[:, 0], p0[:, 1])
    P1 = cm.points_keep(b[1], p1[:, 0], p1[:, 1])
    ok = np.isfinite(P0[:, 0]) & np.isfinite(P1[:, 0]) & (P0[:, 2] > FINGER_NEAR_M) & (P1[:, 2] > FINGER_NEAR_M)
    if ok.sum() < 30:
        return None, int(ok.sum())
    return np.median(P1[ok] - P0[ok], axis=0), int(ok.sum())


def measure_jacobian(env, cam, cm):
    """J[:,k] = 基座沿 k 轴移动 1m 时，静态场景点在相机系里的位移。正反各一次取平均。

    不用目标来测：第一版拿单个小球测，球挨着手指/套筒时测出列长 0.5~4.4（理论 1）。
    """
    J = np.zeros((3, 3))
    for k in range(3):
        for attempt in range(3):
            cols, ns = [], []
            for sign in (+1, -1):
                d = np.zeros(3); d[k] = sign * JOG_M
                a = _frame(cam)
                step_and_settle(env, d); time.sleep(0.35)
                b = _frame(cam)
                step_and_settle(env, -d); time.sleep(0.35)
                sh, n = scene_shift(cm, a, b)
                ns.append(n)
                if sh is not None:
                    cols.append(sh / (sign * JOG_M))
            print(f"   点动 {'xyz'[k]} 第{attempt + 1}次（特征点 {ns}）: " + "  ".join(
                f"{'+-'[i]}:{np.round(c, 2).tolist()}|{np.linalg.norm(c):.2f}" for i, c in enumerate(cols)))
            if len(cols) == 2 and np.linalg.norm(cols[0] - cols[1]) < 0.25 \
                    and 0.75 < np.linalg.norm(0.5 * (cols[0] + cols[1])) < 1.25:
                J[:, k] = 0.5 * (cols[0] + cols[1])
                break
        else:
            raise RuntimeError(f"{'xyz'[k]} 轴点动 3 次都测不出一致的结果")
    U, S, Vt = np.linalg.svd(J)
    print(f"   J 奇异值 {np.round(S, 3).tolist()}（刚体平移理论全为 1）")
    # 刚体约束：基座平移 b 让静态点在相机系里移动 -R b，J 必然是 -R（正交）。投影去噪。
    return U @ Vt


J_CACHE = Path(__file__).resolve().parent / "calibration" / "wrist_J.json"


def jacobian_cached(env, cam, cm):
    """现场测 J；测不出来就用上次测成功的。固定朝向下 J 是常量（十几次实测一致），
    但光流要纹理：俯视粗对准把臂带到板子远端时，腕部只看到白墙和素色地面，三次都测不出（9/28 水瓶）。
    朝向变了（pregrasp_pose.json 的 q 不同）缓存就不认。"""
    import json
    q_ref = json.loads((Path(__file__).resolve().parent / "calibration" / "pregrasp_pose.json").read_text())["q"]
    try:
        J = measure_jacobian(env, cam, cm)
        J_CACHE.write_text(json.dumps(dict(J=J.tolist(), q_ref=q_ref, date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
        return J
    except RuntimeError as exc:
        if J_CACHE.exists():
            d = json.loads(J_CACHE.read_text())
            if np.allclose(d["q_ref"], q_ref):
                print(f"   现场测不出（{exc}），用 {d['date']} 测的 J")
                return np.asarray(d["J"], float)
        raise


# ---------------------------------------------------------------- 阶段

def stage_locate(tf):
    from .camera import TopCamera

    with TopCamera() as cam:
        time.sleep(1.0)
        img = cam.read().color.copy()
    cands = tf._candidates(img, max_area_frac=0.25)
    if not cands:
        raise SystemExit(f"俯视相机里没找到「{tf.phrase}」（查询 {tf.queries}）")
    i, rank, m = tf._pick(img, cands, view="top")
    print(f"指令短语「{tf.phrase}」 查询 {tf.queries}")
    for c, rk, mm in sorted(zip(cands, rank, m), key=lambda t: (t[1], -t[2]))[:5]:
        print(f"   「{c.phrase}」det={c.conf:.2f} ({c.centre[0]:.0f},{c.centre[1]:.0f}) CLIP 名次={rk} 余量={mm:+.3f}")
    if i is None:
        raise SystemExit(f"CLIP 认为俯视画面里没有一个候选像目标（要求名次 <= {tf.TOP_K}）")
    H, meta = homography()
    u, v = cands[i].centre
    # 覆盖检查只管粗定位（挑预抓取姿态），真正的对准靠腕部深度，所以留 40px 余量
    if not in_coverage(meta, u, v, margin=40):
        raise SystemExit(f"目标像素 ({u:.0f},{v:.0f}) 在示范覆盖范围 {meta['px_range']} 之外，不抓")
    xy = pixel_to_xy(H, u, v)
    pre = choose_pregrasp(xy, np.asarray(meta["tip_offset_flange"], float))
    if pre is None:
        raise SystemExit("示范里挑不出预抓取姿态")
    print(f"选中 ({u:.0f},{v:.0f}) -> 粗定位 xy ({xy[0]:.3f},{xy[1]:.3f})；"
          f"预抓取姿态来自 {pre['src']} 帧 {pre['frame']}")
    return pre


def look_below_move(cam, cm, J):
    """算一步水平移动，让「夹持中心正下方的桌面点」落到腕部画面中心。全部现场测：

      桌面          腕部深度里夹持中心下方最高的一层大面积水平面，再 RANSAC
      基座竖直方向  J 的第 3 列（臂上移 1m，静态点在相机系里移动 J[:,2]，所以基座 +z = -J[:,2]）
      画面中心点    光轴和桌面的交点
      O             夹持中心沿竖直方向投到桌面 —— 俯视对准后物体大致在这里
      b = Jᵀ(p − O) 让静态点 O 移到 p 的臂移动；只取水平分量（高度不变）
    """
    from .objects import ransac_plane

    _, depth = _frame(cam)
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    Q = cm.points(depth, xs.ravel(), ys.ravel())
    Q = Q[Q[:, 2] > FINGER_NEAR_M]
    up = -J[:, 2]
    if len(Q) < 200:
        return None
    # 桌面 = 夹持中心下方、**水平方向离夹持中心最近**的那一层大面积水平面（夹爪悬在它上方）。
    # 不能取内点最多的平面：交接时相机朝前看，地面（再低 20cm）和墙占大半，桌面层只占 33%。
    # 也不能取最高的一层：远处别的水平面（高于泡沫板）会被选中，算出夹持中心离桌面只有 7mm。
    hg = (Q - GRASP_CENTER_CAM) @ up
    horiz = np.linalg.norm((Q - GRASP_CENTER_CAM) - np.outer(hg, up), axis=1)
    hist, edges = np.histogram(hg, bins=np.arange(-1.5, -0.02, 0.005))
    dense = np.nonzero(hist >= 0.03 * len(Q))[0]
    if len(dense) == 0:
        return None
    near = [np.percentile(horiz[np.abs(hg - edges[i] - 0.0025) < 0.0025], 5) for i in dense]
    level = edges[dense[int(np.argmin(near))]] + 0.0025
    print(f"   腕部深度里的水平层: " + "  ".join(
        f"{edges[i] * 1000:.0f}mm(离夹爪{d * 1000:.0f}mm)" for i, d in zip(dense, near)) +
        f"  -> 取 {level * 1000:.0f}mm")
    sub = Q[np.abs(hg - level) < 0.015]
    plane = ransac_plane(sub, min_frac=0.5) if len(sub) > 100 else None
    if plane is None:
        return None
    c, n = plane
    if abs(up @ n) < 0.9:
        return None                                   # 拟合出的平面不水平，不是桌面
    O = GRASP_CENTER_CAM - ((GRASP_CENTER_CAM - c) @ n) / (up @ n) * up
    p = np.array([0.0, 0.0, (c @ n) / n[2]])
    b = J.T @ (p - O)
    b[2] = 0.0
    return b, O, p


def acquire(env, tf, cam, J, look_below=False):
    rec.tag("acquire")
    """找到目标并把它挪到画面中部。

    look_below（俯视粗对准之后）：物体大致在夹持中心正下方，先按深度算一步移过去看，见 look_below_move。
    还看不到才兜底搜索：沿相机视线的水平方向往回退，再左右试。视线方向来自 J：基座沿 a 移动时
    静态点在相机系里沿 -z 移动，a = -J[2,:]。
    """
    if look_below:
        tf.last = None
        if tf.observe_median(cam, allow_track=False, margin=20) is None:
            r = look_below_move(cam, tf.cam, J)
            if r is None:
                print("   腕部深度里拟合不出桌面，直接兜底搜索")
            else:
                b, O, p = r
                print(f"   夹持中心正下方桌面点(相机系) {np.round(O * 1000).astype(int).tolist()}mm，"
                      f"画面中心桌面点 {np.round(p * 1000).astype(int).tolist()}mm -> 臂水平移 "
                      f"{np.round(b * 1000).astype(int).tolist()}mm")
                n = int(np.ceil(np.linalg.norm(b) / A.COARSE_M))
                for _ in range(max(n, 1)):
                    step_and_settle(env, b / max(n, 1))
                    tf.shift_track(J @ (b / max(n, 1)))
                time.sleep(0.2)
    a = -J[2, :2]
    a = a / np.linalg.norm(a)
    back, side = -a * 0.012, np.array([-a[1], a[0]]) * 0.012
    # 兜底：look_below 之后只剩俯视视差那几厘米，搜小范围；没有 look_below 时保持原来的范围
    moves = ([tuple(back)] * 5 + [tuple(side)] * 3 + [tuple(-side)] * 6) if look_below else \
            ([tuple(back)] * 20 + [tuple(side)] * 4 + [tuple(-side)] * 8)
    tf.last = None
    obs, k = tf.observe_median(cam, allow_track=False, margin=20), 0
    while obs is None and k < len(moves):
        step_and_settle(env, [moves[k][0], moves[k][1], 0.0]); time.sleep(0.2)
        tf.shift_track(J @ np.array([moves[k][0], moves[k][1], 0.0]))     # 点选模式：按 J 预测物体在画面里挪到哪
        obs = tf.observe_median(cam, allow_track=False, margin=20)
        k += 1
    if obs is None:
        return None
    if k:
        print(f"   搜索 {k} 步后看到目标")
    # 挪到画面中部：让相机光轴对准目标（深度不变）。按 J 换算成基座运动。
    for it in range(6):
        u, v = obs.pixel
        if abs(u - 320) < 90 and abs(v - 240) < 80:
            break
        vcam = np.array([obs.centre[0], obs.centre[1], 0.0])       # 相机该朝目标横移的量
        b = J.T @ (-vcam)
        step_and_settle(env, b); time.sleep(0.2)
        tf.shift_track(J @ b)
        o2 = tf.observe_median(cam, allow_track=True)
        if o2 is None:
            break
        obs = o2
    return obs


def finger_zone_count(cam, cm, width):
    """两指之间那块区域里的深度点数（相机系）：x 在两指内侧面之间（按物体宽度），y、z 在手指根部到指尖之间。

    用来判「抓着东西没有」：合爪前数一次、抬起后数一次，物体跟着走点数就还在。看得到物体时用表面位移判，
    看不到时（水瓶抓瓶盖那次，瓶盖贴着镜头认不出，真抓住了却报「看不到」判失败）用这个。
    """
    from .calibrate_gripper import load as load_gripper

    g = load_gripper()
    root = np.asarray(g["root_cam"]) if g else np.array([0.009, 0.031, 0.056])
    tip = np.asarray(g["tip_cam"]) if g else np.array([0.011, 0.039, 0.097])
    xc = 0.5 * (root[0] + tip[0])
    half = 0.5 * width - 0.002                     # 让开手指内侧面本身
    if half <= 0.002:
        return 0
    _, depth = _frame(cam)
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:2, 0:w:2]
    P = cm.points(depth, xs.ravel(), ys.ravel())
    inside = (np.abs(P[:, 0] - xc) < half) \
        & (P[:, 1] > min(root[1], tip[1]) - 0.012) & (P[:, 1] < max(root[1], tip[1]) + 0.012) \
        & (P[:, 2] > root[2]) & (P[:, 2] < tip[2] + 0.005)
    return int(inside.sum())


def stage_grasp(env, tf, cam, J, max_steps=45, grip_plan=None):
    # 伺服阶段 margin=0：越接近，物体在画面里越大，最后几厘米本来就会顶到画面下沿。
    # 「框碰边就不算」只该用在搜索阶段 —— 用在这里时，充电器和胶带卷都在离抓取点 2~3cm 时被丢掉
    # （胶带卷那次 SAM 分割和高度都是对的，只因框下沿到了 474/480）。
    # 也要连续：接着交接时认出的那个（tf.last）往下跟。不连续的话 CLIP 会重新挑，9/29 鸡蛋这里挑中了
    # 框住整片桌面的大框（174×132mm），之后的连续性反而一直锁在它上面
    obs = tf.observe_median(cam, allow_track=True, margin=0, continuity=True)
    if obs is None:
        print("对准开始前看不到目标")
        return False
    for step in range(max_steps):
        obs = tf.observe_median(cam, allow_track=True, margin=0, continuity=True)
        if obs is None:
            print(f"[{step:2d}] 看不到目标，停")
            return False
        e = obs.grasp - GRASP_CENTER_CAM                 # 抓取点（顶部那一截），不是物体中心
        if np.linalg.norm(e) < ALIGN_TOL_M:
            print(f"[{step:2d}] 目标中心离夹持中心 {np.round(e * 1000, 1).tolist()} mm，到位")
            break
        exy = e[:2].copy()
        # 只把占主导的分量画给 PlayJev：小于最大分量 30% 的先不画。两个都画时它常先修小的那个
        # （ex≈+60mm、ey≈+4mm 连选 30 多次 down，每步只走几毫米，伺服用满 40 步）。
        exy[np.abs(exy) < max(AXIS_DEADBAND_M, 0.3 * float(np.max(np.abs(exy))))] = 0.0
        v = np.zeros(3)
        what = "画面内已对准"
        if np.any(exy != 0):
            g, t = np.zeros(2), np.array([-exy[1], -exy[0]])       # 棋盘上 = 画面上(-y)，左 = 画面左(-x)
            img, _, _ = render(g, t, adaptive_board(g, t))
            move = ask(img, INSTRUCTIONS, OPTIONS)["answers"]["move"]["choice"]
            axis, sgn = {"up": (1, -1), "down": (1, +1), "left": (0, -1), "right": (0, +1)}[move]
            err = abs(exy[axis])
            v[axis] = sgn * (A.COARSE_M if err > A.COARSE_M else min(A.FINE_M, err))
            what = f"PlayJev={move:5s} {'✓' if np.sign(v[axis]) == np.sign(exy[axis]) else '✗'}"
        # 沿光轴前进只在「物体已落在两指之间」时做：开合方向余量 = (满开 − 实测宽)/2，
        # 另一方向不超过物体自身半尺寸。套筒那次横向还差 48mm（余量只有 29mm）就往前推，
        # 一根手指插进了套筒孔里，之后臂被顶住（leashed=154），剩下 25 步全白走。后退不受限。
        room_x = max(0.0, (GRIPPER_FULL_OPEN_M - float(obs.grasp_extent[0])) / 2)
        inside = abs(e[0]) < room_x and abs(e[1]) < float(obs.grasp_extent[1]) / 2
        if abs(e[2]) > AXIS_DEADBAND_M and (e[2] < 0 or inside):
            v[2] = float(np.clip(e[2], -A.COARSE_M, A.COARSE_M))
        b = J.T @ (-v)                                  # 相机移动 v <=> 静态点在相机系移动 -v
        rec.tag(f"servo{step:02d}")
        print(f"[{step:2d}] 中心-夹持 {np.round(e * 1000).astype(int).tolist()} mm "
              f"抓取处尺寸 {np.round(obs.grasp_extent * 1000).astype(int).tolist()} mm [{obs.source}]  {what}  "
              f"臂走 {np.round(b * 1000).astype(int).tolist()} mm  "
              f"法兰 {np.round(np.asarray(env.arm.get_ee_pose())[:3] * 1000).astype(int).tolist()}")
        leashed0, unreach0 = env.stats["leashed"], env.stats["unreachable"]
        step_and_settle(env, b)
        tf.shift_track(J @ b)                           # 点选模式：静态物体在相机系里移动 J·b
        # 目标超出关节范围被拒 = 固定朝向下够不着（离底座太近/太远）。不停的话同一指令空转到步数用完
        # （套筒放在离底座 0.24m 处：j4 顶到 142°，38 次被拒，45 步白走）。
        if env.stats["unreachable"] > unreach0:
            print(f"[{step:2d}] 这个朝向下够不着（逆解超关节范围），停止伺服")
            return False
        # 每步不超过 12mm < 缰绳 15mm，正常走完缰绳从不触发（成功的几次都是 0）。
        # 触发 = 实际位置落后指令 = 被东西顶住了，再推只会压得更狠。
        if env.stats["leashed"] > leashed0:
            print(f"[{step:2d}] 臂跟不上指令（被顶住），停止伺服")
            return False
    else:
        print("步数用完仍未到位")
        return False

    width = float(obs.grasp_extent[0])                   # 抓取那一截的宽度；相机 x ≈ 手指开合方向
    return close_lift_judge(env, width, grip_plan, WristFollows(tf, cam, width, obs.surface.copy(), box=obs.xyxy))


class WristFollows:
    """「物体跟着手走了吗」—— 腕部看：物体可见表面在相机系里几乎不动 = 跟着走（相机和夹爪一起动）。
    看不到时（贴着镜头）改看两指之间的深度点。阈值取抬起距离的一半，来自动作本身，与物体无关。

    可见表面直接用合爪时那个框给 SAM 分割、取掩码点的中位（box 固定在相机画面里，物体跟着走就还在框里）。
    9/30 以前走 observe()：它按桌面算中心 / 抓取点，物体一离开桌面，这两个量跳好几厘米，被连续性门槛拒掉，
    真夹住了也报「看不到物体」（长手指时后备的两指间点数又因为合爪前算进了物体后面的桌面而掉光）。"""

    def __init__(self, tf, cam, width, surface0, box=None):
        self.tf, self.cam, self.width = tf, cam, width
        self.box = None if box is None else np.asarray(box, float)
        self.zone0 = finger_zone_count(cam, tf.cam, width)
        s0 = self._surface()
        self.before = s0 if s0 is not None else surface0

    def _surface(self):
        if self.box is None:
            o = self.tf.observe_median(self.cam, allow_track=True, margin=0, continuity=True)
            return None if o is None else o.surface.copy()
        ss = []
        for _ in range(8):
            f = self.cam.read()
            if not (f.ok and f.depth is not None):
                continue
            m = self.tf._segment(f.color, self.box)
            if m is None or m.sum() < 30:
                continue
            ys, xs = np.nonzero(m)
            P = self.tf.cam.points(f.depth, xs.astype(float), ys.astype(float))
            if len(P) >= 30:
                P = P[(P[:, 2] - np.median(P[:, 2])) < 0.10]
                ss.append(np.median(P, axis=0))
            if len(ss) >= 3:
                break
        return np.median(ss, axis=0) if len(ss) >= 2 else None

    def rebase(self):
        s = self._surface()
        if s is not None:
            self.before = s

    def __call__(self, lift):
        s = self._surface()
        if s is None:
            z = finger_zone_count(self.cam, self.tf.cam, self.width)
            return (self.zone0 >= 50 and z >= 0.3 * self.zone0), f"腕部看不到物体，两指之间深度点 {self.zone0}->{z}"
        moved = float(np.linalg.norm(s - self.before))
        return moved < 0.5 * lift, f"腕部看物体表面相对相机移动 {moved * 1000:.0f}mm"


class TopFollows:
    """「物体跟着手走了吗」—— 顶部 335 看：物体在基座系里升高了超过抬起距离一半 = 跟着走。
    腕部对这个物体测不准时用（透明物体斜着看，深度会挪到物体旁边去）。"""

    def __init__(self, tf, cam335, cm335, R, t, before=None, click=False):
        self.tf, self.cam, self.cm, self.R, self.t, self.click = tf, cam335, cm335, R, t, click
        self.before, self.px0 = None, None
        self.before = before if before is not None else self._pos()

    def _pos(self, lift=0.0):
        from .top335 import locate, to_flange
        if self.click and self.before is not None:
            return self._pos_point(lift)
        # 只认原来那个位置上的东西（按基座系水平位置）：335 视野里可能还有别的同类物体，没有工作空间盒挡着了
        r = locate(self.tf, self.cam, self.cm, self.R, self.t, n=3,
                   anchor_xy=None if self.before is None else self.before[:2])
        # 用可见表面中位点，不用抓取点：抓取点按「离桌面多高」定义，物体被提起来它的高度也不变
        return None if r is None else to_flange(r[0].surface, self.R, self.t)

    def _pos_point(self, lift):
        """点选模式：物体没有名字，按「跟着走的话它该在的位置」（原位置 + 抬起量）投影到 335 画面给 SAM 点提示。
        面积和抬起前差一倍以上的不认（多半分到了手指上）。"""
        from .top335 import measure_mask, to_flange
        p335 = self.R @ (np.asarray(self.before, float) + np.array([0.0, 0.0, lift])) + self.t
        px = self.cm.project(p335)[0]
        got = []
        for _ in range(10):
            f = self.cam.read()
            if not (f.ok and f.depth is not None):
                continue
            ms = self.tf.segment_prompt(f.color, points=[tuple(px)])
            if not ms:
                continue
            r = measure_mask(self.tf, f.depth, ms[0], self.cm, self.R, self.t)
            if r is None:
                continue
            if self.px0 and not 0.5 <= ms[0].sum() / self.px0 <= 2.0:
                continue
            got.append(to_flange(r[0].surface, self.R, self.t))
            if len(got) >= 3:
                break
        return np.median(got, axis=0) if got else None

    def rebase(self):
        p = self._pos()
        if p is not None:
            self.before = p

    def __call__(self, lift):
        p = self._pos(lift)
        if p is None or self.before is None:
            return None, "335 看不到物体"
        dz = float((p - self.before)[2])
        return dz > 0.5 * lift, f"335 看物体升高 {dz * 1000:.0f}mm"


def close_lift_judge(env, width, grip_plan, follows):
    """合爪 → 抬到 60mm → 报告成败。follows(lift) -> (跟着走?, 说明)。

    力度只由 grip_plan（CLIP：腕部截图 + 物体名提示词判出的易碎 / 软 / 硬）决定：先快速合到「比抓取处宽
    PRE_GAP_M」（碰不到），再按这一类的力度和速度合一次，直接抬。9/29 按用户要求去掉了滑落检测
    （原来从 1 档起、试抬 15mm 看跟没跟着走、滑了放回加一档）。
    抬起后的「跟没跟着走」只用来报告这次成没成功，不再改变力度。看不到物体时的兜底：抬起后再读一次夹爪 ——
    物体掉了手指会接着合到底（读数 ≥0.97）。
    """
    from .grip_force import PRE_GAP_M

    expect = float(np.clip(1.0 - width / GRIPPER_FULL_OPEN_M, 0.0, 1.0))

    def hold(v, s):
        t_end = time.time() + s
        while time.time() < t_end:
            env.step(A.hold(v)); time.sleep(0.05)

    pre = float(np.clip(1.0 - (width + PRE_GAP_M) / GRIPPER_FULL_OPEN_M, 0.0, 1.0))
    rec.tag("close")
    env.hand.set_grip_params(speed=1000)
    hold(pre, 1.2)
    F = grip_plan.force if grip_plan is not None else None
    if F is not None:
        env.hand.set_grip_params(force=F, speed=grip_plan.speed)
    hold(1.0, 2.5 if (grip_plan is not None and grip_plan.speed <= 300) else 2.0)
    grip = float(env.hand.get_opening())
    print(f"   力 {F}（读回 {env.hand.force_readback()}）: 合爪读数 {grip:.2f}"
          + ("，合到底了 —— 两指之间没东西" if grip >= 0.97 else ""))
    lift = 5 * A.COARSE_M
    rec.tag("lift")
    n = max(1, int(round(lift / A.COARSE_M)))
    for _ in range(n):
        step_and_settle(env, [0, 0, lift / n], grip=1.0)
    time.sleep(0.3)
    ok, txt = follows(lift)
    g_after = float(env.hand.get_opening())
    if ok is None:
        ok, txt = g_after < 0.97, txt + f"；抬起后夹爪读数 {g_after:.2f}"
    grip_ok = grip < 0.97 and abs(grip - expect) < 0.2 and g_after < 0.97
    print(f"宽度 {width * 1000:.0f}mm -> 应读 {expect:.2f}，实读 {grip:.2f}（抬起后 {g_after:.2f}） {'✓' if grip_ok else '✗'}；"
          f"力 {F if F is not None else '—'}（{grip_plan.kind if grip_plan else '—'}）；"
          f"抬起 {lift * 1000:.0f}mm 后 {txt}  {'✓' if ok else '✗'}  -> {'成功' if grip_ok and ok else '失败'}")
    RUN.update(kind=grip_plan.kind if grip_plan else None, force=F, speed=grip_plan.speed if grip_plan else None,
               scores=grip_plan.scores if grip_plan else None, grip=grip, expect=expect, width=width,
               result="成功" if grip_ok and ok else "失败", closed=True)
    return bool(grip_ok and ok)


def print_summary():
    """运行结束时在终端最后打印：这次采取的力度模式（CLIP 判的类别和三类得分 -> 档位、速度）、抓取位姿、结果。"""
    print("=" * 60)
    if RUN.get("target"):
        print(f"目标：{RUN['target']}")
    if RUN.get("force") is None and RUN.get("kind") is None:
        print("力度模式：本次没有合爪" + (f"（{RUN['stop']}）" if RUN.get("stop") else ""))
    else:
        sc = RUN.get("scores") or {}
        print(f"力度模式：「{RUN['kind']}」-> 力 {RUN['force']} 档、合爪速度 {RUN['speed']}"
              + ("（CLIP 得分 " + " / ".join(f"{k} {v:.3f}" for k, v in sc.items()) + "）" if sc else "")
              + ("" if RUN.get("closed", True) else "  —— 判定了，但这次没走到合爪"))
    if RUN.get("pose"):
        print(f"抓取位姿：「{RUN['pose']}」，离支撑面 {RUN['z'] * 1000:.0f}mm")
    if RUN.get("result"):
        print(f"结果：{RUN['result']}（夹爪读数 {RUN['grip']:.2f}，按宽度 {RUN['width'] * 1000:.0f}mm 应读 {RUN['expect']:.2f}）")
    print("=" * 60)


def grasp_open_loop(env, tf, cam, top, cm335, ctx, click=False):
    """腕部测不准时：抓取力度/位姿用 335 的测量定，臂从当前（预抓取点）沿直线走到 335 给的抓取点，合爪。
    每段 ≤12mm，被顶住或超关节范围就停。成败用 335 看物体有没有升高。"""
    from atom_robot.ar5.kinematics import AR5Kinematics

    from .grip_force import grasp_height, judge
    from .top335 import to_flange

    o, R, t = ctx["obs"], ctx["R"], ctx["t"]
    up = R[:, 2] / np.linalg.norm(R[:, 2])
    h = o.extra.get("height")
    img = top.read().color
    plan = judge(tf, img, o.xyxy, tf.phrase)
    if FORCE_OVERRIDE is not None:
        plan.force = FORCE_OVERRIDE
    RUN.update(kind=plan.kind, force=plan.force, speed=plan.speed, scores=plan.scores, closed=False)
    z_top = o.extra.get("z_g")
    z, used_pose, why = grasp_height(plan.pose, h, z_top, tf.grasp_band)
    RUN.update(pose=used_pose, z=z)
    z_ref = z_top if z_top is not None else 0.5 * h
    p = o.grasp + (z - z_ref) * up                     # 335 的抓取点沿竖直挪到选定高度
    f_target = to_flange(p, R, t)
    print(f"抓取力度：判为「{plan.kind}」-> 力 {plan.force}、速度 {plan.speed}；抓取位姿：CLIP 选「{plan.pose}」-> 用「{used_pose}」，"
          f"离桌面 {z * 1000:.0f}mm（335 量物体高 {h * 1000:.0f}mm）")
    for w in why:
        print(f"   不用{w}")
    kin = AR5Kinematics.from_robot(env.arm)
    f_now = kin.fk(np.asarray(env.arm.get_joint_positions(), float))[:3, 3]
    d = f_target - f_now
    n = int(np.ceil(np.linalg.norm(d) / A.COARSE_M))
    print(f"   直线接近：法兰 {np.round(f_now * 1000).astype(int).tolist()} -> {np.round(f_target * 1000).astype(int).tolist()}mm，{n} 段")
    rec.tag("open_loop")
    for k in range(n):
        leashed0, unreach0 = env.stats["leashed"], env.stats["unreachable"]
        step_and_settle(env, d / n)
        if env.stats["unreachable"] > unreach0 or env.stats["leashed"] > leashed0:
            print(f"   第 {k + 1} 段{'超关节范围' if env.stats['unreachable'] > unreach0 else '被顶住'}，停")
            return False
    # 335 量的宽度是沿它自己画面的横向，不一定是夹爪开合方向：取两个方向里大的（预张开留得更宽，更安全）
    width = float(np.max(o.grasp_extent))
    follows = TopFollows(tf, top, cm335, R, t, before=to_flange(o.surface, R, t), click=click)
    follows.px0 = o.mask_px if click else None           # 点选模式：面积门槛以点选时 335 里的掩码为准
    return close_lift_judge(env, width, plan, follows)


def main() -> int:
    global GRIPPER_FULL_OPEN_M
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", default=None, help="要抓什么（英文）；--click 时不用给")
    ap.add_argument("--click", action="store_true",
                    help="不给指令，在 335 的一帧里用鼠标点要抓的物体（robojev.click_grasp）；只配 --locate top335")
    ap.add_argument("--stage", choices=["locate", "pregrasp", "grasp"], default="locate")
    ap.add_argument("--locate", choices=["top335", "top", "demo", "wrist"], default="top335",
                    help="top335=俯视 Gemini 335 看一眼定位（默认，要先有 calibrate_top335 的外参）；"
                         "top=旧的俯视 UGREEN 像素闭环（UGREEN 已换下，别用）；demo=示范单应性；"
                         "wrist=不用俯视相机，到起点后只靠腕部找")
    ap.add_argument("--calib-gripper", action="store_true",
                    help="开始时在 0 位用腕部相机量一遍夹爪几何（满开、手指根部、指尖），写 calibration/gripper.json")
    ap.add_argument("--record", default=None, help="把「俯视|腕部」每 0.25s 存一帧到这个目录，事后对照日志")
    ap.add_argument("--force", type=int, default=None,
                    help="只这一次：合爪力度档位（1~100），覆盖 CLIP 按易碎/软/硬判出的值。9/30 标定抓球用过 30")
    ap.add_argument("--finger-floor", type=float, default=None,
                    help="只这一次：两指中间最低离支撑面多高（米），覆盖按夹爪标定算的值。"
                         "9/30 长手指在泡沫板上抓 40mm 的球时用过 0.012（指尖会压进泡沫约 6mm，硬桌面别用）")
    a = ap.parse_args()
    if not a.click and not a.instruction:
        ap.error("要么给 --instruction，要么用 --click 点选")
    if a.click and a.locate != "top335":
        ap.error("--click 只配 --locate top335（点选用的是 335 那一帧）")
    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(message)s")
    import warnings
    warnings.filterwarnings("ignore")

    from .objects import TargetFinder

    tf = TargetFinder(a.instruction or "pick up the object")
    import json as _json
    from . import grip_force
    from .calibrate_gripper import apply_calibration
    apply_calibration(np.asarray(_json.loads(J_CACHE.read_text())["J"], float))
    global FORCE_OVERRIDE
    FORCE_OVERRIDE = a.force
    if a.force is not None:
        print(f"⚠ --force：合爪力度固定 {a.force} 档，只这一次（不按 CLIP 判的类别）")
    if a.finger_floor is not None:
        print(f"⚠ --finger-floor：离支撑面下限 {grip_force.FINGER_FLOOR_M * 1000:.0f}mm（按标定）-> {a.finger_floor * 1000:.0f}mm，只这一次；"
              f"指尖会低于支撑面 {(grip_force.FINGER_FLOOR_M - 0.002 - a.finger_floor) * 1000:.0f}mm")
        grip_force.FINGER_FLOOR_M = a.finger_floor
    print(f"夹爪标定：夹持中心 G {np.round(GRASP_CENTER_CAM * 1000, 1).tolist()}mm（相机系）；"
          f"两指中间最低离支撑面 {grip_force.FINGER_FLOOR_M * 1000:.0f}mm（= 指尖在 G 下方多深 + 2mm）")
    RUN.clear()
    RUN["target"] = a.instruction
    if a.locate == "demo":
        pre = stage_locate(tf)
        if a.stage == "locate":
            return 0

    from .camera import OrbbecRGBD, TopCamera
    from .move_pregrasp import slow_goto

    cam = OrbbecRGBD()
    cam.connect()                 # 先开相机（含 USB 复位），再连臂
    top = None
    if a.locate == "top":
        top = TopCamera(); top.connect(); time.sleep(1.0)
    elif a.locate == "top335":
        from .camera import TOP_SERIAL
        from .objects import CameraModel
        top = OrbbecRGBD(serial=TOP_SERIAL); top.connect()
        cm335 = CameraModel(Path(__file__).resolve().parent / "calibration" / "orbbec335_params.json")
    sel = None
    if a.click:
        # 点选在连臂之前：人在犹豫选哪个的时候臂不该连着
        from .calibrate_gripper import load as _load_gripper
        from .click_grasp import select_from_camera
        sel = select_from_camera(tf, top, ((_load_gripper() or {}).get("full_open_m")) or GRIPPER_FULL_OPEN_M)
        if sel is None:
            print("没选，退出")
            cam.disconnect(); top.disconnect()
            return 0
        tf.set_instruction(f"pick up the {sel['name']}")   # 名字只给抓取力度 / 位姿的 CLIP 判断当参考
        RUN["target"] = f"点选的物体（CLIP 叫它 {sel['name']}）"
    recorder = None
    if a.record:
        from .rec import Recorder
        recorder = Recorder(Path(a.record), [c for c in (top, cam) if c is not None]).__enter__()
        print(f"录像 -> {a.record}")
    env = connect_arm()
    ka = KeepAlive(env).__enter__()
    how = None
    ctx335 = None
    try:
        if a.locate == "top":
            # 先回 0 位让臂离开俯视画面：上次失败时臂停在物体上方，俯视里物体被挡住就找不到
            import json as _json
            centre = np.asarray(_json.loads((Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101" / "atom_robot" / "ar5"
                                             / "joint_calibration.json").read_text())["centre_joints"], float)
            slow_goto(env, centre)
            if a.calib_gripper:
                from .calibrate_gripper import measure
                print("标定夹爪几何（0 位，只动夹爪）:")
                measure(env, cam, tf.cam, GRASP_CENTER_CAM)
            from .top_servo import object_pixel
            obj_px, cand = object_pixel(tf, top)
            print(f"俯视里物体像素 {np.round(obj_px).astype(int).tolist()}")
            if a.stage == "locate":
                return 0
        if a.locate == "demo":
            reached, el = slow_goto(env, np.asarray(pre["q"], float))
            print(f"到预抓取姿态: {reached}  {el:.1f}s")
        elif a.locate == "top335":
            import json as _json
            from .calibrate_top import path_min_z
            from .top335 import approach as approach335
            centre = np.asarray(_json.loads((Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101" / "atom_robot" / "ar5"
                                             / "joint_calibration.json").read_text())["centre_joints"], float)
            slow_goto(env, centre)
            if a.calib_gripper:
                from .calibrate_gripper import measure
                print("标定夹爪几何（0 位，只动夹爪）:")
                measure(env, cam, tf.cam, GRASP_CENTER_CAM)
            # 规划用缓存的 J（固定朝向下测的；0 位朝向不同，现场测的用不上）；到预抓取点后再现场测
            J_plan = np.asarray(_json.loads(J_CACHE.read_text())["J"], float)
            from .calibrate_gripper import load as _load_gripper
            full_open = ((_load_gripper() or {}).get("full_open_m")) or GRIPPER_FULL_OPEN_M
            ctx335 = approach335(env, tf, top, cm335, J_plan, slow_goto, step_and_settle, path_min_z,
                                 full_open=full_open, click=sel)
            if ctx335 is None:
                return 1
            if a.stage == "locate":
                return 0
            how = "seen"
        elif a.locate == "wrist":
            import json as _json
            from .top_servo import goto_start
            centre = np.asarray(_json.loads((Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101" / "atom_robot" / "ar5"
                                             / "joint_calibration.json").read_text())["centre_joints"], float)
            slow_goto(env, centre)
            goto_start(env, slow_goto)
            how = "seen"                     # 不做「按深度后退」那一步（它假设俯视已对准）
        else:
            from .top_servo import approach
            how = approach(env, tf, top, cam, step_and_settle, slow_goto, obj_px)
            if not how:
                return 1
        print("测相机运动映射（静态场景光流）:")
        J = jacobian_cached(env, cam, tf.cam)
        tf.up = -J[:, 2] / np.linalg.norm(J[:, 2])       # 腕部相机系里的基座竖直方向：支撑面只找水平面
        from .calibrate_gripper import band_from, load as load_gripper
        g = load_gripper()
        if g is not None:
            if g.get("full_open_m"):
                GRIPPER_FULL_OPEN_M = g["full_open_m"]
            tf.grasp_band = max(0.0, band_from(g, GRASP_CENTER_CAM, J))
            print(f"   夹爪（{g['date']} 实测）：满开 {GRIPPER_FULL_OPEN_M * 1000:.0f}mm，"
                  f"夹持中心往上到手指根部 {tf.grasp_band * 1000:.0f}mm（= 抓顶部时伸进去的深度）")
        if sel is not None:
            # 点选的目标没有可靠的名字：按 335 测的抓取点投影到腕部画面，在那里给 SAM 点提示，之后一路按位置跟。
            # 静态点在相机系里随臂移动 Δp = J·Δf；臂到 f_grasp 时抓取点正好在夹持中心 G，所以现在它在 G − J·(f_grasp − f_now)
            f_now = np.asarray(env.arm.get_ee_pose(), float)[:3]
            tf.track_xyz = GRASP_CENTER_CAM - J @ (ctx335["plan"]["f_grasp"] - f_now)
            tf.track_px = tuple(tf.cam.project(tf.track_xyz)[0])
            print(f"   点选目标投影到腕部画面：相机系 {np.round(tf.track_xyz * 1000).astype(int).tolist()}mm -> "
                  f"像素 {np.round(tf.track_px).astype(int).tolist()}")
            # 投影点可能落在物体旁边的桌面上（外参 ~2mm + J 的误差），那样一路跟的就是桌面。先看一眼：
            # 腕部分出来的抓取点要和 335 预测的在 30mm 内，否则不用腕部，走 335 直线接近（腕部测不准时本来就走这条）
            p_pred = tf.track_xyz.copy()
            tf.last = None
            o0 = tf.observe_median(cam, allow_track=False, margin=0)
            d0 = None if o0 is None else float(np.linalg.norm(o0.grasp - p_pred))
            print(f"   腕部在投影处分割：" + ("没分出来" if o0 is None else
                  f"{o0.mask_px}px，抓取点离 335 的预测 {d0 * 1000:.0f}mm"))
            if o0 is None or d0 > 0.03:
                print("   腕部认不准点选的目标，改用 335 直线接近")
                if a.stage == "pregrasp":
                    return 0
                return 0 if grasp_open_loop(env, tf, cam, top, cm335, ctx335, click=True) else 1
        # 腕部在粗对准时已经看到物体就不用再按深度后退（那一步假设物体在夹持中心正下方）
        obs = acquire(env, tf, cam, J, look_below=(a.locate == "top" and how == "aligned"))
        # 交叉验证：腕部和顶部 335 量出的物体高度差太多 = 腕部这次测不准（透明物体斜着看，深度会挪到物体旁边；
        # 9/29 蓝盖水瓶 335 量 181mm、腕部量 5mm）。那就不走腕部伺服，用 335 的抓取点（外参 ~2mm）直线接近
        wrist_ok = obs is not None
        if ctx335 is not None:
            h335 = ctx335["obs"].extra.get("height")
            hw = None if obs is None else obs.extra.get("height")
            wrist_ok = (hw is not None and h335 is not None and abs(hw - h335) <= max(0.02, 0.25 * h335))
            print(f"交叉验证：335 量高度 {'—' if h335 is None else f'{h335 * 1000:.0f}mm'}，腕部量 "
                  f"{'—' if hw is None else f'{hw * 1000:.0f}mm'} -> {'一致，用腕部伺服' if wrist_ok else '不一致，腕部这次测不准，改用 335 直线接近'}")
            if not wrist_ok:
                if a.stage == "pregrasp":
                    return 0
                return 0 if grasp_open_loop(env, tf, cam, top, cm335, ctx335, click=sel is not None) else 1
        if obs is None:
            print("找不到目标")
            return 1
        h = obs.extra.get("height")
        print(f"目标中心(相机系) {np.round(obs.centre * 1000).astype(int).tolist()} mm  "
              f"尺寸 {np.round(obs.extent * 1000).astype(int).tolist()} mm  "
              f"高度 {'拟合不出支撑面' if h is None else f'{h * 1000:.0f} mm'}  像素 {np.round(obs.pixel).tolist()}\n"
              f"抓取点(相机系) {np.round(obs.grasp * 1000).astype(int).tolist()} mm  "
              f"抓取处尺寸 {np.round(obs.grasp_extent * 1000).astype(int).tolist()} mm")
        # 抓取力度 + 抓取位姿（在 PlayJev 伺服之前定）：CLIP 按物体名 + 腕部截图判断；位姿再过几何检查。
        # 抓取高度在这里（离物体 20~30cm、侧面看得清）定下来并冻结；贴近后只看得到顶面，逐层宽度量不准
        from .grip_force import grasp_height, judge
        grip_plan = judge(tf, cam.read().color, obs.xyxy, tf.phrase)
        if FORCE_OVERRIDE is not None:
            grip_plan.force = FORCE_OVERRIDE
        RUN.update(kind=grip_plan.kind, force=grip_plan.force, speed=grip_plan.speed, scores=grip_plan.scores, closed=False)
        print(f"抓取力度：判为「{grip_plan.kind}」（" + " ".join(f"{k} {v:.3f}" for k, v in grip_plan.scores.items())
              + f"）-> 力 {grip_plan.force}、合爪速度 {grip_plan.speed}")
        if h is not None:
            sup = obs.extra.get("support") or {}
            own = obs.extra.get("height_own")
            print(f"支撑面：{'就近那一层' if sup else '周围点最多的平面（没有竖直方向时的旧做法）'}"
                  + (f"（物体脚下共 {sup.get('n_layers')} 个水平层，取紧挨物体的）" if sup else "")
                  + f"；物体高 {h * 1000:.0f}mm，自身高度 {'—' if own is None else f'{own * 1000:.0f}mm'}"
                  + ("" if obs.extra.get("support_ok") in (True, None) else " ⚠ 支撑面在物体底部以上，不可信"))
            # 位姿：CLIP 选的先试；几何做不到或手指扫掠体里有别的东西（盘沿、碗壁、旁边的物体），按就近顺序换
            from .calibrate_gripper import load as load_gripper
            from .clearance import check as clearance_check
            g_geom = load_gripper()
            f_now = cam.read()
            mask_now = tf._segment(f_now.color, obs.xyxy)
            z_obs = obs.extra.get("z_g") if obs.extra.get("z_g") is not None else 0.5 * h
            order = {"上": ("上", "中", "下"), "中": ("中", "上", "下"), "下": ("下", "中", "上")}[grip_plan.pose]
            tried, chosen = set(), None
            for want in order:
                z, used_pose, why = grasp_height(want, h, obs.extra.get("z_g"), tf.grasp_band)
                if used_pose in tried:
                    continue
                tried.add(used_pose)
                for w in why:
                    print(f"   不用{w}")
                if g_geom is None or mask_now is None:
                    chosen = (z, used_pose)
                    break
                p_z = obs.grasp + (z - z_obs) * tf.up
                res = clearance_check(tf.cam, f_now.depth, mask_now, p_z, GRASP_CENTER_CAM,
                                      float(np.max(obs.grasp_extent)), g_geom, tf.up,
                                      support=(obs.grasp - z_obs * tf.up, tf.up))
                print(f"   手指扫掠检查（{used_pose}，离支撑面 {z * 1000:.0f}mm）: 碰撞点 {res['hits']}"
                      + (f" {'；'.join(res['where'])}" if res['where'] else "") + f" -> {'可以' if res['ok'] else '会碰'}")
                if res["ok"]:
                    chosen = (z, used_pose)
                    break
            if chosen is None:
                print("每个可行的抓取位姿，手指都会碰到别的东西，不抓")
                return 1
            z, used_pose = chosen
            tf.grasp_height = z
            RUN.update(pose=used_pose, z=z)
            print(f"抓取位姿：CLIP 选「{grip_plan.pose}」（" + " ".join(f"{k} {v:.3f}" for k, v in grip_plan.pose_scores.items())
                  + f"）-> 用「{used_pose}」，抓取高度冻结在离支撑面 {z * 1000:.0f}mm（物体高 {h * 1000:.0f}mm）")
            if getattr(tf, "_last_profile", None) and used_pose == "上":
                print("   顶部逐层宽度（离桌面 mm: 宽 mm）: " + " ".join(f"{zz}:{ww}" for zz, ww in tf._last_profile))
        if a.stage == "pregrasp":
            return 0
        ok = stage_grasp(env, tf, cam, J, grip_plan=grip_plan)
        print(f"env 统计: unreachable={env.stats['unreachable']} leashed={env.stats['leashed']}")
        return 0 if ok else 1
    finally:
        ka.__exit__(None, None, None)
        if recorder is not None:
            recorder.__exit__(None, None, None)
            print(f"录了 {recorder.n} 帧")
        env.close()
        cam.disconnect()
        if top is not None:
            top.disconnect()
        print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
