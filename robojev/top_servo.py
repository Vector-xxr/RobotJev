"""俯视相机图像空间闭环粗对准：把夹爪带到物体上方，直到腕部相机看到物体。

不用示范，不用离线标定。所有量现场测：
  物体像素      YOLO-World -> CLIP -> SAM 质心；开始时测一次（物体不动，之后可能被臂挡住）
  夹爪像素      夹爪开/合一次，俯视画面里只有手指在动，变化像素的质心
  像素<->臂移动 臂沿 x、y 各点动 ±2cm 看夹爪像素怎么动，得到 2x2 矩阵 A
方向由 PlayJev 在 Snake 里选；每步之后重测夹爪像素；腕部相机看到物体就交接。

臂只在一个固定高度的水平面里动（固定朝向 R0，之前抓取成功时的朝向），不会往下。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import actions as A
from . import rec
from .calibrate_top import finger_pixel, median_frame
from .probe_playjev import ask
from .run_playjev import adaptive_board
from .snake_render import INSTRUCTIONS, OPTIONS, render

CALIB = Path(__file__).resolve().parent / "calibration"
START_FLANGE = np.array([0.333, -0.257, 0.33])   # 起点：泡沫板中部上方，手指离板面约 8cm
JOG_M = 0.02
MAX_STEP_M = 0.15                                # 每步最多 15cm（分成 12mm 小段走，仍在同一水平面）
A_CACHE = CALIB / "top_A.json"                   # 2x2 矩阵 + 起点处夹爪像素；俯视相机/起点/朝向不变就一直有效
GRIP_A, GRIP_B = 0.0, 0.6                        # 开合在这两档之间切换，不合死
ALIGN_PX = 30                                    # 俯视里夹爪离物体 < 30px（约 3cm）算粗对准完成；
                                                 # 只需让物体进腕部视野，余下交给按深度后退 + 兜底搜索
MIN_AREA, MAX_AREA = 300, 40000
START_CHECK_PX = 30                              # 起点夹爪实测和缓存差超过这个 -> 缓存作废
NEAR_PX = 120                                    # 变化块离预测夹爪位置超过这个就不算（夹爪本身约 150px 宽）


class Gripper2D:
    """在俯视画面里找手指：每调用一次，夹爪在 A/B 两档之间切换一次，取前后两帧的差。"""

    def __init__(self, env, top):
        self.env, self.top, self.state = env, top, GRIP_A

    def _hold(self, v, s):
        t_end = time.time() + s
        while time.time() < t_end:
            self.env.step(A.hold(v)); time.sleep(0.05)

    def locate(self, near=None, tries=2):
        """near：预测的夹爪像素。差分里只留离它 NEAR_PX 以内的变化块 —— 画面里有人走动时，
        人腿也在差分里，不筛会把质心拉偏几十到几百像素（点动 2cm 测出每米 13000px）。
        碰画面边缘的块单独丢掉；若留下的块本身碰边（手指被切掉一截），整次作废。"""
        import cv2

        for _ in range(tries):
            time.sleep(0.3)
            a = median_frame(self.top)
            self.state = GRIP_B if self.state == GRIP_A else GRIP_A
            self._hold(self.state, 1.2); time.sleep(0.2)
            b = median_frame(self.top)
            _, _, m = finger_pixel(a, b)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
            keep = np.zeros_like(m)
            for i in range(1, n):
                x, y, w, h = stats[i, :4]
                on_border = x <= 3 or y <= 3 or x + w >= m.shape[1] - 3 or y + h >= m.shape[0] - 3
                if near is not None and np.linalg.norm(cents[i] - near) > NEAR_PX:
                    continue
                if on_border:
                    if near is not None:
                        break                       # 夹爪自己碰边
                    continue
                keep[lab == i] = 1
            else:
                area = int(keep.sum())
                if MIN_AREA <= area <= MAX_AREA:
                    ys, xs = np.nonzero(keep)
                    return np.array([xs.mean(), ys.mean()])
        return None


def step_xy(env, b_xy, step_and_settle):
    """水平移动 b_xy（米），按 12mm 一段走。"""
    b = np.r_[b_xy, 0.0]
    n = int(np.ceil(np.linalg.norm(b) / A.COARSE_M - 1e-9))
    for _ in range(max(n, 1)):
        step_and_settle(env, b / max(n, 1), grip=0.0)


def goto_start(env, slow_goto):
    from atom_robot.ar5.kinematics import AR5Kinematics

    kin = AR5Kinematics.from_robot(env.arm)
    q_ref = np.asarray(json.loads((CALIB / "pregrasp_pose.json").read_text())["q"], float)
    T = np.eye(4); T[:3, :3] = kin.fk(q_ref)[:3, :3]; T[:3, 3] = START_FLANGE
    q, info = kin.ik(T, q_ref)
    if q is None:
        raise RuntimeError(f"起点逆解失败: {info['reason']}")
    reached, el = slow_goto(env, q)
    print(f"到起点（法兰 {START_FLANGE.tolist()}）: {reached} {el:.1f}s")


def measure_A(env, grip, step_and_settle, attempts=3, near=None):
    """A[:,k] = 臂沿 k 轴移动 1m 时夹爪像素的变化。正反各一次，两次不一致就重测（同测 J）。"""
    A2 = np.zeros((2, 2))
    p0 = grip.locate(near=near)
    if p0 is None:
        raise RuntimeError("俯视画面里认不出夹爪（开合无变化，或手指在画面外）")
    for k in range(2):
        for attempt in range(attempts):
            cols = []
            for sgn in (+1, -1):
                d = np.zeros(2); d[k] = sgn * JOG_M
                step_xy(env, d, step_and_settle)
                p = grip.locate(near=p0)
                step_xy(env, -d, step_and_settle)
                if p is not None:
                    cols.append((p - p0) / (sgn * JOG_M))
            txt = " / ".join(str(np.round(c).astype(int).tolist()) for c in cols)
            if len(cols) == 2 and np.linalg.norm(cols[0] - cols[1]) < 0.35 * np.linalg.norm(cols[0] + cols[1]) / 2:
                A2[:, k] = 0.5 * (cols[0] + cols[1])
                print(f"   点动 {'xy'[k]}: 每米 {txt} 像素")
                break
            print(f"   点动 {'xy'[k]} 第{attempt + 1}次正反不一致（{txt}），重测")
        else:
            raise RuntimeError(f"俯视点动 {'xy'[k]} {attempts} 次都测不出一致的结果")
    return A2, p0


def load_A():
    """读缓存。起点或朝向变了就不认（这两者决定 A）。"""
    if not A_CACHE.exists():
        return None
    d = json.loads(A_CACHE.read_text())
    q_ref = json.loads((CALIB / "pregrasp_pose.json").read_text())["q"]
    if not np.allclose(d["start_flange"], START_FLANGE) or not np.allclose(d["q_ref"], q_ref):
        return None
    return np.asarray(d["A"], float), np.asarray(d["p0"], float)


def save_A(A2, p0):
    A_CACHE.write_text(json.dumps(dict(
        A=A2.tolist(), p0=p0.tolist(), start_flange=START_FLANGE.tolist(),
        q_ref=json.loads((CALIB / "pregrasp_pose.json").read_text())["q"],
        date=time.strftime("%Y-%m-%d %H:%M")), indent=1))


def approach(env, tf, top, wrist, step_and_settle, slow_goto, obj_px, max_steps=10):
    """返回 "seen" = 腕部相机已看到物体；"aligned" = 俯视里已对准；False = 失败。"""
    rec.tag("to_start")
    goto_start(env, slow_goto)
    rec.tag("start_check")
    grip = Gripper2D(env, top)
    cached = load_A()
    if cached is not None:
        A2, p0 = cached
        # 先在起点认一次夹爪核对缓存：俯视相机被挪过时，缓存的起点像素会差很远，
        # 照用的话第一步就会朝错的方向走（最多 15cm）。只多一次开合（约 2s）。
        g = grip.locate(near=None)
        if g is None or np.linalg.norm(g - p0) > START_CHECK_PX:
            print(f"起点夹爪实测 {None if g is None else np.round(g).astype(int).tolist()}，缓存是 "
                  f"{np.round(p0).astype(int).tolist()} —— 俯视相机可能动过，重测 A")
            cached = None
        else:
            print(f"用缓存的俯视「像素 <-> 臂移动」（{A_CACHE.name}），起点夹爪实测 "
                  f"{np.round(g).astype(int).tolist()}（缓存 {np.round(p0).astype(int).tolist()}）")
    if cached is None:
        print("测俯视「像素 <-> 臂移动」:")
        A2, g = measure_A(env, grip, step_and_settle)
        save_A(A2, g)
    Ainv = np.linalg.inv(A2)
    print(f"   A 条件数 {np.linalg.cond(A2):.1f}")
    for step in range(max_steps):
        rec.tag(f"top{step:02d}")
        tf.last = None
        if tf.observe_median(wrist, allow_track=False, margin=20) is not None:
            print(f"[{step:2d}] 腕部相机看到物体，交接")
            return "seen"
        e = obj_px - g                                   # 夹爪还要在画面里移动多少像素
        if np.linalg.norm(e) < ALIGN_PX:
            # 俯视没有深度：夹爪离板面 ~10cm，画面里重合时夹爪实际在物体前方（靠俯视相机一侧），
            # 而腕部相机正朝那个方向看 —— 物体在它身后。交给 acquire 沿视线水平方向后退去找。
            print(f"[{step:2d}] 俯视里已对准（差 {np.round(e).astype(int).tolist()}px），交给腕部相机搜索")
            return "aligned"
        es = e.copy()
        es[np.abs(es) < 0.3 * np.max(np.abs(es))] = 0.0  # 只画占主导的分量（同腕部伺服）
        t = np.array([-es[1], -es[0]]) * 1e-3            # 棋盘上 = 画面上(-v)；1px 画成 1mm
        img, _, _ = render(np.zeros(2), t, adaptive_board(np.zeros(2), t))
        move = ask(img, INSTRUCTIONS, OPTIONS)["answers"]["move"]["choice"]
        axis, sgn = {"up": (1, -1), "down": (1, +1), "left": (0, -1), "right": (0, +1)}[move]
        dpx = np.zeros(2); dpx[axis] = sgn * abs(e[axis])
        b = Ainv @ dpx
        if np.linalg.norm(b) > MAX_STEP_M:
            b *= MAX_STEP_M / np.linalg.norm(b)
        ok = np.sign(sgn) == np.sign(e[axis])
        print(f"[{step:2d}] 夹爪 {np.round(g).astype(int).tolist()} -> 物体 {np.round(obj_px).astype(int).tolist()} "
              f"差 {np.round(e).astype(int).tolist()}px  PlayJev={move:5s} {'✓' if ok else '✗'}  "
              f"臂走 {np.round(b * 1000).astype(int).tolist()} mm  "
              f"法兰 {np.round(np.asarray(env.arm.get_ee_pose())[:3] * 1000).astype(int).tolist()}")
        step_xy(env, b, step_and_settle)
        pred = g + A2 @ b
        g2 = grip.locate(near=pred)
        if g2 is None:
            print(f"[{step:2d}] 俯视画面里认不出夹爪，停")
            return False
        # 只在「方向明显不对」时怀疑：沿预测方向实际走了预测的 r 倍。走多走少（A 在画面不同区域
        # 本来就不一样，实测左侧超 35px）交给下一步闭环修；之前按距离判，一超就重测，而重测时手指
        # 悬在物体上方，开合一挡一露，差分质心乱跳，测不出一致的 A。
        def ratio(gm):
            d = A2 @ b
            return float((gm - g) @ d / max(d @ d, 1e-9))
        r = ratio(g2)
        if not (0.3 <= r <= 3.0):
            g3 = grip.locate(near=pred)                  # 先怀疑这一次认错了
            if g3 is not None and 0.3 <= ratio(g3) <= 3.0:
                print(f"   夹爪误认成 {np.round(g2).astype(int).tolist()}，重认得 {np.round(g3).astype(int).tolist()}")
                g2, r = g3, ratio(g3)
        if not (0.3 <= r <= 3.0):
            # 两次都不对：俯视相机被碰过、或缓存对不上。作废缓存，回起点从头测最稳
            print(f"   夹爪实到 {np.round(g2).astype(int).tolist()}，按 A 应到 {np.round(pred).astype(int).tolist()}"
                  f"（只走了预测的 {r:.2f} 倍），回起点重测 A")
            goto_start(env, slow_goto)
            A2, g2 = measure_A(env, grip, step_and_settle)
            save_A(A2, g2)
            Ainv = np.linalg.inv(A2)
        g = g2
    print("步数用完，腕部相机仍没看到物体")
    return False


def object_pixel(tf, top):
    """俯视图里物体的像素（SAM 掩码质心）。返回 (像素, 描述)。"""
    img = median_frame(top)
    cands = tf._candidates(img, max_area_frac=0.25)
    if not cands:
        raise SystemExit(f"俯视相机里没找到「{tf.phrase}」（查询 {tf.queries}）")
    i, rank, m = tf._pick(img, cands, view="top")
    print(f"指令短语「{tf.phrase}」 查询 {tf.queries}")
    for c, rk, mm in sorted(zip(cands, rank, m), key=lambda t: (t[1], -t[2]))[:5]:
        print(f"   「{c.phrase}」det={c.conf:.2f} ({c.centre[0]:.0f},{c.centre[1]:.0f}) CLIP 名次={rk} 余量={mm:+.3f}")
    if i is None:
        raise SystemExit(f"CLIP 认为俯视画面里没有一个候选像目标（要求名次 <= {tf.TOP_K}）")
    mask = tf._segment(img, cands[i].xyxy)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return np.array([xs.mean(), ys.mean()]), cands[i]
    return np.asarray(cands[i].centre, float), cands[i]
