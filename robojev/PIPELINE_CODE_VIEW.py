# 只读汇总：抓取流程实际执行的全部代码（从各源文件按原样抽出，去掉了注释和文档字符串）。
# 不要 import 这个文件；要改请改对应的源文件。

# ======================================================================
# robojev/run_composite.py
# ======================================================================
import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))
from . import actions as A
from .composite import choose_pregrasp, homography, in_coverage, pixel_to_xy
from .probe_playjev import ask
from .run_playjev import adaptive_board, settle
from .snake_render import INSTRUCTIONS, OPTIONS, render
GRASP_CENTER_CAM = np.array([0.0148, 0.0323, 0.0882])
GRIPPER_FULL_OPEN_M = 0.098
JOG_M = 0.01
ALIGN_TOL_M = 0.006
AXIS_DEADBAND_M = A.FINE_M / 2
FINGER_NEAR_M = 0.13

class KeepAlive:

    def __init__(self, env, period: float=0.05):
        import threading
        self.arm, self.period = (env.arm, period)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name='rt-keepalive', daemon=True)

    def _run(self):
        lock = getattr(self.arm, '_setpoint_lock', None)
        while lock is not None and (not self._stop.wait(self.period)):
            with lock:
                self.arm._goal_ticks = 0

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=1.0)

def connect_arm():
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True
    env = AR5Env(cfg)
    env.connect()
    settle(env, gripper=0.0)
    return env

def step_and_settle(env, delta_xyz, grip=0.0):
    d = np.asarray(delta_xyz, float)
    n = np.linalg.norm(d)
    if n > A.COARSE_M:
        d = d * (A.COARSE_M / n)
    env.step(np.r_[d, 0, 0, 0, grip])
    settle(env, gripper=grip)

def _frame(cam):
    for _ in range(20):
        f = cam.read()
        if f.ok and f.depth is not None:
            return (f.color.copy(), f.depth.copy())
        time.sleep(0.03)
    raise RuntimeError('腕部相机取不到帧')

def scene_shift(cm, a, b):
    import cv2
    ga, gb = (cv2.cvtColor(a[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(b[0], cv2.COLOR_BGR2GRAY))
    pts = cv2.goodFeaturesToTrack(ga, 600, 0.01, 7)
    if pts is None or len(pts) < 30:
        return (None, 0)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(21, 21), maxLevel=3)
    st = st.ravel().astype(bool)
    p0, p1 = (pts.reshape(-1, 2)[st], nxt.reshape(-1, 2)[st])
    P0 = cm.points_keep(a[1], p0[:, 0], p0[:, 1])
    P1 = cm.points_keep(b[1], p1[:, 0], p1[:, 1])
    ok = np.isfinite(P0[:, 0]) & np.isfinite(P1[:, 0]) & (P0[:, 2] > FINGER_NEAR_M) & (P1[:, 2] > FINGER_NEAR_M)
    if ok.sum() < 30:
        return (None, int(ok.sum()))
    return (np.median(P1[ok] - P0[ok], axis=0), int(ok.sum()))

def measure_jacobian(env, cam, cm):
    J = np.zeros((3, 3))
    for k in range(3):
        for attempt in range(3):
            cols, ns = ([], [])
            for sign in (+1, -1):
                d = np.zeros(3)
                d[k] = sign * JOG_M
                a = _frame(cam)
                step_and_settle(env, d)
                time.sleep(0.35)
                b = _frame(cam)
                step_and_settle(env, -d)
                time.sleep(0.35)
                sh, n = scene_shift(cm, a, b)
                ns.append(n)
                if sh is not None:
                    cols.append(sh / (sign * JOG_M))
            print(f"   点动 {'xyz'[k]} 第{attempt + 1}次（特征点 {ns}）: " + '  '.join((f"{'+-'[i]}:{np.round(c, 2).tolist()}|{np.linalg.norm(c):.2f}" for i, c in enumerate(cols))))
            if len(cols) == 2 and np.linalg.norm(cols[0] - cols[1]) < 0.25 and (0.75 < np.linalg.norm(0.5 * (cols[0] + cols[1])) < 1.25):
                J[:, k] = 0.5 * (cols[0] + cols[1])
                break
        else:
            raise RuntimeError(f"{'xyz'[k]} 轴点动 3 次都测不出一致的结果")
    U, S, Vt = np.linalg.svd(J)
    print(f'   J 奇异值 {np.round(S, 3).tolist()}（刚体平移理论全为 1）')
    return U @ Vt

def stage_locate(tf):
    from .camera import TopCamera
    with TopCamera() as cam:
        time.sleep(1.0)
        img = cam.read().color.copy()
    cands = tf._candidates(img)
    if not cands:
        raise SystemExit(f'俯视相机里没找到「{tf.phrase}」（查询 {tf.queries}）')
    i, rank, m = tf._pick(img, cands, view='top')
    print(f'指令短语「{tf.phrase}」 查询 {tf.queries}')
    for c, rk, mm in sorted(zip(cands, rank, m), key=lambda t: (t[1], -t[2]))[:5]:
        print(f'   「{c.phrase}」det={c.conf:.2f} ({c.centre[0]:.0f},{c.centre[1]:.0f}) CLIP 名次={rk} 余量={mm:+.3f}')
    if i is None:
        raise SystemExit(f'CLIP 认为俯视画面里没有一个候选像目标（要求名次 <= {tf.TOP_K}）')
    H, meta = homography()
    u, v = cands[i].centre
    if not in_coverage(meta, u, v, margin=40):
        raise SystemExit(f"目标像素 ({u:.0f},{v:.0f}) 在示范覆盖范围 {meta['px_range']} 之外，不抓")
    xy = pixel_to_xy(H, u, v)
    pre = choose_pregrasp(xy, np.asarray(meta['tip_offset_flange'], float))
    if pre is None:
        raise SystemExit('示范里挑不出预抓取姿态')
    print(f"选中 ({u:.0f},{v:.0f}) -> 粗定位 xy ({xy[0]:.3f},{xy[1]:.3f})；预抓取姿态来自 {pre['src']} 帧 {pre['frame']}")
    return pre

def acquire(env, tf, cam, J):
    moves = [(-0.012, 0.0)] * 14 + [(0.0, 0.012)] * 4 + [(0.0, -0.012)] * 8
    tf.last = None
    obs, k = (tf.observe_median(cam, allow_track=False, margin=20), 0)
    while obs is None and k < len(moves):
        step_and_settle(env, [moves[k][0], moves[k][1], 0.0])
        time.sleep(0.2)
        obs = tf.observe_median(cam, allow_track=False, margin=20)
        k += 1
    if obs is None:
        return None
    if k:
        print(f'   搜索 {k} 步后看到目标')
    for it in range(6):
        u, v = obs.pixel
        if abs(u - 320) < 90 and abs(v - 240) < 80:
            break
        vcam = np.array([obs.centre[0], obs.centre[1], 0.0])
        b = J.T @ -vcam
        step_and_settle(env, b)
        time.sleep(0.2)
        o2 = tf.observe_median(cam, allow_track=True)
        if o2 is None:
            break
        obs = o2
    return obs

def stage_grasp(env, tf, cam, J, max_steps=45):
    obs = tf.observe_median(cam, allow_track=True, margin=0)
    if obs is None:
        print('对准开始前看不到目标')
        return False
    for step in range(max_steps):
        obs = tf.observe_median(cam, allow_track=True, margin=0)
        if obs is None:
            print(f'[{step:2d}] 看不到目标，停')
            return False
        e = obs.centre - GRASP_CENTER_CAM
        if np.linalg.norm(e) < ALIGN_TOL_M:
            print(f'[{step:2d}] 目标中心离夹持中心 {np.round(e * 1000, 1).tolist()} mm，到位')
            break
        exy = e[:2].copy()
        exy[np.abs(exy) < AXIS_DEADBAND_M] = 0.0
        v = np.zeros(3)
        what = '画面内已对准'
        if np.any(exy != 0):
            g, t = (np.zeros(2), np.array([-exy[1], -exy[0]]))
            img, _, _ = render(g, t, adaptive_board(g, t))
            move = ask(img, INSTRUCTIONS, OPTIONS)['answers']['move']['choice']
            axis, sgn = {'up': (1, -1), 'down': (1, +1), 'left': (0, -1), 'right': (0, +1)}[move]
            err = abs(exy[axis])
            v[axis] = sgn * (A.COARSE_M if err > A.COARSE_M else min(A.FINE_M, err))
            what = f"PlayJev={move:5s} {('✓' if np.sign(v[axis]) == np.sign(exy[axis]) else '✗')}"
        if abs(e[2]) > AXIS_DEADBAND_M:
            v[2] = float(np.clip(e[2], -A.COARSE_M, A.COARSE_M))
        b = J.T @ -v
        print(f'[{step:2d}] 中心-夹持 {np.round(e * 1000).astype(int).tolist()} mm 尺寸 {np.round(obs.extent * 1000).astype(int).tolist()} mm [{obs.source}]  {what}  臂走 {np.round(b * 1000).astype(int).tolist()} mm')
        step_and_settle(env, b)
    else:
        print('步数用完仍未到位')
        return False
    width = float(obs.extent[0])
    expect = float(np.clip(1.0 - width / GRIPPER_FULL_OPEN_M, 0.0, 1.0))
    surf_before = obs.surface.copy()
    t_end = time.time() + 2.0
    while time.time() < t_end:
        env.step(A.hold(1.0))
        time.sleep(0.05)
    grip = float(env.hand.get_opening())
    lift = 5 * A.COARSE_M
    for _ in range(5):
        step_and_settle(env, [0, 0, A.COARSE_M], grip=1.0)
    time.sleep(0.3)
    o2 = tf.observe_median(cam, allow_track=True, margin=0)
    moved = None if o2 is None else float(np.linalg.norm(o2.surface - surf_before))
    grip_ok = grip < 0.97 and abs(grip - expect) < 0.2
    held = moved is not None and moved < 0.5 * lift
    print(f"宽度 {width * 1000:.0f}mm -> 应读 {expect:.2f}，实读 {grip:.2f} {('✓' if grip_ok else '✗')}；抬起 {lift * 1000:.0f}mm 后可见表面在相机系里移动 {('看不到' if moved is None else f'{moved * 1000:.0f}mm')}（<{lift * 500:.0f}mm 算跟着走） {('✓' if held else '✗')}  -> {('成功' if grip_ok and held else '失败')}")
    return grip_ok and held

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--instruction', required=True)
    ap.add_argument('--stage', choices=['locate', 'pregrasp', 'grasp'], default='locate')
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname).1s %(message)s')
    import warnings
    warnings.filterwarnings('ignore')
    from .objects import TargetFinder
    tf = TargetFinder(a.instruction)
    pre = stage_locate(tf)
    if a.stage == 'locate':
        return 0
    from .camera import OrbbecRGBD
    from .move_pregrasp import slow_goto
    cam = OrbbecRGBD()
    cam.connect()
    env = connect_arm()
    ka = KeepAlive(env).__enter__()
    try:
        reached, el = slow_goto(env, np.asarray(pre['q'], float))
        print(f'到预抓取姿态: {reached}  {el:.1f}s')
        print('测相机运动映射（静态场景光流）:')
        J = measure_jacobian(env, cam, tf.cam)
        obs = acquire(env, tf, cam, J)
        if obs is None:
            print('找不到目标')
            return 1
        h = obs.extra.get('height')
        print(f"目标中心(相机系) {np.round(obs.centre * 1000).astype(int).tolist()} mm  尺寸 {np.round(obs.extent * 1000).astype(int).tolist()} mm  高度 {('拟合不出支撑面' if h is None else f'{h * 1000:.0f} mm')}  像素 {np.round(obs.pixel).tolist()}")
        if a.stage == 'pregrasp':
            return 0
        ok = stage_grasp(env, tf, cam, J)
        print(f"env 统计: unreachable={env.stats['unreachable']} leashed={env.stats['leashed']}")
        return 0 if ok else 1
    finally:
        ka.__exit__(None, None, None)
        env.close()
        cam.disconnect()
if __name__ == '__main__':
    raise SystemExit(main())

# ======================================================================
# robojev/objects.py
# ======================================================================
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
CALIB = Path(__file__).resolve().parent / 'calibration'
COLOURS = {'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white', 'black', 'grey', 'gray', 'brown', 'silver', 'golden', 'transparent', 'clear'}
_VERBS = '^(please\\s+)?(pick\\s+up|pick|grab|grasp|take|get|lift|hold|fetch)\\s+'
_ARTICLES = '^(the|a|an|that|this)\\s+'

def object_phrase(instruction: str) -> str:
    s = instruction.strip().lower().rstrip('.!?')
    s = re.sub(_VERBS, '', s)
    s = re.sub(_ARTICLES, '', s)
    s = re.split('\\s+(on|from|in|at|near|next to|to)\\s+', s)[0]
    return s.strip()

def query_phrases(phrase: str) -> List[str]:
    words = phrase.split()
    out = [phrase]
    cols = [w for w in words if w in COLOURS]
    if cols:
        out.append(f'{cols[0]} object')
    if len(words) > 1 and words[-1] not in COLOURS:
        out.append(words[-1])
    return list(dict.fromkeys(out))

def coco_names() -> List[str]:
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    return list(YAML.load(check_yaml('coco.yaml'))['names'].values())

class CameraModel:

    def __init__(self, path: Optional[Path]=None):
        d = json.loads((path or CALIB / 'orbbec_params.json').read_text())
        ci, di = (d['color_intrinsic'], d['depth_intrinsic'])
        self.Kc = np.array([[ci['fx'], 0, ci['cx']], [0, ci['fy'], ci['cy']], [0, 0, 1.0]])
        self.Kd = np.array([[di['fx'], 0, di['cx']], [0, di['fy'], di['cy']], [0, 0, 1.0]])
        self.dist_c = np.array(d['color_distortion']['opencv'], float)
        self.R = np.array(d['depth_to_color']['R'], float)
        self.t = np.array(d['depth_to_color']['t_mm'], float) / 1000.0
        self.scale = float(d['depth_scale_mm_per_unit']) / 1000.0

    def points(self, depth_raw: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        P = self.points_keep(depth_raw, us, vs)
        return P[np.isfinite(P[:, 0])]

    def points_keep(self, depth_raw: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        import cv2
        pts = np.stack([us, vs], axis=1).astype(np.float64).reshape(-1, 1, 2)
        nrm = cv2.undistortPoints(pts, self.Kc, self.dist_c).reshape(-1, 2)
        rays_c = np.c_[nrm, np.ones(len(nrm))]
        rays_d = rays_c @ self.R
        pd = rays_d @ self.Kd.T
        ud, vd = (pd[:, 0] / pd[:, 2], pd[:, 1] / pd[:, 2])
        h, w = depth_raw.shape
        ui, vi = (np.round(ud).astype(int), np.round(vd).astype(int))
        ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        z = np.zeros(len(ui))
        z[ok] = depth_raw[vi[ok], ui[ok]].astype(float)
        ok &= (z > 0) & (z < 65535)
        z = z * self.scale
        P_d = np.c_[(ud - self.Kd[0, 2]) / self.Kd[0, 0], (vd - self.Kd[1, 2]) / self.Kd[1, 1], np.ones(len(ud))] * z[:, None]
        P_c = P_d @ self.R.T + self.t
        P_c[~ok] = np.nan
        return P_c

@dataclass
class ObjectObs:
    phrase: str
    xyxy: np.ndarray
    mask_px: int
    centre: np.ndarray
    surface: np.ndarray
    extent: np.ndarray
    source: str
    clip: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def pixel(self):
        x0, y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

class TargetFinder:
    TOP_K = 3

    def __init__(self, instruction: str, device: str='cuda'):
        from ultralytics import SAM
        from .bench_select import ClipSelect
        from .perception import OpenVocab
        self.instruction = instruction
        self.phrase = object_phrase(instruction)
        self.queries = query_phrases(self.phrase)
        self.det = OpenVocab()
        self.clip = ClipSelect()
        self._set_texts()
        self.sam = SAM('mobile_sam.pt')
        self.cam = CameraModel()
        self.last: Optional[ObjectObs] = None

    def _set_texts(self):
        words = set(self.phrase.split())
        others = [f'a photo of a {n}' for n in coco_names() if not set(n.split()) & words]
        target = [f'a photo of a {self.phrase}']
        wrist_neg = list(self.clip.NEGATIVES)
        top_neg = [t for t in wrist_neg if 'finger' not in t]
        self.texts_by_view = {'wrist': target + wrist_neg + others, 'top': target + top_neg + others}
        self.texts = self.texts_by_view['wrist']

    def set_instruction(self, instruction: str):
        self.instruction = instruction
        self.phrase = object_phrase(instruction)
        self.queries = query_phrases(self.phrase)
        self._set_texts()
        self.last = None

    def _candidates(self, img):
        return self.det.detect(img, self.queries, per_phrase=3)

    def _rank(self, img, cands, view: str='wrist'):
        S = self.clip.scores(img, cands, self.texts_by_view[view])
        rank = (S[:, 1:] > S[:, [0]]).sum(axis=1) + 1
        margin = S[:, 0] - S[:, 1:].max(axis=1)
        return (rank, margin)

    def _pick(self, img, cands, view: str='wrist'):
        rank, margin = self._rank(img, cands, view)
        ok = np.nonzero(rank <= self.TOP_K)[0]
        if len(ok) == 0:
            return (None, rank, margin)
        i = min(ok, key=lambda j: (rank[j], -margin[j]))
        return (int(i), rank, margin)

    def _segment(self, img, xyxy):
        r = self.sam(img, bboxes=[list(map(float, xyxy))], verbose=False)[0]
        if r.masks is None or len(r.masks.data) == 0:
            return None
        return r.masks.data[0].cpu().numpy().astype(bool)

    def _support_plane(self, depth_raw, mask):
        import cv2
        m8 = mask.astype(np.uint8)
        near = cv2.dilate(m8, np.ones((9, 9), np.uint8))
        far = cv2.dilate(m8, np.ones((41, 41), np.uint8))
        ring = (far > 0) & (near == 0)
        ys, xs = np.nonzero(ring)
        if len(xs) < 80:
            return None
        if len(xs) > 3000:
            sel = np.random.default_rng(1).choice(len(xs), 3000, replace=False)
            xs, ys = (xs[sel], ys[sel])
        Q = self.cam.points(depth_raw, xs, ys)
        if len(Q) < 80:
            return None
        for _ in range(3):
            c = np.median(Q, axis=0)
            _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
            n = Vt[2]
            r = np.abs((Q - c) @ n)
            keep = r < max(0.004, 2.5 * np.median(r))
            if keep.sum() < 60:
                return None
            Q = Q[keep]
        if np.median(r) > 0.004:
            return None
        if n @ -c < 0:
            n = -n
        return (c, n)

    def _geometry(self, depth_raw, mask):
        ys, xs = np.nonzero(mask)
        if len(xs) < 30:
            return None
        if len(xs) > 4000:
            sel = np.random.default_rng(0).choice(len(xs), 4000, replace=False)
            xs, ys = (xs[sel], ys[sel])
        P = self.cam.points(depth_raw, xs, ys)
        if len(P) < 30:
            return None
        zmed = np.median(P[:, 2])
        P = P[np.abs(P[:, 2] - zmed) < 0.06]
        if len(P) < 30:
            return None
        surf = np.median(P, axis=0)
        ext = np.percentile(P[:, :2], 98, axis=0) - np.percentile(P[:, :2], 2, axis=0)
        plane = self._support_plane(depth_raw, mask)
        if plane is not None:
            c0, n = plane
            d = (P - c0) @ n
            P = P[d > 0.003]
            if len(P) >= 30:
                d = (P - c0) @ n
                height = float(np.percentile(d, 97))
                foot = P - np.outer(d, n)
                centre = np.median(foot, axis=0) + 0.5 * height * n
                ext = np.percentile(P[:, :2], 98, axis=0) - np.percentile(P[:, :2], 2, axis=0)
                return (surf, ext, centre, height)
        ray = surf / np.linalg.norm(surf)
        centre = surf + 0.5 * float(np.min(ext)) * ray
        return (surf, ext, centre, None)

    def observe(self, color, depth_raw, allow_track: bool=True, margin: int=6) -> Optional[ObjectObs]:
        h, w = color.shape[:2]
        cands = self._candidates(color)
        best = None
        if cands:
            inside = [c for c in cands if c.xyxy[0] >= margin and c.xyxy[1] >= margin and (c.xyxy[2] <= w - margin) and (c.xyxy[3] <= h - margin)]
            if inside:
                i, rank, m = self._pick(color, inside, view='wrist')
                if i is not None:
                    best = (inside[i].xyxy, inside[i].phrase, float(m[i]), 'detect')
        if best is None and allow_track and (self.last is not None):
            x0, y0, x1, y1 = self.last.xyxy
            gx, gy = (0.2 * (x1 - x0), 0.2 * (y1 - y0))
            best = (np.array([max(0, x0 - gx), max(0, y0 - gy), min(w, x1 + gx), min(h, y1 + gy)]), self.last.phrase, 0.0, 'track')
        if best is None:
            return None
        xyxy, phrase, sc, src = best
        mask = self._segment(color, xyxy)
        if mask is None:
            return None
        geo = self._geometry(depth_raw, mask)
        if geo is None:
            return None
        surf, ext, centre, height = geo
        if src == 'track' and np.linalg.norm(centre - self.last.centre) > 0.03:
            return None
        ys, xs = np.nonzero(mask)
        obs = ObjectObs(phrase=phrase, xyxy=np.array([xs.min(), ys.min(), xs.max(), ys.max()], float), mask_px=int(mask.sum()), centre=centre, surface=surf, extent=ext, source=src, clip=sc, extra={'height': height})
        self.last = obs
        return obs

    def observe_median(self, cam, n: int=5, allow_track: bool=True, margin: int=6, tries: int=15) -> Optional[ObjectObs]:
        import time
        obs: List[ObjectObs] = []
        for _ in range(tries):
            f = cam.read()
            if f.ok and f.depth is not None:
                o = self.observe(f.color, f.depth, allow_track=allow_track, margin=margin)
                if o is not None:
                    obs.append(o)
                    if len(obs) >= n:
                        break
            time.sleep(0.03)
        if len(obs) < 3:
            return None
        o = obs[-1]
        o.centre = np.median([x.centre for x in obs], axis=0)
        o.surface = np.median([x.surface for x in obs], axis=0)
        o.extent = np.median([x.extent for x in obs], axis=0)
        o.extra['n'] = len(obs)
        o.extra['sources'] = [x.source for x in obs]
        return o

# ======================================================================
# robojev/composite.py
# ======================================================================
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
CALIB = Path(__file__).resolve().parent / 'calibration'
EPISODES = Path.home() / 'atom_episodes'

def homography():
    d = json.loads((CALIB / 'homography.json').read_text())
    return (np.array(d['H'], float), d)

def pixel_to_xy(H, u, v) -> np.ndarray:
    p = H @ np.array([u, v, 1.0])
    return p[:2] / p[2]

def in_coverage(meta, u, v, margin: float=0.0) -> bool:
    (u0, v0), (u1, v1) = meta.get('px_range', [[-1000000000.0, -1000000000.0], [1000000000.0, 1000000000.0]])
    return u0 - margin <= u <= u1 + margin and v0 - margin <= v <= v1 + margin

def choose_pregrasp(xy_target, t_offset, back_s: float=2.0, min_rise: float=0.04, max_xy_drift: float=0.04):
    import cv2
    good = json.loads((CALIB / 'demo_ball_episodes.json').read_text())
    best = None
    for r in good:
        d = np.load(EPISODES / r['f'])
        ee, st, fps, i = (d['ee_pose'].astype(float), d['state'].astype(float), float(d['fps']), r['i'])

        def tip(k):
            R, _ = cv2.Rodrigues(ee[k, 3:6])
            return ee[k, :3] + R @ t_offset
        j = i - int(back_s * fps)
        if j < 0:
            continue
        tg, tj = (tip(i), tip(j))
        if tj[2] - tg[2] < min_rise or np.linalg.norm(tj[:2] - tg[:2]) > max_xy_drift:
            continue
        dist = float(np.linalg.norm(tg[:2] - np.asarray(xy_target)))
        if best is None or dist < best['dist']:
            best = dict(q=st[j, :7].tolist(), src=r['f'], frame=j, tip_pre=tj.tolist(), tip_grasp_demo=tg.tolist(), dist=dist)
    return best

# ======================================================================
# robojev/perception.py
# ======================================================================
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np

@dataclass
class Candidate:
    phrase: str
    conf: float
    xyxy: np.ndarray
    hue: float = 0.0
    sat: float = 0.0
    val: float = 0.0
    area: float = 0.0
    extra: Dict = field(default_factory=dict)

    @property
    def centre(self):
        x0, y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

def pixel_stats(img_bgr: np.ndarray, xyxy) -> Dict[str, float]:
    import cv2
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    h, w = img_bgr.shape[:2]
    cx0, cx1 = (x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4)
    cy0, cy1 = (y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4)
    roi = img_bgr[max(0, cy0):min(h, cy1), max(0, cx0):min(w, cx1)]
    if roi.size == 0:
        return {'hue': 0.0, 'sat': 0.0, 'val': 0.0}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return {'hue': float(np.median(hsv[:, 0])), 'sat': float(np.median(hsv[:, 1])), 'val': float(np.median(hsv[:, 2]))}

class OpenVocab:

    def __init__(self, weights: str='yolov8x-worldv2.pt', device: str='cuda'):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self._phrases: List[str] = []

    def detect(self, img_bgr: np.ndarray, phrases: Sequence[str], conf: float=0.02, per_phrase: int=1) -> List[Candidate]:
        phrases = list(phrases)
        if phrases != self._phrases:
            self.model.set_classes(phrases)
            self._phrases = phrases
        r = self.model.predict(img_bgr, conf=conf, verbose=False, device=self.device)[0]
        by: Dict[str, List[Candidate]] = {}
        for b in r.boxes:
            ph = phrases[int(b.cls)]
            c = Candidate(phrase=ph, conf=float(b.conf), xyxy=b.xyxy[0].cpu().numpy().astype(float))
            by.setdefault(ph, []).append(c)
        out: List[Candidate] = []
        for ph, cs in by.items():
            cs.sort(key=lambda c: -c.conf)
            out.extend(cs[:per_phrase])
        for c in out:
            st = pixel_stats(img_bgr, c.xyxy)
            c.hue, c.sat, c.val = (st['hue'], st['sat'], st['val'])
            x0, y0, x1, y1 = c.xyxy
            c.area = float((x1 - x0) * (y1 - y0))
        return merge_duplicates(out)

def merge_duplicates(cands: List[Candidate], iou_thr: float=0.5) -> List[Candidate]:

    def iou(a, b):
        x0, y0 = (max(a[0], b[0]), max(a[1], b[1]))
        x1, y1 = (min(a[2], b[2]), min(a[3], b[3]))
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    cands = sorted(cands, key=lambda c: -c.conf)
    kept: List[Candidate] = []
    for c in cands:
        for k in kept:
            if iou(c.xyxy, k.xyxy) > iou_thr:
                k.extra.setdefault('also', []).append((c.phrase, round(c.conf, 3)))
                break
        else:
            kept.append(c)
    return kept

# ======================================================================
# robojev/bench_select.py
# ======================================================================
import argparse
import time
import warnings
import numpy as np

class ClipSelect:

    def __init__(self):
        import clip
        import torch
        self.torch = torch
        import os
        weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'clip')
        self.model, self.pre = clip.load('ViT-B/32', device='cuda', download_root=weights)  # 用项目里的权重，不去 ~/.cache/clip
        self.clip = clip
    NEGATIVES = ['a photo of the empty table surface', 'a photo of a black robot gripper finger', 'a photo of a white wall', 'a photo of a cable']

    def scores(self, img, cands, texts):
        from PIL import Image
        crops = []
        for c in cands:
            x0, y0, x1, y1 = [int(v) for v in c.xyxy]
            pad = 6
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad, ::-1]
            crops.append(self.pre(Image.fromarray(np.ascontiguousarray(crop))))
        with self.torch.no_grad():
            ie = self.model.encode_image(self.torch.stack(crops).cuda())
            te = self.model.encode_text(self.clip.tokenize(list(texts)).cuda())
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            return (ie @ te.T).float().cpu().numpy()

    def verified(self, img, cand, instruction) -> bool:
        s = self.scores(img, [cand], [f'a photo of the object in: {instruction}'] + self.NEGATIVES)[0]
        return bool(s[0] > s[1:].max())

    def __call__(self, img, cands, instruction):
        from PIL import Image
        crops = []
        for c in cands:
            x0, y0, x1, y1 = [int(v) for v in c.xyxy]
            pad = 6
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad, ::-1]
            crops.append(self.pre(Image.fromarray(np.ascontiguousarray(crop))))
        with self.torch.no_grad():
            ie = self.model.encode_image(self.torch.stack(crops).cuda())
            te = self.model.encode_text(self.clip.tokenize([f'a photo of the object in: {instruction}']).cuda())
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            s = (ie @ te.T).squeeze(1).float().cpu().numpy()
        return (int(np.argmax(s)), s)

# ======================================================================
# robojev/camera.py
# ======================================================================
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
OBSENSOR_INDEX = 0
ORBBEC_VID = '2bc5'

def find_orbbec_port() -> str | None:
    from pathlib import Path
    for dev in Path('/sys/bus/usb/devices').iterdir():
        vid = dev / 'idVendor'
        if ':' not in dev.name and vid.exists() and (vid.read_text().strip() == ORBBEC_VID):
            return dev.name
    return None
UGREEN_NODE = '/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:10.4:1.0-video-index0'
DEPTH_INVALID = (0, 65535)

def _disable_dynamic_framerate(node: str) -> None:
    import subprocess
    try:
        subprocess.run(['v4l2-ctl', '-d', node, '--set-ctrl', 'exposure_dynamic_framerate=0'], capture_output=True, timeout=3)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f'[Cam] 关不掉 exposure_dynamic_framerate: {exc}')

def reset_orbbec(wait_s: float=8.0) -> bool:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    port = find_orbbec_port()
    if port is None:
        logger.warning('[RGBD] USB 上找不到 Orbbec（厂商 2bc5），跳过复位')
        return False
    try:
        from atom_robot.ar5.reset_camera_usb import reset
        reset(port)
    except (ImportError, OSError, SystemExit) as exc:
        logger.warning(f'[RGBD] USB 复位没发出去：{exc}')
        return False
    time.sleep(wait_s)
    return True

@dataclass
class RGBDFrame:
    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    stamp: float = 0.0

    @property
    def ok(self) -> bool:
        return self.color is not None

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def depth_at(self, u: int, v: int, patch: int=5) -> Optional[float]:
        if self.depth is None:
            return None
        h, w = self.depth.shape
        r = patch // 2
        u0, u1 = (max(0, u - r), min(w, u + r + 1))
        v0, v1 = (max(0, v - r), min(h, v + r + 1))
        if u0 >= u1 or v0 >= v1:
            return None
        win = self.depth[v0:v1, u0:u1]
        mask = np.ones(win.shape, dtype=bool)
        for bad in DEPTH_INVALID:
            mask &= win != bad
        if not mask.any():
            return None
        return float(np.median(win[mask])) / 1000.0

class TopCamera:

    def __init__(self, node: str=UGREEN_NODE, width: int=640, height: int=480, fps: int=30):
        self.node, self.width, self.height, self.fps_target = (node, width, height, fps)
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._frame = RGBDFrame()
        self._count = 0
        self._t0: Optional[float] = None
        self.errors = 0

    def connect(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.node)
        if not cap.isOpened():
            raise IOError(f'打不开俯视相机 {self.node}\n  先查有没有别的进程占着：fuser /dev/video2\n  （采数据的 collect 会占这个节点）')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        _disable_dynamic_framerate(self.node)
        self._cap = cap
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name='robojev-top', daemon=True)
        self._thread.start()
        deadline = time.perf_counter() + 8.0
        while True:
            with self._lock:
                if self._frame.ok:
                    break
            if time.perf_counter() > deadline:
                self.disconnect()
                raise IOError(f'俯视相机打开了但 8 秒没出帧（{self.errors} 次失败读）')
            time.sleep(0.02)
        logger.info(f'[Top] 俯视相机就绪 {self.fps:.1f} Hz')

    def _run(self) -> None:
        while not self._halt.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.errors += 1
                self._halt.wait(0.02)
                continue
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            with self._lock:
                self._frame = RGBDFrame(color=frame, depth=None, stamp=now)
                self._count += 1

    def read(self) -> RGBDFrame:
        with self._lock:
            return self._frame

    @property
    def fps(self) -> float:
        if self._t0 is None or self._count < 2:
            return 0.0
        return (self._count - 1) / max(time.perf_counter() - self._t0, 1e-09)

    def disconnect(self) -> None:
        self._halt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

class OrbbecRGBD:

    def __init__(self, index: int=OBSENSOR_INDEX, auto_reset: bool=True):
        self.index = index
        self.auto_reset = auto_reset
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._frame = RGBDFrame()
        self._count = 0
        self._t0: Optional[float] = None
        self.errors = 0

    def connect(self) -> None:
        import cv2
        if self.auto_reset:
            logger.info('[RGBD] 复位 Orbbec 并等待重新枚举 ...')
            reset_orbbec()
        cap = cv2.VideoCapture(self.index, cv2.CAP_OBSENSOR)
        if not cap.isOpened():
            raise IOError(f'OBSENSOR 打不开设备 {self.index}。\n  先确认没有别的进程占着 Orbbec：fuser /dev/video0\n  再手动复位：python -m atom_robot.ar5.reset_camera_usb && sleep 8')
        self._cap = cap
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name='robojev-rgbd', daemon=True)
        self._thread.start()
        deadline = time.perf_counter() + 10.0
        while True:
            with self._lock:
                if self._frame.ok:
                    break
            if time.perf_counter() > deadline:
                self.disconnect()
                raise IOError(f'Orbbec 打开了但 10 秒内没出成对的帧（{self.errors} 次失败）。\n  多半是被 v4l2 路污染了，手动复位后重试：python -m atom_robot.ar5.reset_camera_usb && sleep 8')
            time.sleep(0.02)
        logger.info(f'[RGBD] Orbbec 就绪，{self.fps:.1f} Hz')

    def _run(self) -> None:
        import cv2
        while not self._halt.is_set():
            if not self._cap.grab():
                self.errors += 1
                self._halt.wait(0.02)
                continue
            ok_c, color = self._cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
            ok_d, depth = self._cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
            if not (ok_c and ok_d) or color is None or depth is None:
                self.errors += 1
                continue
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            with self._lock:
                self._frame = RGBDFrame(color=color, depth=depth, stamp=now)
                self._count += 1

    def read(self) -> RGBDFrame:
        with self._lock:
            f = self._frame
            return f

    @property
    def fps(self) -> float:
        if self._t0 is None or self._count < 2:
            return 0.0
        return (self._count - 1) / max(time.perf_counter() - self._t0, 1e-09)

    def disconnect(self) -> None:
        self._halt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

# ======================================================================
# robojev/snake_render.py
# ======================================================================
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
W, H = (448, 429)
BG = (168, 0, 0)
HEAD = (255, 255, 255)
FOOD = (0, 1, 253)
CELL = 17
PITCH = 19
COLS = W // PITCH
ROWS = H // PITCH
INSTRUCTIONS = 'Which move should the player make next?'
OPTIONS: Dict[str, str] = {'up': 'turn the snake to move up', 'down': 'turn the snake to move down', 'left': 'turn the snake to move left', 'right': 'turn the snake to move right'}
SNAKE_TO_BASE: Dict[str, Tuple[float, float]] = {'up': (+1.0, 0.0), 'down': (-1.0, 0.0), 'left': (0.0, +1.0), 'right': (0.0, -1.0)}
SNAKE_TO_ACTION: Dict[str, str] = {'up': 'forward', 'down': 'backward', 'left': 'left', 'right': 'right'}

@dataclass
class Board:
    x_min: float = 0.2
    x_max: float = 0.66
    y_min: float = -0.516
    y_max: float = 0.43

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        u = (self.y_max - y) / (self.y_max - self.y_min)
        v = (self.x_max - x) / (self.x_max - self.x_min)
        col = int(np.clip(round(u * (COLS - 1)), 0, COLS - 1))
        row = int(np.clip(round(v * (ROWS - 1)), 0, ROWS - 1))
        return (col, row)

    @property
    def metres_per_cell(self) -> Tuple[float, float]:
        return ((self.x_max - self.x_min) / (ROWS - 1), (self.y_max - self.y_min) / (COLS - 1))

def draw(head: Tuple[int, int], food: Tuple[int, int]) -> np.ndarray:
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = BG
    for (c, r), colour in ((food, FOOD), (head, HEAD)):
        x0, y0 = (2 + c * PITCH, 2 + r * PITCH)
        img[y0:y0 + CELL, x0:x0 + CELL] = colour
    return img

def render(gripper_xy, target_xy, board: Board=Board()) -> Tuple[np.ndarray, tuple, tuple]:
    head = board.to_cell(*gripper_xy)
    food = board.to_cell(*target_xy)
    return (draw(head, food), head, food)

# ======================================================================
# robojev/probe_playjev.py
# ======================================================================
import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from typing import Dict, List
import numpy as np
SERVER = 'http://127.0.0.1:18732/v1/systemone'
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def ask(img_bgr: np.ndarray, instructions: str, moves: Dict[str, str]=None, timeout: float=60.0) -> dict:
    import cv2
    moves = moves or MOVES
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError('JPEG 编码失败')
    b64 = base64.b64encode(buf.tobytes()).decode()
    body = {'state': {'frames': ['data:image/jpeg;base64,' + b64]}, 'questions': {'move': {'type': 'choice', 'instructions': instructions, 'criteria': dict(moves)}}}
    req = urllib.request.Request(SERVER, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        return json.load(_OPENER.open(req, timeout=timeout))
    except urllib.error.URLError as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raise
        raise ConnectionError(f'连不上 PlayJev 服务 {SERVER}：{exc.reason}。先起服务：cd 算法/Jev/RobotJev/PlayJev && python -m playjev.serve --ckpt weights/PlayJev-0.8B --port 18732') from exc

# ======================================================================
# robojev/run_playjev.py
# ======================================================================
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
from . import actions as A
from .probe_playjev import ask
from .snake_render import INSTRUCTIONS, OPTIONS, SNAKE_TO_ACTION, Board, render
MIN_HALF_M = 0.03

def settle(env, gripper: float=0.0, timeout_s: float=2.0, tol_m: float=0.001) -> bool:
    if not getattr(env.arm, '_rt_active', False):
        return True
    from . import actions as _A
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        time.sleep(0.05)
        env.step(_A.hold(gripper))
        tgt = env._target_pose
        cur = np.asarray(env.arm.get_ee_pose(), dtype=np.float64)
        v = getattr(env.arm, '_v_cmd', None)
        still = v is None or float(np.max(np.abs(v))) < 0.001
        if tgt is not None and np.linalg.norm(cur[:3] - tgt[:3]) < tol_m and still:
            return True
    logger.warning('settle: 2s 内没停稳')
    return False

def adaptive_board(g_xy, t_xy) -> Board:
    c = (np.asarray(g_xy) + np.asarray(t_xy)) / 2.0
    half = max(float(np.max(np.abs(np.asarray(g_xy) - np.asarray(t_xy)))) * 0.75, MIN_HALF_M)
    return Board(x_min=c[0] - half, x_max=c[0] + half, y_min=c[1] - half, y_max=c[1] + half)

# ======================================================================
# robojev/move_pregrasp.py
# ======================================================================
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
STEP_RAD = 0.01

def slow_goto(env, q_goal, step_rad: float=STEP_RAD, timeout_s: float=60.0):
    q_goal = np.asarray(q_goal, float)
    if hasattr(env.arm, 'clear_latch'):
        env.arm.clear_latch()
    t_start = time.time()
    while time.time() - t_start < timeout_s:
        cur = np.asarray(env.arm.get_joint_positions(), float)
        err = q_goal - cur
        if np.max(np.abs(err)) < np.deg2rad(0.5):
            break
        scale = min(1.0, step_rad / max(np.max(np.abs(err)), 1e-09))
        env.arm.servo_to_joints(cur + err * scale)
        if env.arm.has_motion_error():
            raise RuntimeError('控制器报运动错误，已停下')
        time.sleep(0.05)
    reached = env.goto_joints(q_goal, timeout_s=10.0)
    return (reached, time.time() - t_start)

# ======================================================================
# robojev/actions.py
# ======================================================================
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
COARSE_M = 0.012
FINE_M = 0.005
AXES = [(0, +1, 'forward', 'away from the robot base, further out over the bench'), (0, -1, 'backward', 'back toward the robot base'), (1, +1, 'left', "to the robot's left"), (1, -1, 'right', "to the robot's right"), (2, +1, 'up', 'up, away from the bench surface'), (2, -1, 'down', 'down, toward the bench surface')]
SCALES = [('big', COARSE_M), ('small', FINE_M)]

@dataclass(frozen=True)
class Action:
    name: str
    axis: int
    sign: int
    scale: str
    metres: float
    blurb: str

    def to_delta(self, gripper: float) -> np.ndarray:
        a = np.zeros(7, dtype=np.float64)
        a[self.axis] = self.sign * self.metres
        a[6] = float(np.clip(gripper, 0.0, 1.0))
        return a

def _build() -> List[Action]:
    out: List[Action] = []
    for axis, sign, word, where in AXES:
        for scale, metres in SCALES:
            out.append(Action(name=f'{word}_{scale}', axis=axis, sign=sign, scale=scale, metres=metres, blurb=f'move the gripper {word} ({where}) by {metres * 1000:.0f} mm'))
    return out
ACTIONS: List[Action] = _build()
BY_NAME: Dict[str, Action] = {a.name: a for a in ACTIONS}

def lookup(direction: str, scale: str) -> Action:
    return BY_NAME[f'{direction}_{scale}']

def hold(gripper: float) -> np.ndarray:
    a = np.zeros(7, dtype=np.float64)
    a[6] = float(np.clip(gripper, 0.0, 1.0))
    return a

# ======================================================================
# robojev/place_and_home.py
# ======================================================================
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))
from . import actions as A
from .move_pregrasp import slow_goto
from .run_playjev import GripperXYZ, settle
CENTRE = Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101" / 'atom_robot' / 'ar5' / 'joint_calibration.json'

def hold_gripper(env, value: float, seconds: float) -> None:
    t_end = time.time() + seconds
    while time.time() < t_end:
        env.step(A.hold(value))
        time.sleep(0.05)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--place-z', type=float, default=0.18, help='放下时的指尖高度（米）')
    ap.add_argument('--no-place', action='store_true', help='不下降，直接张爪回 0 位')
    ap.add_argument('--down', type=float, default=None, help='相对当前位置竖直下降多少（米）再张爪。抬了多少就降多少，不需要知道桌面/指尖高度（指尖偏移示范定不住，用绝对高度 --place-z 不可靠）')
    ap.add_argument('--lift', type=float, default=0.07, help='张爪后抬起多少（米）')
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname).1s %(message)s')
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    centre = np.asarray(json.loads(CENTRE.read_text())['centre_joints'], float)
    cfg = AR5Config()
    cfg.mock = False
    cfg.cameras.sources = {}
    cfg.arm.collision_detection_enabled = True
    env = AR5Env(cfg)
    env.connect()
    try:
        grip0 = float(env.hand.get_opening())
        settle(env, gripper=grip0)
        tip = GripperXYZ(env, mock=False)
        g = tip()
        print(f'当前指尖 {np.round(g, 3).tolist()}   夹爪 {grip0:.2f}（1=合拢）')
        if a.down is not None:
            steps = int(round(a.down / A.COARSE_M))
            for _ in range(steps):
                env.step(np.array([0, 0, -A.COARSE_M, 0, 0, 0, grip0]))
                settle(env, gripper=grip0)
            print(f'相对下降 {steps * A.COARSE_M * 100:.1f}cm')
        elif not a.no_place:
            n = 0
            while g[2] > a.place_z + 0.002 and n < 30:
                dz = max(-A.COARSE_M, a.place_z - g[2])
                env.step(np.array([0, 0, dz, 0, 0, 0, grip0]))
                settle(env, gripper=grip0)
                g = tip()
                n += 1
            print(f'放下到指尖 z={g[2]:.3f}（{n} 步）')
        hold_gripper(env, 0.0, 1.5)
        print(f'张爪，夹爪 {env.hand.get_opening():.2f}')
        steps = int(round(a.lift / A.COARSE_M))
        for _ in range(steps):
            env.step(A.lookup('up', 'big').to_delta(gripper=0.0))
            settle(env, gripper=0.0)
        print(f'抬起 {steps * A.COARSE_M * 100:.0f}cm，指尖 {np.round(tip(), 3).tolist()}')
        reached, el = slow_goto(env, centre)
        q = np.asarray(env.arm.get_joint_positions(), float)
        print(f'回 0 位: 到位={reached} 用时 {el:.1f}s  关节(度) {np.round(np.rad2deg(q), 2).tolist()}  和 0 位最大差 {np.rad2deg(np.max(np.abs(q - centre))):.2f}°')
        print(f"运动错误: {env.arm.has_motion_error()}  env 统计 unreachable={env.stats['unreachable']}")
    finally:
        env.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

