"""通用的「找到指令说的那个物体，并测出它在腕部相机系里的 3D 位置和尺寸」。

**没有任何物体专属的常数**（用户要求：换一个物体也要能做）。之前只对乒乓球成立的东西都换掉了：

  乒乓球半径 20mm         -> 物体自己的 SAM 掩码 + 深度点云，中心 = 可见表面中位点 + 沿视线补「最窄那一维的一半」
  手写同义词表            -> 从指令里抽物体短语，按规则扩展（颜色+object、中心名词）；干扰项用 COCO 80 类
  颜色直方图跟踪/取深度    -> MobileSAM 分割（白色物体放在白泡沫板上颜色也分不开）
  （成功判据的固定区间）   -> 见 run_composite：按测到的宽度算这个物体合爪后应有的读数

流水线：YOLO-World-x（候选）-> 说法一致性（候选说法必须来自这条指令）-> CLIP（排序 + 比背景描述更像）
       -> MobileSAM（框提示分割）-> 去畸变 + 映射到深度像素 -> 3D 点云。
检测认不出来时（常见：一半藏在手指后）用上一帧的框提示 SAM，3D 中心离上一帧 <3cm 才接受。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

CALIB = Path(__file__).resolve().parent / "calibration"
GRASP_BAND_M = 0.027     # 抓顶部这么高的一截 = 夹持中心往上到手指根部的竖直距离（calibrate_gripper 实测 27~28mm，
                         # 是下限）。运行时测完 J 会用 gripper.json 重算并写进 TargetFinder.grasp_band

COLOURS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "grey",
           "gray", "brown", "silver", "golden", "transparent", "clear"}
_VERBS = r"^(please\s+)?(pick\s+up|pick|grab|grasp|take|get|lift|hold|fetch)\s+"
_ARTICLES = r"^(the|a|an|that|this)\s+"


def object_phrase(instruction: str) -> str:
    """'Pick up the ping pong ball.' -> 'ping pong ball'。纯规则，不认识具体物体。"""
    s = instruction.strip().lower().rstrip(".!?")
    s = re.sub(_VERBS, "", s)
    s = re.sub(_ARTICLES, "", s)
    s = re.split(r"\s+(on|from|in|at|near|next to|to)\s+", s)[0]
    return s.strip()


def query_phrases(phrase: str) -> List[str]:
    """YOLO-World 对措辞非常敏感（blue cylinder 检不出、blue tape 0.68），所以一个目标多查几种说法。
    扩展规则是通用的：原短语、「颜色 + object」、中心名词（最后一个词）。"""
    words = phrase.split()
    out = [phrase]
    cols = [w for w in words if w in COLOURS]
    if cols:
        out.append(f"{cols[0]} object")
    if len(words) > 1 and words[-1] not in COLOURS:
        out.append(words[-1])
    return list(dict.fromkeys(out))


def coco_names() -> List[str]:
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    return list(YAML.load(check_yaml("coco.yaml"))["names"].values())


# ------------------------------------------------------------------- 相机几何

def ransac_plane(Q: np.ndarray, thr: float = 0.004, iters: int = 300, min_frac: float = 0.4,
                 up=None, max_tilt_deg: float = 10.0):
    """内点最多的平面 (点, 单位法向)；内点不到 min_frac 返回 None（没有占多数的平面）。
    给了 up（相机系里的基座竖直方向）就只接受法向和它夹角 ≤ max_tilt_deg 的平面，即水平面。"""
    rng = np.random.default_rng(0)
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best_k, best = 0, None
    for _ in range(iters):
        a, b, c3 = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c3 - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if up is not None and abs(n @ up) < cos_tilt:
            continue
        k = int((np.abs((Q - a) @ n) < thr).sum())
        if k > best_k:
            best_k, best = k, (a, n)
    if best is None or best_k < max(60, min_frac * len(Q)):
        return None
    a, n = best
    inl = Q[np.abs((Q - a) @ n) < thr]
    c = inl.mean(axis=0)
    return c, np.linalg.svd(inl - c, full_matrices=False)[2][2]


SUPPORT_BAND_M = 0.004   # 离支撑面这么近的点算支撑面本身（就近那一层取层上点、RANSAC 内点都用这个宽度）
JUMP_M = 0.05            # 伺服中目标中心一帧内最多跳这么远（每步臂最多走 12mm，留足余量）


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0


class CameraModel:
    """腕部 Orbbec：彩色像素 -> 去畸变 -> 深度像素 -> 3D（彩色相机系，米）。参数全部来自相机自报。"""

    def __init__(self, path: Optional[Path] = None):
        d = json.loads((path or CALIB / "orbbec_params.json").read_text())
        ci, di = d["color_intrinsic"], d["depth_intrinsic"]
        self.Kc = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1.0]])
        self.Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
        self.dist_c = np.array(d["color_distortion"]["opencv"], float)   # k1,k2,p1,p2,k3,k4,k5,k6
        self.R = np.array(d["depth_to_color"]["R"], float)               # p_color = R p_depth + t
        self.t = np.array(d["depth_to_color"]["t_mm"], float) / 1000.0
        self.scale = float(d["depth_scale_mm_per_unit"]) / 1000.0
        # 335：OBSENSOR 给的深度图已对齐到彩色（同像素），不能再按深度内参 + depth_to_color 投一遍
        # —— 那样每个像素查到的是错位的深度，9/29 外参标定残差 ~10mm、尺度偏 7% 就是这个。305 仍走原路。
        self.aligned = bool(d.get("obsensor_depth_aligned_to_color", False))
        if self.aligned:
            oi = d["obsensor_intrinsic"]
            self.Ka = np.array([[oi["fx"], 0, oi["cx"]], [0, oi["fy"], oi["cy"]], [0, 0, 1.0]])

    def project(self, P: np.ndarray) -> np.ndarray:
        """彩色相机系 3D 点 (N,3) -> 彩色像素 (N,2)，points 的逆（含同一套畸变模型）。"""
        import cv2

        P = np.atleast_2d(np.asarray(P, float))
        if self.aligned:
            K = self.Ka
            return np.c_[K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]]
        uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.Kc, self.dist_c)
        return uv.reshape(-1, 2)

    def points(self, depth_raw: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """一批彩色像素 -> 彩色相机系 3D 点 (N,3)。无效深度的点剔除。"""
        P = self.points_keep(depth_raw, us, vs)
        return P[np.isfinite(P[:, 0])]

    def points_keep(self, depth_raw: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """同 points，但保持输入顺序，无效点是 NaN —— 光流前后两帧的点要一一对应。"""
        import cv2

        if self.aligned:
            us, vs = np.asarray(us, float), np.asarray(vs, float)
            h, w = depth_raw.shape
            ui, vi = np.round(us).astype(int), np.round(vs).astype(int)
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            z = np.zeros(len(ui))
            z[ok] = depth_raw[vi[ok], ui[ok]].astype(float)
            ok &= (z > 0) & (z < 65535)
            z = z * self.scale
            K = self.Ka
            P = np.c_[(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))] * z[:, None]
            P[~ok] = np.nan
            return P

        pts = np.stack([us, vs], axis=1).astype(np.float64).reshape(-1, 1, 2)
        nrm = cv2.undistortPoints(pts, self.Kc, self.dist_c).reshape(-1, 2)      # 彩色归一化坐标
        rays_c = np.c_[nrm, np.ones(len(nrm))]
        rays_d = rays_c @ self.R                                                   # R^T 作用在行向量上
        pd = rays_d @ self.Kd.T
        ud, vd = pd[:, 0] / pd[:, 2], pd[:, 1] / pd[:, 2]
        h, w = depth_raw.shape
        ui, vi = np.round(ud).astype(int), np.round(vd).astype(int)
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
    centre: np.ndarray           # 估计的物体中心，彩色相机系（米）
    surface: np.ndarray          # 可见表面中位点
    extent: np.ndarray           # 相机 x/y 方向的尺寸（米，5~95 分位）
    source: str                  # "detect" / "track"
    clip: float = 0.0
    extra: dict = field(default_factory=dict)
    grasp: Optional[np.ndarray] = None         # 抓取点（相机系）：顶部那一截的中心，见 _geometry
    grasp_extent: Optional[np.ndarray] = None  # 抓取那一截在相机 x/y 方向的尺寸（夹爪按它张合）

    @property
    def pixel(self):
        x0, y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def interior_point(mask):
    """掩码里离边缘最远的像素 (u, v)。环形/凹形物体（套筒、U 形件）的质心可能落在掩码外面，这个点总在里面。"""
    import cv2

    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(u), float(v)


class TargetFinder:
    TOP_K = 3
    grasp_band = GRASP_BAND_M
    grasp_height = None          # 冻结的抓取高度（离支撑面，米）；None = 每次按 _grasp_height 现算
    up = None                    # 当前相机系里的基座竖直方向（单位向量）；有了就只拿水平面当支撑面
    # 点选模式（鼠标点的目标，没有可靠的名字）：不用 YOLO-World 按名字找，每帧在 track_px 处给 SAM 一个点提示，
    # 分割完把 track_px 挪到新掩码的内点；臂动了用 shift_track 按 J 预测物体在画面里挪到哪。None = 按名字检测
    track_px = None
    track_xyz = None             # track_px 那个内点的 3D（相机系），预测位移用
    def __init__(self, instruction: str, device: str = "cuda"):
        from ultralytics import SAM

        from .bench_select import ClipSelect
        from .perception import OpenVocab

        self.instruction = instruction
        self.phrase = object_phrase(instruction)
        self.queries = query_phrases(self.phrase)
        self.det = OpenVocab()
        self.clip = ClipSelect()
        self._set_texts()
        self.sam = SAM("mobile_sam.pt")
        self.cam = CameraModel()
        self.last: Optional[ObjectObs] = None

    def _set_texts(self):
        """CLIP 的对照文字：目标 + 背景描述 + 和目标不共享词的 COCO 类名。

        背景描述**按相机分**：「黑色机械手指」只对腕部相机有意义（手指就在画面下沿）。
        俯视画面里用它，会把压在黑色底座上的白色充电器拒掉 —— 检测框带进了黑色底座，
        CLIP 觉得像黑手指（目标描述排 2/85，只输给这一句）。
        """
        words = set(self.phrase.split())
        others = [f"a photo of a {n}" for n in coco_names() if not (set(n.split()) & words)]
        target = [f"a photo of a {self.phrase}"]
        wrist_neg = list(self.clip.NEGATIVES)
        top_neg = [t for t in wrist_neg if "finger" not in t]
        self.texts_by_view = {"wrist": target + wrist_neg + others, "top": target + top_neg + others}
        self.texts = self.texts_by_view["wrist"]

    def set_instruction(self, instruction: str):
        self.instruction = instruction
        self.phrase = object_phrase(instruction)
        self.queries = query_phrases(self.phrase)
        self._set_texts()
        self.last = None

    # ---------------------------------------------------------- 各环节
    def _candidates(self, img, max_area_frac=None):
        """检测器**只查指令派生出的说法**。

        YOLO-World 每个锚框只留得分最高的类别，类别之间是抢的：词表里加了 COCO 干扰项后，
        插排被标成 remote(0.82)，「white power strip」根本没机会出现；只查目标说法时它能拿到 0.46。
        所以检测器只管召回，辨别交给 CLIP（见 `_rank`）。
        """
        # conf 0.005：换了俯视视角后套筒只拿到 0.0073，0.02 时一个候选都没有。检测器只管召回。
        cands = self.det.detect(img, self.queries, conf=0.005, per_phrase=3)
        # 俯视（max_area_frac=0.25）：大于画面 1/4 的框不可能是能夹的东西（夹爪最多张 98mm）。
        # 不排除的话，框住整块泡沫板的大框因为里面有蓝套筒，CLIP 也排第 1（余量 0.004）——
        # 套筒本身没检出时就会去抓板子中心。腕部不设上限：贴近时物体本来就占大半个画面。
        if max_area_frac is None:
            return cands
        h, w = img.shape[:2]
        return [c for c in cands
                if (c.xyxy[2] - c.xyxy[0]) * (c.xyxy[3] - c.xyxy[1]) < max_area_frac * h * w]

    def _rank(self, img, cands, view: str = "wrist"):
        """每个候选返回 (目标描述在所有对照文字里的名次, 余量)。名次 <= TOP_K 才算是目标。

        对照文字 = 背景描述 + 和目标不共享词的 COCO 80 类名（共 ~85 句）。
        第一版要求排第 1：插在黑色底座上的白色充电器被「遥控器/鼠标」以零点零几的差距挤掉 ——
        它们本来就长得像。零样本分类通常看前几名；随机排进前 3 的概率只有 3/85。
        """
        S = self.clip.scores(img, cands, self.texts_by_view[view])
        rank = (S[:, 1:] > S[:, [0]]).sum(axis=1) + 1
        margin = S[:, 0] - S[:, 1:].max(axis=1)
        return rank, margin

    def _pick(self, img, cands, view: str = "wrist"):
        """选出目标：名次 <= TOP_K 的候选里，名次最高、余量最大的那个。没有就返回 None。"""
        rank, margin = self._rank(img, cands, view)
        ok = np.nonzero(rank <= self.TOP_K)[0]
        if len(ok) == 0:
            return None, rank, margin
        i = min(ok, key=lambda j: (rank[j], -margin[j]))
        return int(i), rank, margin

    def _segment(self, img, xyxy):
        r = self.sam(img, bboxes=[list(map(float, xyxy))], verbose=False)[0]
        if r.masks is None or len(r.masks.data) == 0:
            return None
        return r.masks.data[0].cpu().numpy().astype(bool)

    def segment_prompt(self, img, points=None, labels=None, box=None, multimask=False):
        """SAM 按点（正/负）或框分割，返回掩码列表（multimask=True 时是 [默认那个] + 三个粒度，去重）。

        一个点对应的「物体」有歧义（点瓶口上的球：球 / 球+瓶颈 / 球+整瓶），SAM 本来就会给三个粒度，
        ultralytics 的上层接口不收 multimask_output，这里临时包一层 predictor.prompt_inference 打开它。"""
        kw = {}
        if points:
            kw["points"], kw["labels"] = [list(map(float, p)) for p in points], list(labels or [1] * len(points))
            kw["points"] = [kw["points"]]; kw["labels"] = [kw["labels"]]          # 一个物体、多个点
        if box is not None:
            kw["bboxes"] = [list(map(float, box))]

        def run(mm):
            pred = getattr(self.sam, "predictor", None)
            orig = None
            if mm and pred is not None:
                orig = pred.prompt_inference
                pred.prompt_inference = lambda im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False: \
                    orig(im, bboxes, points, labels, masks, True)
            try:
                r = self.sam(img, verbose=False, **kw)[0]
            finally:
                if orig is not None:
                    pred.prompt_inference = orig
            return [] if r.masks is None else list(r.masks.data.cpu().numpy().astype(bool))

        out = run(False)                             # 先跑一次：predictor 第一次调用时才建
        if multimask:
            for m in run(True):
                if m.any() and all(_mask_iou(m, o) < 0.95 for o in out):
                    out.append(m)
        return [m for m in out if m.any()]

    def shift_track(self, d_cam):
        """臂动了：静态物体在相机系里移动 d_cam（= J·臂的位移），按它预测 track_px 的新位置。"""
        if self.track_px is None or self.track_xyz is None:
            return
        self.track_xyz = np.asarray(self.track_xyz, float) + np.asarray(d_cam, float)
        if self.track_xyz[2] > 0.02:
            self.track_px = tuple(self.cam.project(self.track_xyz)[0])

    def _point_mask(self, color, depth_raw):
        """点选模式下这一帧的目标掩码：在 track_px 给 SAM 一个点提示。"""
        if self.track_px is None:
            return None
        h, w = color.shape[:2]
        u, v = self.track_px
        if not (0 <= u < w and 0 <= v < h):
            return None
        ms = self.segment_prompt(color, points=[(u, v)])
        return ms[0] if ms else None

    def _support_plane(self, depth_raw, mask):
        """物体周围一圈的深度点拟合支撑面（桌面）。返回 (点, 朝向相机的单位法向) 或 None。"""
        import cv2

        m8 = mask.astype(np.uint8)
        near = cv2.dilate(m8, np.ones((9, 9), np.uint8))           # 贴边那一圈常有阴影/边缘噪声，跳过
        far = cv2.dilate(m8, np.ones((41, 41), np.uint8))
        ring = (far > 0) & (near == 0)
        ys, xs = np.nonzero(ring)
        if len(xs) < 80:
            return None
        if len(xs) > 3000:
            sel = np.random.default_rng(1).choice(len(xs), 3000, replace=False)
            xs, ys = xs[sel], ys[sel]
        Q = self.cam.points(depth_raw, xs, ys)
        if len(Q) < 80:
            return None
        # RANSAC：取内点最多的平面。靠近时这一圈里混着手指、边缘飞点（深度 65~144mm 都有），
        # 旧的「中位点 + SVD 再剔残差」起点就歪，套筒高度量成 29~33mm（实为 47），
        # 中心随臂位置跳 ±13mm，伺服在两点间来回 30 步。RANSAC 同一帧量出 47mm、帧间稳定。
        # 知道竖直方向（self.up，由 J 或外参给）时只找水平面：高物体斜着看，外面一圈有一大半落在
        # 远处的墙和地上，「占多数的平面」找不到（9/29 蓝盖水瓶，腕部离瓶 ~0.3m，支撑面 None → 高度 8mm）。
        # 桌面是水平的，这对任何物体都成立；限定水平后门槛放到 20%
        plane = ransac_plane(Q, iters=400, min_frac=0.2, up=self.up) if self.up is not None else ransac_plane(Q)
        if plane is None:
            return None
        c, n = plane
        if n @ (-c) < 0:                                            # 法向朝向相机
            n = -n
        return c, n

    def _local_support(self, depth_raw, mask, P):
        """知道竖直方向（self.up）时的支撑面：物体脚下**紧挨着**的那一层水平面，不是周围点最多的那一层。

        垫了盘子/书/盒子时，外面一圈里既有垫子表面又有桌面，「点最多」不一定是物体正下方那层 ——
        选到桌面，物体高度多算一个垫子，抓取点偏低、指尖戳垫子。做法：
          1. 物体最低可见点 h_low（沿竖直方向）
          2. 物体外面 0.8~3.5cm（按实际距离换成像素）一圈的点，只留不高于 h_low+1cm 的（支撑面不会比物体底还高）
          3. 沿竖直方向 3mm 一层数点，取**最高的够密的一层**（离物体底最近）= 物体放在上面的那一层
        法向就用 up（桌面/垫子表面是水平的）。返回 (层上一点, 法向) 或 None。
        """
        import cv2

        up = self.up
        hv = P @ up
        h_low = float(np.percentile(hv, 2))
        K = getattr(self.cam, "Ka", None) if getattr(self.cam, "aligned", False) else self.cam.Kc
        px_per_m = float(K[0, 0]) / max(float(np.median(P[:, 2])), 0.05)
        r_in = int(np.clip(0.008 * px_per_m, 5, 40))
        r_out = int(np.clip(0.035 * px_per_m, r_in + 6, 120))
        m8 = mask.astype(np.uint8)
        ring = (cv2.dilate(m8, np.ones((2 * r_out + 1, 2 * r_out + 1), np.uint8)) > 0) & \
               (cv2.dilate(m8, np.ones((2 * r_in + 1, 2 * r_in + 1), np.uint8)) == 0)
        ys, xs = np.nonzero(ring)
        if len(xs) > 4000:
            sel = np.random.default_rng(1).choice(len(xs), 4000, replace=False)
            xs, ys = xs[sel], ys[sel]
        Q = self.cam.points(depth_raw, xs, ys)
        if len(Q) < 30:
            return None
        hq = Q @ up
        Qb, hb = Q[hq <= h_low + 0.01], hq[hq <= h_low + 0.01]
        if len(hb) < 30:
            return None
        edges = np.arange(hb.min(), hb.max() + 0.003, 0.003)
        if len(edges) < 2:
            edges = np.array([hb.min() - 0.0015, hb.max() + 0.0015])
        hist, edges = np.histogram(hb, bins=edges)
        dense = np.nonzero(hist >= max(20, 0.05 * len(hb)))[0]
        if len(dense) == 0:
            return None
        k = dense.max()                                             # 最高的够密的一层
        level = 0.5 * (edges[k] + edges[k + 1])
        layer = Qb[np.abs(hb - level) < SUPPORT_BAND_M]
        n = up if up @ (-layer.mean(axis=0)) > 0 else -up           # 朝向相机
        self._support_info = dict(h_low=h_low, level=float(level), n_layers=len(dense),
                                  layer_pts=len(layer), ring_px=(r_in, r_out))
        return layer.mean(axis=0), n

    def _grasp_height(self, P, d, h):
        """抓取高度（离支撑面）。矮物体（< 2×grasp_band）抓半高。

        高物体抓顶部：从顶往下每 3mm 一层量宽度（相机 x ≈ 夹爪开合方向），一直往下到
        「比顶部宽 15%+3mm」或「这层点太少、深度不可靠」为止，指尖停在那一层，最深 grasp_band。
        平行夹爪要两边是竖直面：水瓶那次伸进 28mm，指尖落到瓶盖下的凸环/瓶肩斜面上，
        合爪时把瓶子挤走 2cm、合空（读 0.99）。瓶盖 26~28mm 宽到 157mm，凸环 38mm 在 154~151mm。
        近处从正上方只看得到顶面、侧面几乎没点，量不准 —— 所以要在远处量一次冻结（grasp_height）。
        """
        # 门槛 2.5 倍手指长度（~70mm）：原来 2 倍（56mm），竖放的鸡蛋 ~57mm 刚好越线，
        # 会去抓蛋尖（宽度一路在变、立刻就停）。半高时物体顶离掌心只差 1~2mm，手指长度 28mm 本来就是下限
        if h < 2.5 * self.grasp_band:
            return 0.5 * h
        step, lo = 0.003, h - self.grasp_band
        def width(z):
            sel = (d <= z) & (d > z - step)
            if sel.sum() < 10:
                return None
            return float(np.percentile(P[sel, 0], 98) - np.percentile(P[sel, 0], 2))
        # 停在「和上面各层中位宽度差出 ±3mm」的那一层之上 —— 宽度一变就是换到了物体的另一部分。
        # 只判变宽不够：透明瓶颈只剩零星深度点，量出来反而变窄（19/13mm），一路伸到下限，
        # 指尖落到瓶颈/凸环上，抬起滑脱（9/28 两次）。第一版「比顶部宽 15%+3mm」更松，38mm 凸环也混过去。
        z, ws, prof = h, [], []
        while z - step >= lo:
            w = width(z)
            prof.append((round(z * 1000), None if w is None else round(w * 1000)))
            if w is None or (len(ws) >= 2 and abs(w - float(np.median(ws))) > 0.003):
                break
            ws.append(w)
            z -= step
        self._last_profile = prof
        return float(min(max(z, lo), h - 0.008))

    def _geometry(self, depth_raw, mask):
        """物体中心 = 它在支撑面上的投影中心，沿法向抬高半个物体高度。

        高度是量出来的（物体点到支撑面的最大距离），不假设形状。
        之前用「可见面 + 最窄那一维的一半」，等于假设物体厚度 ≈ 宽度：平放的胶带卷
        （外径 ~7cm、厚 ~2cm）会被估到桌面以下，夹爪往泡沫里压。拟合不出支撑面时才退回旧算法。

        抓取点 = 物体**顶部那一截**：高度 z_g = max(h/2, h − GRASP_BAND_M)，水平位置和尺寸只用
        z_g 以上的点。夹爪从斜上方沿光轴接近，抓得越高，掌心越不会碰到物体顶部：
        水瓶（175mm）抓半高时，瓶盖/瓶颈正挡在接近路径上，相机和掌心先到瓶盖上方（9/28 录像）。
        矮物体（< 2×GRASP_BAND_M）不变，仍抓半高。

        返回 (可见面, 尺寸, 中心, 高度, 抓取点, 抓取处尺寸)。
        """
        ys, xs = np.nonzero(mask)
        if len(xs) < 30:
            return None
        if len(xs) > 4000:
            sel = np.random.default_rng(0).choice(len(xs), 4000, replace=False)
            xs, ys = xs[sel], ys[sel]
        P = self.cam.points(depth_raw, xs, ys)
        if len(P) < 30:
            return None
        zmed = np.median(P[:, 2])
        # 只切远的一侧：漏进掩码的背景只会在物体后面。近的一侧不设限 —— 高物体贴近看时瓶盖离相机最近，
        # 原来 ±10cm（更早 ±6cm）把瓶盖切掉，蓝盖水瓶 209mm 量成 111mm（9/29）
        P = P[(P[:, 2] - zmed) < 0.10]
        if len(P) < 30:
            return None
        surf = np.median(P, axis=0)
        ext = np.percentile(P[:, :2], 98, axis=0) - np.percentile(P[:, :2], 2, axis=0)
        self._support_info = None
        plane = self._local_support(depth_raw, mask, P) if self.up is not None else None
        if plane is None:
            plane = self._support_plane(depth_raw, mask)
        if plane is not None:
            c0, n = plane
            d = (P - c0) @ n                                        # 到支撑面的高度（朝相机为正）
            P = P[d > 0.003]                                        # 掩码里漏进来的桌面点不算物体
            if len(P) >= 30:
                d = (P - c0) @ n
                height = float(np.percentile(d, 97))
                # 物体自己的高度（自身最高 − 最低，沿竖直）：不依赖支撑面，是「顶部 − 支撑面」的下限。
                # 底部看不见时它偏小（白盖水瓶远处 113 vs 175mm），所以只用来核对：
                # 顶部 − 支撑面 比它还小 1cm 以上 = 支撑面取到了物体底部以上，不可信
                own = float(np.percentile(d, 99) - np.percentile(d, 1))
                self._height_own = own
                self._support_ok = height >= own - 0.01
                foot = P - np.outer(d, n)                           # 投影到支撑面
                centre = np.median(foot, axis=0) + 0.5 * height * n
                ext = np.percentile(P[:, :2], 98, axis=0) - np.percentile(P[:, :2], 2, axis=0)
                z_g = self.grasp_height if self.grasp_height is not None else self._grasp_height(P, d, height)
                self._last_z_g = z_g
                top = d >= z_g - 0.003
                if top.sum() >= 15:
                    grasp = np.median(foot[top], axis=0) + z_g * n
                    gext = np.percentile(P[top, :2], 98, axis=0) - np.percentile(P[top, :2], 2, axis=0)
                    self._grasp_pts = P[top]                        # 抓取高度以上的物体点（量夹爪方向的宽度用）
                else:
                    grasp, gext = centre, ext
                    self._grasp_pts = P
                return surf, ext, centre, height, grasp, gext
        self._grasp_pts = P
        ray = surf / np.linalg.norm(surf)
        centre = surf + 0.5 * float(np.min(ext)) * ray               # 退回：可见面 + 最窄维一半
        return surf, ext, centre, None, centre, ext

    # ---------------------------------------------------------- 对外
    def observe(self, color, depth_raw, allow_track: bool = True, margin: int = 6,
                view: str = "wrist", max_area_frac=None, continuity: bool = False) -> Optional[ObjectObs]:
        """continuity=True（伺服中用）：目标必须和上一次连续 —— 优先选和上一次框重叠最多的候选，
        中心一次跳 > JUMP_M 的不认。9/29 抓水瓶：前 7 步正常，第 7 步重新选中了 36cm 外另一个东西（CLIP 也放行），
        臂追着它走到工作空间边上撞 j5 限位。已经确认过的目标，连续性比 CLIP 分数更可靠。"""
        h, w = color.shape[:2]
        if self.track_px is not None:
            return self._observe_point(color, depth_raw, margin, continuity)
        cands = self._candidates(color, max_area_frac=max_area_frac)
        best = None
        if cands:
            inside = [c for c in cands if c.xyxy[0] >= margin and c.xyxy[1] >= margin
                      and c.xyxy[2] <= w - margin and c.xyxy[3] <= h - margin]
            if inside and continuity and self.last is not None:
                # 框的大小也要连续：只看重叠的话，一个把目标连同周围一片桌面都框进去的大框
                # 和目标框重叠也不低，会被当成同一个东西（9/29 鸡蛋 48×40mm 被换成 174×132mm，
                # 「进了两指之间才前推」永远不满足，伺服空转 45 步）
                a_last = _area(self.last.xyxy)
                ok = [c for c in inside if 0.5 <= _area(c.xyxy) / max(a_last, 1.0) <= 2.0]
                ious = [_iou(c.xyxy, self.last.xyxy) for c in ok]
                if ious and max(ious) >= 0.3:
                    k = int(np.argmax(ious))
                    best = (ok[k].xyxy, ok[k].phrase, 0.0, "detect")
            elif inside:
                i, rank, m = self._pick(color, inside, view=view)
                if i is not None:
                    best = (inside[i].xyxy, inside[i].phrase, float(m[i]), "detect")
        if best is None and allow_track and self.last is not None:
            x0, y0, x1, y1 = self.last.xyxy
            gx, gy = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
            best = (np.array([max(0, x0 - gx), max(0, y0 - gy), min(w, x1 + gx), min(h, y1 + gy)]),
                    self.last.phrase, 0.0, "track")
        if best is None:
            return None
        xyxy, phrase, sc, src = best
        mask = self._segment(color, xyxy)
        if mask is None:
            return None
        return self._finish(depth_raw, mask, phrase, sc, src, continuity)

    def _observe_point(self, color, depth_raw, margin, continuity):
        """点选模式的 observe：SAM 点提示代替「YOLO-World 按名字找 + CLIP 挑」，其余（几何、连续性门槛）不变。"""
        h, w = color.shape[:2]
        mask = self._point_mask(color, depth_raw)
        if mask is None:
            return None
        ys, xs = np.nonzero(mask)
        if margin and (xs.min() < margin or ys.min() < margin or xs.max() > w - margin or ys.max() > h - margin):
            return None                           # 和按名字找时一样：搜索阶段碰画面边的不算
        if continuity and self.last is not None and \
                not 0.5 <= mask.sum() / max(self.last.mask_px, 1) <= 2.0:
            return None                           # 面积一下变了一倍以上：点到了别的东西上
        obs = self._finish(depth_raw, mask, self.phrase, 0.0, "point", continuity)
        if obs is not None:
            u, v = interior_point(mask)
            P = self.cam.points_keep(depth_raw, np.array([u]), np.array([v]))[0]
            self.track_px = (u, v)
            self.track_xyz = P if np.all(np.isfinite(P)) else obs.surface.copy()
        return obs

    def _finish(self, depth_raw, mask, phrase, sc, src, continuity):
        """掩码 -> 几何 -> 连续性门槛 -> ObjectObs（按名字找和点选两种模式共用）。"""
        self._last_z_g = self._height_own = self._support_ok = None
        geo = self._geometry(depth_raw, mask)
        if geo is None:
            return None
        surf, ext, centre, height, grasp, gext = geo
        if src == "track" and np.linalg.norm(centre - self.last.centre) > 0.03:
            return None                       # 跟踪结果跳太远：多半分到了别的东西上
        if continuity and self.last is not None and np.linalg.norm(centre - self.last.centre) > JUMP_M:
            return None                       # 伺服每步最多 12mm，真目标不会一下跳这么远
        if continuity and self.last is not None and np.any(gext > 1.6 * np.maximum(self.last.grasp_extent, 0.005)):
            return None                       # 尺寸一下变大 1.6 倍以上：分到了别的东西上
        ys, xs = np.nonzero(mask)
        obs = ObjectObs(phrase=phrase, xyxy=np.array([xs.min(), ys.min(), xs.max(), ys.max()], float),
                        mask_px=int(mask.sum()), centre=centre, surface=surf, extent=ext, source=src, clip=sc,
                        extra={"height": height, "z_g": getattr(self, "_last_z_g", None) if height is not None else None,
                               "height_own": getattr(self, "_height_own", None) if height is not None else None,
                               "support_ok": getattr(self, "_support_ok", None) if height is not None else None,
                               "support": getattr(self, "_support_info", None)},
                        grasp=grasp, grasp_extent=gext)
        self.last = obs
        return obs

    def observe_median(self, cam, n: int = 5, allow_track: bool = True, margin: int = 6,
                       tries: int = 15, **kw) -> Optional[ObjectObs]:
        """连看几帧取中位，抗单帧抖动。"""
        import time

        obs: List[ObjectObs] = []
        kw_now = dict(kw)
        for _ in range(tries):
            f = cam.read()
            if f.ok and f.depth is not None:
                o = self.observe(f.color, f.depth, allow_track=allow_track, margin=margin, **kw_now)
                if o is not None:
                    obs.append(o)
                    # 这一次取中位的几帧必须是同一个东西：第一帧按 CLIP 挑，之后要和它连续。
                    # 9/30 腕部交接：每帧各挑各的，几帧是球、几帧是远处的白蛋壳，中位数混了两者，
                    # 对中又用最后一帧（蛋壳）的框，臂一路挪向蛋壳，球出了画面
                    kw_now = dict(kw, continuity=True)
                    if len(obs) >= n:
                        break
            time.sleep(0.03)
        if len(obs) < 3:
            return None
        o = obs[-1]
        o.centre = np.median([x.centre for x in obs], axis=0)
        o.surface = np.median([x.surface for x in obs], axis=0)
        o.extent = np.median([x.extent for x in obs], axis=0)
        o.grasp = np.median([x.grasp for x in obs], axis=0)
        o.grasp_extent = np.median([x.grasp_extent for x in obs], axis=0)
        hs = [x.extra.get("height") for x in obs if x.extra.get("height") is not None]
        o.extra["height"] = float(np.median(hs)) if hs else None
        zs = [x.extra.get("z_g") for x in obs if x.extra.get("z_g") is not None]
        o.extra["z_g"] = float(np.median(zs)) if zs else None
        o.extra["n"] = len(obs)
        o.extra["sources"] = [x.source for x in obs]
        return o
