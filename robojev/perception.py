"""感知层：CV 模型 + 深度相机给出的数，不掺人为估计。

- `OpenVocab`：YOLO-World 开放词汇检测。**一个目标用几种说法同时查**，每种说法取它最好的那个框。
  实测对措辞非常敏感（「blue cylinder」检不出、「blue tape」0.68），置信度也普遍很低
  （对的框也只有 0.06~0.25），所以不设统一阈值，把候选都交给下游去选。
- `pixel_stats`：框里的主色和大小，用来给候选做便宜的校验，也用来写成话给 Laya。
- `WristDepth`：腕部 Orbbec 的像素 + 深度 → 相机坐标系 3D 点。
  **内参和深度单位都是相机自己报的**（calibration/orbbec_params.json，read_orbbec_params.py 读出来的）：
  深度单位 0.1 mm/单位。反投影用深度内参（无畸变）；彩色镜头是广角、畸变很大，不在彩色图上直接算几何。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

CALIB = Path(__file__).resolve().parent / "calibration"


@dataclass
class Candidate:
    phrase: str                 # 命中的那种说法
    conf: float
    xyxy: np.ndarray
    hue: float = 0.0            # 框中心区域的中位色相（OpenCV 0..179）
    sat: float = 0.0
    val: float = 0.0
    area: float = 0.0
    extra: Dict = field(default_factory=dict)

    @property
    def centre(self):
        x0, y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def colour_name(h: float, s: float, v: float) -> str:
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 170 else "grey"
    for lo, hi, name in ((0, 8, "red"), (8, 20, "orange"), (20, 35, "yellow"), (35, 85, "green"),
                         (85, 100, "cyan"), (100, 130, "blue"), (130, 160, "purple"), (160, 180, "red")):
        if lo <= h < hi:
            return name
    return "unknown"


def pixel_stats(img_bgr: np.ndarray, xyxy) -> Dict[str, float]:
    import cv2

    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    h, w = img_bgr.shape[:2]
    # 只看框中间一半，躲开背景
    cx0, cx1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
    cy0, cy1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
    roi = img_bgr[max(0, cy0):min(h, cy1), max(0, cx0):min(w, cx1)]
    if roi.size == 0:
        return {"hue": 0.0, "sat": 0.0, "val": 0.0}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return {"hue": float(np.median(hsv[:, 0])), "sat": float(np.median(hsv[:, 1])),
            "val": float(np.median(hsv[:, 2]))}


class OpenVocab:
    """YOLO-World-x。9/28 在俯视场景上 5 样东西对了 4 样，12 ms/帧；小档只对 1 样。"""

    def __init__(self, weights: str = "yolov8x-worldv2.pt", device: str = "cuda"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self._phrases: List[str] = []

    def detect(self, img_bgr: np.ndarray, phrases: Sequence[str], conf: float = 0.02,
               per_phrase: int = 1) -> List[Candidate]:
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
            c.hue, c.sat, c.val = st["hue"], st["sat"], st["val"]
            x0, y0, x1, y1 = c.xyxy
            c.area = float((x1 - x0) * (y1 - y0))
        return merge_duplicates(out)


def merge_duplicates(cands: List[Candidate], iou_thr: float = 0.5) -> List[Candidate]:
    """同一个物体会被几种说法各框一次；按 IoU 合并，保留置信度最高的那个，记下所有说法。"""
    def iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    cands = sorted(cands, key=lambda c: -c.conf)
    kept: List[Candidate] = []
    for c in cands:
        for k in kept:
            if iou(c.xyxy, k.xyxy) > iou_thr:
                k.extra.setdefault("also", []).append((c.phrase, round(c.conf, 3)))
                break
        else:
            kept.append(c)
    return kept


class ColourTracker:
    """目标被确认之后，用**它自己的**颜色直方图逐帧跟踪。

    为什么需要：腕部相机接近目标时，目标经常有一段藏在手指后面（示范里合爪时球就夹在两指间）。
    被挡住一半的乒乓球 YOLO-World 完全检不出来，第一次闭环就这样在离球很近时「看不到目标」停下。
    直方图从 YOLO-World+CLIP 确认过的那个框的中心取 —— 颜色是测出来的，不是手填的 HSV 区间。
    只接受离上一帧位置 `gate_px` 以内的连通块，防止跳到颜色相近的东西上（黄胶带和橙色球色相很近）。
    """

    def __init__(self, img_bgr, xyxy, gate_px: float = 90.0):
        import cv2

        x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = max(3, min(x1 - x0, y1 - y0) // 4)
        roi = img_bgr[cy - r:cy + r, cx - r:cx + r]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        self.last = np.array([cx, cy], float)
        self.size = float(max(x1 - x0, y1 - y0))
        self.gate = gate_px

    def update_from_detection(self, xyxy):
        x0, y0, x1, y1 = xyxy
        self.last = np.array([(x0 + x1) / 2, (y0 + y1) / 2], float)
        self.size = float(max(x1 - x0, y1 - y0))

    def mask(self, img_bgr, xyxy=None):
        """属于目标颜色的像素（反投影 + 阈值）。给了框就只看框内。"""
        import cv2

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        m = bp > 60
        if xyxy is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
            keep = np.zeros_like(m)
            keep[max(0, y0):y1, max(0, x0):x1] = True
            m &= keep
        return m

    def track(self, img_bgr):
        """返回 (xyxy, 可见面积占比) 或 None。"""
        import cv2

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        _, m = cv2.threshold(bp, 60, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        best, bestd = None, None
        for k in range(1, n):
            if st[k, 4] < 0.05 * self.size ** 2:
                continue
            d = float(np.linalg.norm(cen[k] - self.last))
            if d <= self.gate and (bestd is None or d < bestd):
                best, bestd = k, d
        if best is None:
            return None
        x, y, w, h, area = st[best]
        self.last = cen[best].astype(float)
        return np.array([x, y, x + w, y + h], float), float(area) / max(self.size ** 2 * 0.785, 1.0)


class WristDepth:
    """腕部相机：像素 + 深度 -> 相机系 3D（米）。参数全部来自相机本身。"""

    def __init__(self, path: Optional[Path] = None):
        d = json.loads((path or CALIB / "orbbec_params.json").read_text())
        self.scale_m = float(d["depth_scale_mm_per_unit"]) / 1000.0      # 0.1mm -> 1e-4 m
        di = d["depth_intrinsic"]
        self.fx, self.fy, self.cx, self.cy = di["fx"], di["fy"], di["cx"], di["cy"]
        self.params = d

    def depth_m(self, depth_raw: np.ndarray, u: float, v: float, patch: int = 7) -> Optional[float]:
        h, w = depth_raw.shape
        r = patch // 2
        u, v = int(round(u)), int(round(v))
        win = depth_raw[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)].astype(float)
        good = win[(win > 0) & (win < 65535)]
        if good.size < 3:
            return None
        return float(np.median(good)) * self.scale_m

    def box_depth_m(self, depth_raw: np.ndarray, xyxy, shrink: float = 0.3) -> Optional[float]:
        """框里最近那一截（10 分位）的深度：物体比背景近，取近端能躲开框里漏进来的桌面。"""
        x0, y0, x1, y1 = xyxy
        dx, dy = (x1 - x0) * shrink / 2, (y1 - y0) * shrink / 2
        h, w = depth_raw.shape
        a, b = int(max(0, y0 + dy)), int(min(h, y1 - dy))
        c, e = int(max(0, x0 + dx)), int(min(w, x1 - dx))
        win = depth_raw[a:b, c:e].astype(float)
        good = win[(win > 0) & (win < 65535)]
        if good.size < 5:
            return None
        return float(np.percentile(good, 10)) * self.scale_m

    def centre_depth_m(self, depth_raw: np.ndarray, xyxy, frac: float = 0.25) -> Optional[float]:
        """框中心一小圆盘（半径 = 框短边 × frac）内的深度中位数。

        对球这种目标，中心那一片全是球面；框的 10 分位会被边缘漏进来的手指（~8cm）拉近，
        第一次测点动映射时就因此跳出 8.7cm 的假位移。
        """
        x0, y0, x1, y1 = xyxy
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        r = max(2.0, frac * min(x1 - x0, y1 - y0))
        h, w = depth_raw.shape
        ys, xs = np.ogrid[:h, :w]
        m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        vals = depth_raw[m].astype(float)
        good = vals[(vals > 0) & (vals < 65535)]
        if good.size < 5:
            return None
        return float(np.median(good)) * self.scale_m

    def deproject(self, u: float, v: float, z_m: float) -> np.ndarray:
        """OpenCV 相机系：+x 右、+y 下、+z 沿光轴向前。"""
        return np.array([(u - self.cx) * z_m / self.fx, (v - self.cy) * z_m / self.fy, z_m])
