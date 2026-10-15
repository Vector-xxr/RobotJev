"""按颜色找目标。给 YOLO 认不出来的东西用。

`detector.py` 那套是 COCO 80 类，桌面上大量东西不在里面 —— 胶带卷、积木、
零件、任何自制的标定块。「蓝色圆柱体」就是一个例子：COCO 里没有 cylinder，
也没有 tape，YOLO 再大也变不出这个类。

对这类目标，颜色分割比检测网络**更可靠**，不是将就：
  - 不需要训练数据，也不需要微调
  - 在固定光照的工作台上，一个 HSV 区间就能稳定分出来
  - 单帧 1~2 ms，比 yolo11s 的 45 ms 快一个量级
代价是它只认颜色 —— 场上有第二个同色物体就会混，所以下面按面积和圆度筛，
并且把所有候选都返回，让上层决定（而不是偷偷挑一个）。

输出的是 `detector.Detection`，和 YOLO 那条路完全同构，`localizer` 不用改。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class ColorSpec:
    """一个颜色的 HSV 区间。OpenCV 的 H 是 0..179，不是 0..359。"""

    name: str
    h_lo: int
    h_hi: int
    s_lo: int = 80
    s_hi: int = 255
    v_lo: int = 40
    v_hi: int = 255

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        import cv2

        if self.h_lo <= self.h_hi:
            return cv2.inRange(hsv, (self.h_lo, self.s_lo, self.v_lo),
                               (self.h_hi, self.s_hi, self.v_hi))
        # 红色跨 0 度，要拆成两段再并起来。
        a = cv2.inRange(hsv, (self.h_lo, self.s_lo, self.v_lo), (179, self.s_hi, self.v_hi))
        b = cv2.inRange(hsv, (0, self.s_lo, self.v_lo), (self.h_hi, self.s_hi, self.v_hi))
        return cv2.bitwise_or(a, b)


# 常用颜色。这些区间是通用起点，**在你自己的光照下要用 --tune 复核**：
# 工作台的白泡沫板反光会把 V 拉高、把 S 压低，日光灯和 LED 的色温也会挪 H。
PRESETS = {
    # 蓝：H 100~130 覆盖从青蓝到蓝紫。S 下限抬到 90 是为了把灰色/黑色的
    # 阴影排除掉 —— 低饱和的暗区在 HSV 里 H 是噪声，什么颜色都可能命中。
    "blue": ColorSpec("blue", 100, 130, s_lo=90, v_lo=50),
    "yellow": ColorSpec("yellow", 20, 35, s_lo=90, v_lo=80),
    "green": ColorSpec("green", 40, 85, s_lo=70, v_lo=50),
    "red": ColorSpec("red", 170, 10, s_lo=90, v_lo=50),      # 跨 0 度
    "orange": ColorSpec("orange", 10, 20, s_lo=110, v_lo=80),
}


class ColorDetector:
    """按颜色找连通块，返回和 YOLO 同构的 Detection。"""

    def __init__(self, spec: ColorSpec | str = "blue",
                 min_area_px: int = 300, max_area_px: int = 120_000,
                 min_circularity: float = 0.0, open_px: int = 5, close_px: int = 7):
        self.spec = PRESETS[spec] if isinstance(spec, str) else spec
        self.min_area = min_area_px
        self.max_area = max_area_px
        self.min_circularity = min_circularity
        self.open_px = open_px
        self.close_px = close_px

    def detect(self, image: np.ndarray, keep=None) -> List[Detection]:
        """`keep` 只为和 detector.Detector 接口一致，这里忽略。"""
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m = self.spec.mask(hsv)

        # 先开后闭：开运算去掉零星噪点（桌面上的蓝色反光、远处的蓝色杂物边缘），
        # 闭运算把物体内部的高光空洞填上 —— 胶带卷中间那圈反光会把一个物体
        # 切成两个连通块，不闭上的话面积和中心都是错的。
        if self.open_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_px, self.open_px))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if self.close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_px, self.close_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if not (self.min_area <= area <= self.max_area):
                continue
            peri = float(cv2.arcLength(c, True))
            circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
            if circ < self.min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # conf 用「轮廓面积 / 外接框面积」的填充率：一个实心圆柱俯视接近 0.79
            # （π/4），一条细长的反光条会低得多。这不是概率，只是个可比较的分数，
            # 命名沿用 Detection.conf 是为了接口一致。
            fill = area / max(w * h, 1)
            out.append(Detection(cls_name=self.spec.name, conf=float(fill),
                                 xyxy=np.array([x, y, x + w, y + h], dtype=np.float64)))
        out.sort(key=lambda d: -d.area)
        return out

    def debug_mask(self, image: np.ndarray) -> np.ndarray:
        import cv2

        return self.spec.mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    @staticmethod
    def best(dets, cls_name=None) -> Optional[Detection]:
        """面积最大的那个。同色多目标时这是最不容易错的选法 ——
        小的那些通常是反光、边缘和背景里的同色杂物。"""
        return dets[0] if dets else None
