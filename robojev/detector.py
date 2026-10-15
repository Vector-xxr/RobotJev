"""YOLO 目标检测。只做一件事：给一帧图和几个想要的类名，返回框。

**为什么默认不是最小的 yolo11n。**
2026-09-23 在本机俯视工作台的实拍图上量过（画面里有一个水瓶）：

    模型        水瓶 conf   单帧耗时   结果
    yolo11n       0.07       30 ms    认不出来，还把它误检成 cup(0.13)
    yolo11s       0.48       45 ms    能用
    yolo11m       0.75       84 ms    很稳

差别不是阈值调得对不对 —— 把 conf 压到 0.01，n 给出的最高分仍然是错的那个 cup。
原因是 COCO 的训练图基本是平视角，**俯视桌面是分布外**，小模型先垮。

闭环周期是 100 ms，45 ms 和 84 ms 都放得下，所以没有理由为了省 15 ms 用一个
认不出目标的模型。要更稳就换 `yolo11m`。真要认 COCO 以外的东西（比如真正的
盘子，见 `parser.py`），只能自己标数据微调。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL = "yolo11s.pt"


@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: np.ndarray            # (4,) float，像素

    @property
    def centre(self) -> tuple:
        x0, y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def bottom_centre(self) -> tuple:
        """框底边中点。

        物体贴在桌面上时，这个点才是它和桌面的接触处；框的几何中心是物体
        **半高**的位置，拿它去和桌面平面求交会把目标算得比实际偏远。
        平面假设那条路必须用这个点，深度那条路用哪个都行。
        """
        x0, _y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), y1)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.xyxy
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class Detector:
    def __init__(self, model: str = MODEL, conf: float = 0.25, device: str = "cuda"):
        from ultralytics import YOLO

        self.conf = conf
        self._model = YOLO(model)
        self._device = device
        self.names: Dict[int, str] = dict(self._model.names)
        logger.info(f"[YOLO] {model} 就绪，{len(self.names)} 类")

    @property
    def class_names(self) -> List[str]:
        return list(self.names.values())

    def detect(self, image: np.ndarray,
               keep: Optional[Sequence[str]] = None) -> List[Detection]:
        """跑一帧。`keep` 给了就只返回这些类。"""
        want = set(keep) if keep else None
        # classes= 让 YOLO 在 NMS 阶段就把别的类丢掉，比拿回来再过滤快，
        # 也避免一桌子杂物把 max_det 占满导致目标反而被挤掉。
        ids = None
        if want:
            ids = [i for i, n in self.names.items() if n in want]
            if not ids:
                return []
        res = self._model.predict(image, conf=self.conf, classes=ids,
                                  device=self._device, verbose=False)[0]

        out: List[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            name = self.names[int(b.cls.item())]
            if want and name not in want:
                continue
            out.append(Detection(cls_name=name, conf=float(b.conf.item()),
                                 xyxy=b.xyxy[0].cpu().numpy().astype(np.float64)))
        return out

    @staticmethod
    def best(dets: Sequence[Detection], cls_name: str) -> Optional[Detection]:
        """某一类里置信度最高的那个。

        同类有多个时按置信度挑，不按大小或位置 —— 「最像苹果的那个」比
        「最大的那个」更少出错，尤其是背景里有半个苹果被框进来的时候。
        """
        cand = [d for d in dets if d.cls_name == cls_name]
        return max(cand, key=lambda d: d.conf) if cand else None
