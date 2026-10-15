"""3D 定位：把 YOLO 的像素框变成机械臂基座系的 (x, y, z)。

这一层是**可插拔**的，因为拿深度和不拿深度是两条完全不同的路：

    DepthLocalizer  —— 用 Orbbec 的深度图直接读 z，再反投影。要内参 + 外参。
    PlaneLocalizer  —— 假设物体都贴在桌面上，z 已知，射线与平面求交。要内参 + 外参。

两个都实现了 `Localizer` 协议，闭环只认协议，换相机方案只换这里一个类。

⚠️ **外参没标定之前，这一层出来的坐标是错的。**
`Extrinsics.load()` 读不到标定文件时会给一个单位阵并大声警告。单位阵的意思是
「相机就架在基座原点、朝向和基座系一致」，这在物理上不可能成立。拿这个跑闭环，
臂会朝一个和目标无关的方向稳定地走 —— 看起来像「模型选错动作」，其实是坐标系错。
先跑 `python -m robojev.calibrate`，再谈闭环效果。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .camera import Intrinsics, RGBDFrame
from .detector import Detection

logger = logging.getLogger(__name__)

CALIB_DIR = Path(__file__).resolve().parent / "calibration"


@dataclass
class Extrinsics:
    """相机在基座系下的位姿：X_base = R @ X_cam + t。

    相机是固定在架子上的（eye-to-hand），所以这是个常量，标一次就行 ——
    不像装在腕上那样每帧都要乘一遍 FK。
    """

    R: np.ndarray               # (3,3)
    t: np.ndarray               # (3,)
    calibrated: bool = True
    source: str = ""

    def to_base(self, xyz_cam: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(xyz_cam, dtype=np.float64).reshape(3) + self.t

    @classmethod
    def identity(cls) -> "Extrinsics":
        return cls(R=np.eye(3), t=np.zeros(3), calibrated=False, source="identity")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Extrinsics":
        path = path or (CALIB_DIR / "extrinsics.json")
        if not path.exists():
            logger.warning(
                "=" * 70 + "\n"
                f"外参标定文件不存在：{path}\n"
                "  暂时用单位阵顶上 —— **这条路上算出来的基座系坐标是错的**。\n"
                "  臂会朝一个和目标无关的固定方向走，且看起来很像「模型选错了」。\n"
                "  先跑：python -m robojev.calibrate\n" + "=" * 70
            )
            return cls.identity()
        d = json.loads(path.read_text())
        return cls(R=np.array(d["R"], dtype=np.float64).reshape(3, 3),
                   t=np.array(d["t"], dtype=np.float64).reshape(3),
                   calibrated=True, source=str(path))


def load_intrinsics(path: Optional[Path] = None) -> Intrinsics:
    """读内参。没标定过就按 FOV 粗估一个，并说清楚这是估的。"""
    path = path or (CALIB_DIR / "intrinsics.json")
    if path.exists():
        d = json.loads(path.read_text())
        return Intrinsics(**d)
    # Gemini 305 彩色标称水平 FOV 约 65°。fx = (W/2) / tan(HFOV/2)。
    # 这个数只够让流水线跑起来，横向误差能到厘米级，别拿它做真实抓取。
    w, h = 640, 480
    fx = (w / 2) / np.tan(np.deg2rad(65.0) / 2)
    logger.warning(
        f"内参标定文件不存在：{path}\n"
        f"  按标称 FOV 65° 粗估 fx=fy={fx:.1f}，误差到厘米级。\n"
        f"  要真抓东西先跑：python -m robojev.calibrate --intrinsics"
    )
    return Intrinsics(fx=fx, fy=fx, cx=w / 2, cy=h / 2, width=w, height=h)


class Localizer(Protocol):
    def locate(self, det: Detection, frame: RGBDFrame) -> Optional[np.ndarray]:
        """返回基座系 (x,y,z) 米，定不出来返回 None。"""
        ...


@dataclass
class DepthLocalizer:
    """用深度图读 z，再按内参反投影。Orbbec 对准工作台之后走这条。"""

    intr: Intrinsics
    extr: Extrinsics
    patch: int = 7

    def locate(self, det: Detection, frame: RGBDFrame) -> Optional[np.ndarray]:
        u, v = det.centre
        z = frame.depth_at(int(round(u)), int(round(v)), patch=self.patch)
        if z is None:
            # 框中心是空洞（反光、黑色、边缘都会），退到底边中点再试一次。
            u2, v2 = det.bottom_centre
            z = frame.depth_at(int(round(u2)), int(round(v2)), patch=self.patch)
            if z is None:
                return None
            u, v = u2, v2
        return self.extr.to_base(self.intr.deproject(u, v, z))


@dataclass
class PlaneLocalizer:
    """假设物体贴在一个已知高度的水平桌面上，射线与该平面求交。

    不需要深度，所以俯视那台没有深度的 UGREEN 也能用。代价是物体一旦离开
    桌面（被抓起来、叠在别的东西上），解出来的 z 就是错的 —— 这时候应该
    改用「已抓住」的状态，而不是继续信这个位置。
    """

    intr: Intrinsics
    extr: Extrinsics
    table_z: float = 0.0        # 桌面在基座系下的高度，米

    def locate(self, det: Detection, frame: RGBDFrame) -> Optional[np.ndarray]:
        u, v = det.bottom_centre          # 贴桌面的那个点，见 Detection.bottom_centre
        # 相机系里这条射线的方向（z=1 的归一化坐标）
        d_cam = np.array([(u - self.intr.cx) / self.intr.fx,
                          (v - self.intr.cy) / self.intr.fy,
                          1.0], dtype=np.float64)
        # 转到基座系：原点是相机光心，方向只转不平移
        o = self.extr.t
        d = self.extr.R @ d_cam
        if abs(d[2]) < 1e-9:
            return None                    # 射线和桌面平行，交不上
        s = (self.table_z - o[2]) / d[2]
        if s <= 0:
            return None                    # 交点在相机背后
        return o + s * d
