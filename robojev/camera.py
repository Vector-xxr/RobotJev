"""RGBD 相机源：Orbbec 走 OBSENSOR 拿彩色+深度，UGREEN 走普通 v4l2 拿彩色。

为什么分两个后端，以及为什么它们必须隔离，见 `robojev/depth_probe.py` 的模块
文档。一句话：OBSENSOR 和 v4l2 在同一台 Orbbec 上互相污染，切换之间必须 USB
复位 + 等 8 秒，所以这里在打开前**自动复位一次**，省得每次手动救。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Orbbec 在 OBSENSOR 后端下的序号，不是 /dev/videoN。有序列号时按序列号现算（见 obsensor_index）。
OBSENSOR_INDEX = 0
ORBBEC_VID = "2bc5"
# 按序列号认相机（2026-09-29）：桌上有两台 Orbbec 以后，按厂商 ID 找会找到哪台算哪台 ——
# 复位可能复位错、打开可能打开错。口号也不写死（9/28 重启后腕部从 1-10.3 换到了 1-10.2）。
WRIST_SERIAL = "CV27864000V2"        # 腕部 Gemini 305（产品号 0840）
TOP_SERIAL = "CP0BB53000FC"          # 俯视 Gemini 335（产品号 0800）


def _orbbec_devices():
    """USB 上所有 Orbbec：[(口号, 序列号, 型号)]。"""
    from pathlib import Path

    out = []
    for dev in Path("/sys/bus/usb/devices").iterdir():
        vid = dev / "idVendor"
        if ":" in dev.name or not vid.exists() or vid.read_text().strip() != ORBBEC_VID:
            continue
        rd = lambda f: (dev / f).read_text().strip() if (dev / f).exists() else ""
        out.append((dev.name, rd("serial"), rd("product")))
    return out


def find_orbbec_port(serial: str | None = WRIST_SERIAL) -> str | None:
    """按序列号找 Orbbec 的 USB 口号（形如 '1-10.2'）。serial=None 时只有一台才返回。"""
    devs = _orbbec_devices()
    if serial is not None:
        return next((port for port, sn, _ in devs if sn == serial), None)
    return devs[0][0] if len(devs) == 1 else None


def obsensor_index(serial: str) -> int | None:
    """序列号 -> OBSENSOR 序号。OBSENSOR 按 **USB 口路径** 的顺序给设备编号（不是 /dev/videoN 的顺序）：
    9/29 两次观察都对得上 —— 1-10.2(305) 排在 1-10.4(335) 前；换到 USB3 后 2-5(335) 排在 2-6(305) 前，
    而那时 305 的 videoN 反而更小（第一版按 videoN 排，结果按 335 的序列号打开了腕部 305）。
    打开后 OrbbecRGBD 还会用「335 报内参、305 报 nan」再核对一次。"""
    import re

    key = lambda port: [int(x) for x in re.split(r"[-.]", port) if x.isdigit()]
    devs = sorted(_orbbec_devices(), key=lambda d: key(d[0]))
    order = [sn for _, sn, _ in devs]
    return order.index(serial) if serial in order else None


def _reports_intrinsics(cap) -> bool:
    """OpenCV OBSENSOR 对 335 报内参（深度已对齐彩色），对 305 报 nan —— 用来核对打开的是哪台。"""
    import cv2

    return bool(np.isfinite(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FX)))


# 俯视 UGREEN，by-path 绑物理插口 —— 两台 UGREEN 序列号都是 SN0001，只能靠口区分。
UGREEN_NODE = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:10.4:1.0-video-index0"

DEPTH_INVALID = (0, 65535)


def _disable_dynamic_framerate(node: str) -> None:
    """关掉「用帧率换曝光」。不关的话室内光下这些 UVC 相机只有 19 Hz。

    这是个 UVC 控制，OpenCV 没有对应的属性，只能走 v4l2-ctl。失败不致命，
    只是掉帧率，所以吞掉异常。
    """
    import subprocess

    try:
        subprocess.run(["v4l2-ctl", "-d", node, "--set-ctrl",
                        "exposure_dynamic_framerate=0"], capture_output=True, timeout=3)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"[Cam] 关不掉 exposure_dynamic_framerate: {exc}")


def reset_orbbec(wait_s: float = 8.0, serial: str | None = WRIST_SERIAL) -> bool:
    """给 Orbbec 发一次 USB 复位，然后等它重新枚举。

    **等待时间不能短。** 实测等 3 秒之后彩色节点仍然是卡死的，等 8 秒才活过来。
    复位本身是瞬时的，慢的是设备重新枚举 + uvcvideo 重新绑定。
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"          # .../数据_硬件/SO101（atom_robot 所在）
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    port = find_orbbec_port(serial)
    if port is None:
        logger.warning(f"[RGBD] USB 上找不到序列号 {serial} 的 Orbbec，跳过复位"
                       f"（现有：{_orbbec_devices()}）")
        return False
    try:
        from atom_robot.ar5.reset_camera_usb import reset  # type: ignore
        reset(port)
    except (ImportError, OSError, SystemExit) as exc:
        logger.warning(f"[RGBD] USB 复位没发出去：{exc}")
        return False
    time.sleep(wait_s)
    return True


@dataclass
class Intrinsics:
    """针孔内参。OBSENSOR 后端在这台机器上读出来是 nan，所以必须自己标。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def deproject(self, u: float, v: float, z_m: float) -> np.ndarray:
        """像素 + 深度 → 相机系 3D 点（米）。

        相机系按 OpenCV 惯例：+X 右、+Y 下、+Z 前（沿光轴射出）。
        """
        return np.array([(u - self.cx) * z_m / self.fx,
                         (v - self.cy) * z_m / self.fy,
                         z_m], dtype=np.float64)


@dataclass
class RGBDFrame:
    color: Optional[np.ndarray] = None       # (H, W, 3) BGR uint8
    depth: Optional[np.ndarray] = None       # (H, W) uint16, 毫米；俯视 UGREEN 这路永远是 None
    stamp: float = 0.0

    @property
    def ok(self) -> bool:
        """够不够用。**只要求彩色** —— 俯视那台 UGREEN 根本没有深度，
        要求 depth 非空会让它永远 not ok，闭环一帧都跑不起来。
        深度缺失由 `DepthLocalizer` 自己处理（`depth_at` 返回 None → 该帧定不出位置），
        用 `PlaneLocalizer` 时压根不看深度。"""
        return self.color is not None

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def depth_at(self, u: int, v: int, patch: int = 5) -> Optional[float]:
        """取 (u,v) 附近一小块的中位深度，单位米；全无效返回 None。

        用中位数不用单点：Orbbec 在物体边缘、反光面和黑色表面上会出空洞和
        跳变，单个像素读到 0 或者读到背景距离都很常见。取 patch 内所有有效值
        的中位数，边缘上也能给出偏向物体而不是背景的估计。
        """
        if self.depth is None:
            return None
        h, w = self.depth.shape
        r = patch // 2
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
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
    """俯视 UGREEN，只有彩色，没有深度。

    和 Orbbec 那路完全不冲突：不同 USB 口、不同后端（这路是纯 V4L2），
    所以不受 OBSENSOR 那套互相污染的约束，可以和 OrbbecRGBD 同时开。

    深度永远是 None，所以**必须配 `PlaneLocalizer`** —— 用 `DepthLocalizer`
    的话每一帧都定不出位置，表现是「目标可见但位置未知」，闭环原地不动。
    """

    def __init__(self, node: str = UGREEN_NODE, width: int = 640, height: int = 480,
                 fps: int = 30):
        self.node, self.width, self.height, self.fps_target = node, width, height, fps
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
            raise IOError(
                f"打不开俯视相机 {self.node}\n"
                "  先查有没有别的进程占着：fuser /dev/video2\n"
                "  （采数据的 collect 会占这个节点）"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        # 这两条是 ar5 相机驱动上量出来的，别改：
        # BUFFERSIZE=1 会让帧率正好减半，动态帧率开着室内光下只有 19 Hz。
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        _disable_dynamic_framerate(self.node)
        self._cap = cap

        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name="robojev-top", daemon=True)
        self._thread.start()

        deadline = time.perf_counter() + 8.0
        while True:
            with self._lock:
                if self._frame.ok:
                    break
            if time.perf_counter() > deadline:
                self.disconnect()
                raise IOError(f"俯视相机打开了但 8 秒没出帧（{self.errors} 次失败读）")
            time.sleep(0.02)
        logger.info(f"[Top] 俯视相机就绪 {self.fps:.1f} Hz")

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
        return (self._count - 1) / max(time.perf_counter() - self._t0, 1e-9)

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
    """Orbbec Gemini 305，一个 cap 同时出彩色和深度。

    必须用后台线程抽帧：OBSENSOR 的 grab() 是阻塞的，在控制循环里直接调
    会把循环频率钉死在相机帧率上，而且相机偶尔一次慢帧会直接传导成控制抖动。
    """

    def __init__(self, index: int | None = None, auto_reset: bool = True, serial: str = WRIST_SERIAL):
        """默认按序列号打开腕部那台；给了 index 就按序号（旧用法）。"""
        self.index = index
        self.serial = serial
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
            # 上一个进程可能留下被 v4l2 污染的状态，那样 OBSENSOR 会掉到 0.3 Hz
            # 而且不报错。复位一次是最省心的开场。
            logger.info(f"[RGBD] 复位 Orbbec {self.serial} 并等待重新枚举 ...")
            reset_orbbec(serial=self.serial)

        if self.index is None:
            idx = obsensor_index(self.serial)
            if idx is None:
                raise IOError(f"USB 上找不到序列号 {self.serial} 的 Orbbec（现有：{_orbbec_devices()}）")
            self.index = idx
        cap = cv2.VideoCapture(self.index, cv2.CAP_OBSENSOR)
        want_intr = self.serial == TOP_SERIAL
        if cap.isOpened() and len(_orbbec_devices()) > 1 and _reports_intrinsics(cap) != want_intr:
            # 编号和序列号对不上（口换过、枚举顺序变了）：换另一个编号，再核对一次
            logger.warning(f"[RGBD] 序号 {self.index} 打开的不是 {self.serial}，改用另一个序号")
            cap.release()
            self.index = 1 - self.index if self.index in (0, 1) else self.index
            cap = cv2.VideoCapture(self.index, cv2.CAP_OBSENSOR)
            if not cap.isOpened() or _reports_intrinsics(cap) != want_intr:
                cap.release()
                raise IOError(f"两个 OBSENSOR 序号都对不上序列号 {self.serial}（现有：{_orbbec_devices()}）")
        if not cap.isOpened():
            raise IOError(
                f"OBSENSOR 打不开设备 {self.index}。\n"
                "  先确认没有别的进程占着 Orbbec：fuser /dev/video0\n"
                "  再手动复位：python -m atom_robot.ar5.reset_camera_usb && sleep 8"
            )
        self._cap = cap
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name="robojev-rgbd", daemon=True)
        self._thread.start()

        deadline = time.perf_counter() + 10.0
        while True:
            with self._lock:
                if self._frame.ok:
                    break
            if time.perf_counter() > deadline:
                self.disconnect()
                raise IOError(
                    f"Orbbec 打开了但 10 秒内没出成对的帧（{self.errors} 次失败）。\n"
                    "  多半是被 v4l2 路污染了，手动复位后重试："
                    "python -m atom_robot.ar5.reset_camera_usb && sleep 8"
                )
            time.sleep(0.02)
        logger.info(f"[RGBD] Orbbec 就绪，{self.fps:.1f} Hz")

    def _run(self) -> None:
        import cv2

        while not self._halt.is_set():
            if not self._cap.grab():
                self.errors += 1
                self._halt.wait(0.02)
                continue
            ok_c, color = self._cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
            ok_d, depth = self._cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
            # 只在两路都拿到时才更新。半套帧（只有彩色没有深度）对下游没用，
            # 而且会让 depth_at() 拿到上一帧的深度配这一帧的像素坐标。
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
            # 浅拷贝够了：抽帧线程每次是整个换掉 RGBDFrame，不会就地改数组。
            return f

    @property
    def fps(self) -> float:
        if self._t0 is None or self._count < 2:
            return 0.0
        return (self._count - 1) / max(time.perf_counter() - self._t0, 1e-9)

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
