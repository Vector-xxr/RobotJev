#!/usr/bin/env python3
"""验证 Orbbec Gemini 305 的深度流能不能用。整条感知链的阻塞项，先测这个。

    python -m robojev.depth_probe
    python -m robojev.depth_probe --frames 120

2026-09-23 实测结论（在本机 5070Ti 上）：**能用，走 OBSENSOR 后端。**

  深度   640x480 uint16(mm)  22.1 Hz   有效像素 87.7%
  彩色   640x480 BGR         20.6 Hz
  内参   读出来是 nan，得自己标

走了三条路才找到对的那条，把死路记在这里，别再走一遍：

1. **OpenCV + v4l2 路径 → 死路。**
   `cv2.VideoCapture("/dev/v4l/by-path/...:1.0-...")` 打得开，但 `getBackendName()`
   是 **FFMPEG**，而 FFMPEG 处理不了 Z16，报 `ioctl(VIDIOC_QBUF): Bad file descriptor`。
   显式传 `cv2.CAP_V4L2` 更糟：这个 OpenCV 构建直接拒绝按路径打开
   （"backend is generally available but can't be used to capture by name"）。

2. **裸 v4l2-ctl → 能出流但没用。**
   `v4l2-ctl -d /dev/video0 --set-fmt-video=...pixelformat='Z16 ' --stream-mmap`
   确实能拉到 15 fps，证明节点是好的。但要在 Python 里用就得自己写 mmap 那套，
   而且拿不到和彩色的对齐。

3. **OBSENSOR 后端 → 对的路。**
   `cv2.VideoCapture(0, cv2.CAP_OBSENSOR)`，OpenCV 5.0 内建 Orbbec 支持。
   **一个 cap 同时给深度和彩色**，所以根本不用抢两个 v4l2 节点 ——
   这台是 USB 2.1（480M），两路 640x480@30 各 18.4 MB/s 本来是要打架的。

⚠️ **最重要的约束：OBSENSOR 和 v4l2 彩色路不能混用，而且污染是双向的。**

   跑过 OBSENSOR 之后，`:1.4` 彩色节点（video8）会卡死 —— 不是消失，是
   `read()` 再也不返回，连裸 `v4l2-ctl --stream-mmap` 也卡。而 ar5 数采栈的
   `cam_front` 正是这个节点。

   反过来一样：跑过 v4l2 彩色路之后再开 OBSENSOR，帧率从 24 Hz 塌到
   **0.3 Hz**（实测：彩色 1 帧 / 深度 2 帧 / 3 秒）。不报错，只是慢到没法用 ——
   这个坑尤其阴，看起来像「深度流本来就慢」，其实是上一个进程留下的状态。

   救回来的办法（实测有效）：
       python -m atom_robot.ar5.reset_camera_usb
       sleep 8          # ← 等 3 秒不够，必须 8 秒
       v4l2-ctl -d /dev/video8 --set-ctrl exposure_dynamic_framerate=0

   所以闭环里的分工只能是：
       Orbbec (腕/前视)  -> OBSENSOR 独占，同时拿彩色 + 深度
       UGREEN (俯视)     -> 照旧走 v4l2 by-path
   两台相机各走各的后端，不冲突。**不要**在同一个进程里既用 OBSENSOR
   又去开 Orbbec 的 :1.4。

顺带（这个也别忘）：OBSENSOR 不像 pyorbbecsdk 那样把 v4l2 节点整个弄没 ——
12 个节点跑完还都在，uvcvideo 没被摘掉。这条路比 pyorbbecsdk 干净。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# Orbbec 在 OBSENSOR 后端下的设备序号。这台机器上只有一个 Orbbec，所以是 0。
# 注意这个 0 **不是** /dev/video0，是 OBSENSOR 后端自己的设备枚举。
OBSENSOR_INDEX = 0

# 深度图里的无效值。Orbbec 用 0 表示没测到，65535 表示超量程/溢出，两个都要当无效。
DEPTH_INVALID = (0, 65535)


def open_rgbd(index: int = OBSENSOR_INDEX):
    """打开 Orbbec，一个 cap 同时供深度和彩色。"""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        raise IOError(
            f"OBSENSOR 打不开设备 {index}。\n"
            "  - 确认 OpenCV >= 4.8 且编译时带了 obsensor 支持\n"
            "  - 确认相机没被别的进程占着：fuser /dev/video0\n"
            "  - 刚跑过用 :1.4 彩色路的程序？先复位：\n"
            "      python -m atom_robot.ar5.reset_camera_usb && sleep 8"
        )
    return cap


def read_rgbd(cap):
    """抓一组（彩色, 深度）。任一路没拿到就给 None。

    必须 grab 一次再 retrieve 两次 —— 两路来自同一次 grab，这样才是同一时刻的。
    分别 grab 会让彩色和深度差一帧，物体在动的时候就对不上了。
    """
    import cv2

    if not cap.grab():
        return None, None
    ok_c, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
    ok_d, depth = cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
    return (color if ok_c else None), (depth if ok_d else None)


def depth_valid_mask(depth: np.ndarray) -> np.ndarray:
    m = np.ones(depth.shape, dtype=bool)
    for bad in DEPTH_INVALID:
        m &= depth != bad
    return m


def describe(depth: np.ndarray) -> str:
    valid = depth_valid_mask(depth)
    if not valid.any():
        return "整帧无有效深度"
    v = depth[valid]
    h, w = depth.shape
    c = depth[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    cv_ = depth_valid_mask(c)
    return (f"有效 {100 * valid.mean():5.1f}% (中心区 {100 * cv_.mean():5.1f}%)  "
            f"min={v.min()} med={int(np.median(v))} max={v.max()} mm")


def run(frames: int) -> int:
    import cv2

    print(f"OpenCV {cv2.__version__}  OBSENSOR 设备 {OBSENSOR_INDEX}")
    cap = open_rgbd()
    print(f"backend = {cap.getBackendName()}")

    fx = cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FX)
    fy = cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FY)
    cx = cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CX)
    cy = cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CY)
    print(f"出厂内参 fx={fx} fy={fy} cx={cx} cy={cy}")
    if not all(np.isfinite([fx, fy, cx, cy])):
        print("  → 内参是 nan，这个后端没把出厂标定透出来。得自己标，见 robojev/calibrate.py")

    nd = nc = 0
    last_d = last_c = None
    t0 = time.perf_counter()
    for _ in range(frames):
        color, depth = read_rgbd(cap)
        if color is not None:
            nc += 1
            last_c = color
        if depth is not None:
            nd += 1
            last_d = depth
    el = time.perf_counter() - t0
    cap.release()

    print(f"\n{frames} 次 grab / {el:.1f}s")
    print(f"  彩色 {nc:4d} 帧  {nc / el:5.1f} Hz  shape={None if last_c is None else last_c.shape}")
    print(f"  深度 {nd:4d} 帧  {nd / el:5.1f} Hz  shape={None if last_d is None else last_d.shape} "
          f"dtype={None if last_d is None else last_d.dtype}")
    if last_d is not None:
        print(f"  {describe(last_d)}")

    ok = last_d is not None and last_c is not None and nd / el >= 10.0 and nc / el >= 10.0
    print("\n" + "=" * 62)
    if ok:
        print("结论：RGBD 可用。深度和彩色同源同步，按真深度方案往下写。")
        print("提醒：跑完这个脚本，ar5 的 cam_front(:1.4) 会卡住，要复位才能用：")
        print("  python -m atom_robot.ar5.reset_camera_usb && sleep 8")
    else:
        print("结论：拿不到稳定的 RGBD，回落桌面平面假设。")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=90, help="抓多少次 grab")
    args = ap.parse_args()
    return run(args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
