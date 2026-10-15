#!/usr/bin/env python3
"""从 Orbbec Gemini 305 本身读出厂标定参数，存成 calibration/orbbec_params.json。

    # 必须在 lerobot 环境里跑（pyorbbecsdk 只装在那）
    ~/miniforge3/envs/lerobot/bin/python -m robojev.read_orbbec_params                        # 腕部 305
    ~/miniforge3/envs/lerobot/bin/python -m robojev.read_orbbec_params --serial CP0BB53000FC \
        --out orbbec335_params.json                                                        # 俯视 335

为什么要单独跑这个：闭环用的 OpenCV OBSENSOR 后端对这台相机**不透出内参**，
`CAP_PROP_OBSENSOR_INTRINSIC_*` 一直是 nan（流开着读也一样）。要做「像素 + 深度 → 3D」，
内参、深度单位、深度到彩色的外参都得是相机自己报的数，不能猜。

⚠️ pyorbbecsdk 会把内核驱动从彩色接口上摘掉，v4l2 节点消失。所以这里**只读一次**，
读完立刻发 USB 复位，并检查节点回来了。平时的取流照旧走 OBSENSOR。
"""

from __future__ import annotations

import json

import numpy as np
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "calibration" / "orbbec_params.json"
PROJECT = Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"
W, H, FPS = 640, 480, 30


def intr(i):
    return {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.cx), "cy": float(i.cy),
            "width": int(i.width), "height": int(i.height)}


def dist(d):
    out = {k: float(getattr(d, k)) for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")}
    model = getattr(d, "model", None)
    out["model"] = str(model) if model is not None else None
    # OpenCV 的顺序：k1,k2,p1,p2,k3,k4,k5,k6
    out["opencv"] = [out["k1"], out["k2"], out["p1"], out["p2"], out["k3"], out["k4"], out["k5"], out["k6"]]
    return out


SERIAL = "CV27864000V2"          # 默认腕部 305；两台 Orbbec 同厂商号，必须按序列号选和复位


def reset_usb(serial: str) -> None:
    """把 v4l2 节点要回来。**无论前面成败都要跑**：第一版在写 JSON 时抛了异常，
    复位没执行，节点就一直被摘着。只复位这台（按序列号），别碰另一台 Orbbec。"""
    port = None
    for dev in Path("/sys/bus/usb/devices").iterdir():
        v, sn = dev / "idVendor", dev / "serial"
        if ":" not in dev.name and v.exists() and v.read_text().strip() == "2bc5" \
                and sn.exists() and sn.read_text().strip() == serial:
            port = dev.name
    if port:
        subprocess.run([str(Path.home() / "miniforge3/envs/laya/bin/python"),
                        "-m", "atom_robot.ar5.reset_camera_usb", "--port", port],
                       cwd=str(PROJECT), capture_output=True, timeout=60)
        time.sleep(8)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=SERIAL)
    ap.add_argument("--out", default=OUT.name, help="写到 calibration/ 下的文件名")
    a = ap.parse_args()
    try:
        return _read(a.serial, OUT.parent / a.out)
    finally:
        reset_usb(a.serial)


def _read(serial: str, out_path: Path) -> int:
    import pyorbbecsdk as ob

    ctx = ob.Context()                       # 要一直引用着：临时 Context 被回收后设备列表失效（deviceMgr NULL）
    devs = ctx.query_devices()
    dev = devs.get_device_by_serial_number(serial)
    if dev is None:
        raise SystemExit(f"找不到序列号 {serial} 的 Orbbec")
    pipe = ob.Pipeline(dev)
    cfg = ob.Config()
    got = {}
    for sensor, fmt, key in ((ob.OBSensorType.COLOR_SENSOR, ob.OBFormat.RGB, "color"),
                             (ob.OBSensorType.DEPTH_SENSOR, ob.OBFormat.Y16, "depth")):
        profiles = pipe.get_stream_profile_list(sensor)
        try:
            prof = profiles.get_video_stream_profile(W, H, fmt, FPS)
        except Exception:                                       # noqa: BLE001
            prof = profiles.get_default_video_stream_profile()
        got[key] = f"{prof.get_width()}x{prof.get_height()}@{prof.get_fps()} {prof.get_format()}"
        cfg.enable_stream(prof)
    pipe.start(cfg)
    try:
        depth_scale = None
        for _ in range(60):
            fs = pipe.wait_for_frames(200)
            if fs is None:
                continue
            d = fs.get_depth_frame()
            if d is not None:
                depth_scale = float(d.get_depth_scale())
                break
        p = pipe.get_camera_param()
        ext = p.transform
        out = {
            "streams": got,
            "depth_scale_mm_per_unit": depth_scale,
            "color_intrinsic": intr(p.rgb_intrinsic),
            "depth_intrinsic": intr(p.depth_intrinsic),
            # 8 个系数全存（有理模型 k1..k6 + 切向 p1,p2）。第一版只存了 k1,k2,k3,p1,p2，
            # 而这颗广角镜头 k1=-1.26，缺了 k4..k6（分母那一半）去畸变会完全算错。
            "color_distortion": dist(p.rgb_distortion),
            "depth_distortion": dist(p.depth_distortion),
            # depth -> color：p_color = R @ p_depth + t，t 单位 mm
            "depth_to_color": {"R": np.asarray(ext.rot, float).reshape(3, 3).tolist(),
                               "t_mm": np.asarray(ext.transform, float).ravel().tolist()},
            "serial": serial,
            "read_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    finally:
        pipe.stop()
        del pipe

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"写到 {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
