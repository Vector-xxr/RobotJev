#!/usr/bin/env python3
"""实时看顶部相机（Gemini 335）：左彩色、右深度伪彩色，标帧率。按 q / Esc 关。

    conda activate laya; cd ~/yuhang_workspace/算法/Jev/RoboJev
    PYTHONPATH= python -m robojev.view_top            # 只看顶部
    PYTHONPATH= python -m robojev.view_top --wrist    # 顶部 + 腕部

⚠️ 开着的时候相机被它占着，跑抓取前先关掉（抓取程序打开相机时会复位它，这个窗口会断）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))


def panel(frame, label, fps, depth_scale_m):
    import cv2

    c = frame.color.copy()
    d = frame.depth
    if d is not None:
        v = d[(d > 0) & (d < 65535)]
        top = float(np.percentile(v, 95)) if v.size else 1.0
        dv = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255.0 / max(top, 1.0)), cv2.COLORMAP_JET)
        dv[(d == 0) | (d == 65535)] = 0
        dv = cv2.resize(dv, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = d.shape
        cz = d[h // 2, w // 2]
        txt = f"centre depth {cz * depth_scale_m * 1000:.0f}mm" if 0 < cz < 65535 else "centre depth --"
        cv2.drawMarker(dv, (c.shape[1] // 2, c.shape[0] // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(dv, txt, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        dv = np.zeros_like(c)
    out = np.hstack([c, dv])
    cv2.putText(out, f"{label}  {fps:.1f} Hz", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrist", action="store_true", help="同时显示腕部相机")
    a = ap.parse_args()
    import cv2

    from .camera import TOP_SERIAL, WRIST_SERIAL, OrbbecRGBD

    cams = [("top Gemini 335", OrbbecRGBD(serial=TOP_SERIAL), 0.001)]
    if a.wrist:
        cams.append(("wrist Gemini 305", OrbbecRGBD(serial=WRIST_SERIAL), 0.0001))
    for _, cam, _ in cams:
        cam.connect()
    win = "top camera (q / Esc to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    shown = False
    try:
        while True:
            rows = []
            for label, cam, scale in cams:
                f = cam.read()
                if f.ok:
                    rows.append(panel(f, label, cam.fps, scale))
            if rows:
                cv2.imshow(win, np.vstack(rows))
                shown = True
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if shown:
                # Qt 后端：窗口被点 × 关掉后这个调用直接抛错（NULL guiReceiver），而不是返回 -1
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        cv2.destroyAllWindows()
        for _, cam, _ in cams:
            cam.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
