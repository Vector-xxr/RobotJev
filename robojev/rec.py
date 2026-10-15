"""抓取过程录像：每 0.25s 存一帧「俯视 | 腕部」拼图，文件名带当前阶段/步号和时间，用来事后逐帧对照日志。

只从相机取最新帧（相机有自己的抽帧线程），不碰臂的接口，不影响控制。

    from . import rec
    rec.tag("servo12")        # 在循环里标出当前步
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

_TAG = ["start"]


def tag(s: str) -> None:
    _TAG[0] = s


class Recorder:
    def __init__(self, out: Path, cams, period: float = 0.25):
        self.out, self.cams, self.period = Path(out), cams, period
        self.out.mkdir(parents=True, exist_ok=True)
        self._halt = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.n = 0

    def _run(self):
        import cv2

        t0 = time.time()
        while not self._halt.wait(self.period):
            imgs = []
            for c in self.cams:
                try:
                    f = c.read()
                    if f.ok:
                        imgs.append(cv2.resize(f.color, (480, 360)))
                except Exception:
                    pass
            if not imgs:
                continue
            frame = np.hstack(imgs)
            cv2.putText(frame, f"{_TAG[0]}  t={time.time() - t0:.1f}s", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(self.out / f"{self.n:04d}_{_TAG[0]}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.n += 1

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._t.join(timeout=2.0)
