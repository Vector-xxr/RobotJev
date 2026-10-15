#!/usr/bin/env python3
"""从示范数据拟合「俯视像素 -> 指尖 xy」单应性，不动臂、不手工标点。

    python -m robojev.fit_demo_homography

输出：
  calibration/homography.json         H、指尖相对法兰偏移 t、误差、像素覆盖范围
  calibration/demo_ball_episodes.json  真正抓起了乒乓球的示范片段（choose_pregrasp 用）

步骤（每一步的必要性都是踩出来的）：
1. pingpong + grasp1 按 ee_pose 去重（grasp1 里有一部分就是 pingpong 的拷贝）。
2. 合爪时刻 = **最后一次**「张开->合上」，且只合过一次；合爪后 1s 法兰抬升 >2cm。
3. **按画面判断抓的是不是球**：合爪后 1.5s 球离开了抓前位置（或被手指挡住看不见）。
   task 字段不可信 —— grasp1 里很多条写着 "pick up the pingpong ball"，实际抓的是水瓶。
   不做这步，像素和抓取位置的相关系数只有 +0.26/+0.47，误差中位 93mm。
4. 指尖 = 法兰 + R_flange @ t，t（3 个数）和 H 一起拟合（RANSAC + 随机搜索）。
   夹爪斜 50~60° 装着，指尖比法兰往前伸十几厘米，接近方向一变法兰 xy 就差很多。
像素用的是**合爪前 0.5s 球心**（此时臂还没挡住目标）。球都放在蓝色套筒上，所以这个单应性
对应的是「套筒上的球」那个高度（指尖 z≈0.211）；别的高度有视差，要靠腕部深度相机补。
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np

EP = Path.home() / "atom_episodes"
CALIB = Path(__file__).resolve().parent / "calibration"


def last_single_close(g):
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    if hi - lo < 0.2:
        return None
    closed = g > (lo + hi) / 2
    edges = np.nonzero(~closed[:-1] & closed[1:])[0] + 1
    return int(edges[-1]) if len(edges) == 1 else None


def main() -> int:
    import cv2

    from .color_detector import ColorDetector

    orange = ColorDetector("orange", min_area_px=150)
    seen, rows = set(), []
    for task in ("pingpong", "grasp1"):
        for f in sorted(glob.glob(str(EP / task / "*.npz"))):
            d = np.load(f)
            h = hashlib.md5(d["ee_pose"].tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            g, ee, fps = d["state"][:, 7].astype(float), d["ee_pose"].astype(float), float(d["fps"])
            i = last_single_close(g)
            if i is None or ee[min(i + int(fps), len(ee) - 1), 2] - ee[i, 2] < 0.02:
                continue
            cap = cv2.VideoCapture(f[:-4] + "_cam_high.mp4")

            def ball(k):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(k, len(ee) - 1)))
                ok, fr = cap.read()
                if not ok:
                    return None, False
                o = orange.detect(fr)
                return (o[0].centre if o else None), True

            before, _ = ball(i - int(0.5 * fps))
            if before is None:
                continue
            after, ok = ball(i + int(1.5 * fps))
            moved = after is None or np.hypot(after[0] - before[0], after[1] - before[1]) > 25
            if not moved:
                continue
            R, _ = cv2.Rodrigues(ee[i, 3:6])
            rows.append(dict(f=str(Path(f).relative_to(EP)), i=i, px=list(before),
                             flange=ee[i, :3].tolist(), R=R.tolist()))
    print(f"真正抓起了乒乓球的示范：{len(rows)} 条")

    F = np.array([r["flange"] for r in rows])
    Rs = np.array([r["R"] for r in rows])
    px = np.array([r["px"] for r in rows], float)

    def fit(t):
        tip = F + np.einsum("nij,j->ni", Rs, t)
        H, m = cv2.findHomography(px, tip[:, :2], cv2.RANSAC, 0.02)
        if H is None:
            return 1e9, None, None, tip
        m = m.ravel().astype(bool)
        e = np.linalg.norm(cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2) - tip[:, :2], axis=1)
        return float(np.median(e[m])) + 0.02 * (len(px) - m.sum()) / len(px), H, m, tip

    rng = np.random.default_rng(0)
    best_s, best_t, scale = fit(np.zeros(3))[0], np.zeros(3), 0.15
    for it in range(600):
        t = best_t + rng.normal(0, scale, 3)
        s = fit(t)[0]
        if s < best_s:
            best_s, best_t = s, t
        if it % 100 == 99:
            scale *= 0.5
    _, H, m, tip = fit(best_t)
    e = np.linalg.norm(cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2) - tip[:, :2], axis=1) * 1000
    loo = []
    for k in np.nonzero(m)[0]:
        mm = m.copy(); mm[k] = False
        Hk, _ = cv2.findHomography(px[mm], tip[mm, :2], 0)
        loo.append(np.linalg.norm(cv2.perspectiveTransform(px[k:k + 1].reshape(-1, 1, 2), Hk).reshape(2)
                                  - tip[k, :2]) * 1000)
    loo = np.array(loo)
    tz = tip[m, 2]
    out = {"H": H.tolist(), "tip_offset_flange": best_t.tolist(), "n_inliers": int(m.sum()), "n": len(rows),
           "err_mm_median": float(np.median(e[m])), "loo_mm_median": float(np.median(loo)),
           "loo_mm_p90": float(np.percentile(loo, 90)), "tip_z_grasp_median": float(np.median(tz)),
           "px_range": [px[m].min(0).tolist(), px[m].max(0).tolist()],
           "maps": "俯视像素(乒乓球中心，球在蓝色套筒上) -> 指尖 xy（基座系）",
           "source": "fit_demo_homography.py"}
    CALIB.mkdir(parents=True, exist_ok=True)
    (CALIB / "homography.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    keep = [{"f": r["f"], "i": r["i"]} for r, ok in zip(rows, m) if ok]
    (CALIB / "demo_ball_episodes.json").write_text(json.dumps(keep, indent=2))
    print(f"t = {np.round(best_t * 1000).astype(int).tolist()} mm   内点 {m.sum()}/{len(m)}")
    print(f"拟合误差中位 {np.median(e[m]):.1f}mm  留一法中位 {np.median(loo):.1f}mm p90 {np.percentile(loo, 90):.1f}mm")
    print(f"抓取时指尖 z 中位 {np.median(tz):.3f}   像素覆盖 {np.round(out['px_range']).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
