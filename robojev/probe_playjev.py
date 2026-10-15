#!/usr/bin/env python3
"""PlayJev 能不能直接看俯视图出动作？上臂之前先判这个。

    python -m robojev.probe_playjev                    # 用俯视相机实时抓帧
    python -m robojev.probe_playjev --image foo.png    # 用一张存好的图

**这个脚本要回答的唯一问题：模型的输出到底依不依赖这张图。**

一个在十个 2D 浏览器游戏上训出来的 0.8B，看机械臂工作台是彻底的分布外。
它一定会**返回**一个分布 —— 但返回分布不等于在决策。分不清这两者就上臂，
表现会是「臂朝一个固定方向稳定地走」，而且看起来很像坐标系错了。

判据是三个对照，任何一个不过就别往下走：

  1. 左右镜像。把图水平翻转，left 和 right 的概率应该**互换**。
     不换 -> 模型没在看图里的空间关系。
  2. 纯色空白图。给一张什么都没有的图，得到的就是它的**先验**。
     真实图的分布和先验几乎一样 -> 图没起作用。
  3. 遮掉目标。把蓝色目标涂掉，分布应该明显变化。
     不变 -> 它看的不是目标。

镜像那一条最硬：它不需要知道「正确答案」是什么，只需要模型自洽。
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from typing import Dict, List

import numpy as np

SERVER = "http://127.0.0.1:18732/v1/systemone"
# 本机服务**不走代理**。.bashrc 设了 http_proxy=127.0.0.1:7897，urllib 默认连 127.0.0.1 也走它：
# 服务挂了的时候，代理替它回一个 502 Bad Gateway，而不是「连接被拒」—— 真实原因被遮住了。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 和 actions.py 的 6 个方向同名，方便直接接进闭环。
# 描述里刻意不提坐标轴，只说画面里的方向 —— 模型看的是图。
MOVES: Dict[str, str] = {
    "left": "move the gripper to the left",
    "right": "move the gripper to the right",
    "forward": "move the gripper away from the viewer, toward the top of the image",
    "backward": "move the gripper toward the viewer, toward the bottom of the image",
    "up": "lift the gripper up, away from the table",
    "down": "lower the gripper down, toward the table",
}


def ask(img_bgr: np.ndarray, instructions: str, moves: Dict[str, str] = None,
        timeout: float = 60.0) -> dict:
    import cv2

    moves = moves or MOVES
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    b64 = base64.b64encode(buf.tobytes()).decode()
    body = {
        "state": {"frames": ["data:image/jpeg;base64," + b64]},
        "questions": {"move": {"type": "choice", "instructions": instructions,
                               "criteria": dict(moves)}},
    }
    req = urllib.request.Request(SERVER, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(_OPENER.open(req, timeout=timeout))
    except urllib.error.URLError as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raise
        raise ConnectionError(f"连不上 PlayJev 服务 {SERVER}：{exc.reason}。先起服务："
                              "cd 算法/Jev/RobotJev/PlayJev && python -m playjev.serve "
                              "--ckpt weights/PlayJev-0.8B --port 18732") from exc


def probs_of(res: dict) -> Dict[str, float]:
    return res["answers"]["move"]["probabilities"]


def show(tag: str, p: Dict[str, float], choice: str) -> None:
    order = list(MOVES)
    bar = "  ".join(f"{k}={p.get(k, 0):.3f}" for k in order)
    print(f"  {tag:22s} -> {choice:9s} | {bar}")


def l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in MOVES)


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="", help="用存好的图，不给就开俯视相机")
    ap.add_argument("--instruction", default="Pick up the blue cylinder on the table. "
                                             "Which way should the gripper move next?")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"读不了 {args.image}")
            return 2
    else:
        from .camera import TopCamera
        with TopCamera() as cam:
            time.sleep(1.5)
            img = cam.read().color.copy()

    print(f"图 {img.shape}\n指令: {args.instruction}\n")

    # --- 1) 原图 ---
    t0 = time.perf_counter()
    r = ask(img, args.instruction)
    dt = (time.perf_counter() - t0) * 1000
    p_real = probs_of(r)
    show("原图", p_real, r["answers"]["move"]["choice"])
    print(f"  ({dt:.0f} ms, forward {r['timing']['forward_ms']:.0f} ms, "
          f"visual_tokens {r['timing']['visual_tokens']})")

    # --- 2) 左右镜像 ---
    r = ask(cv2.flip(img, 1), args.instruction)
    p_mir = probs_of(r)
    show("左右镜像", p_mir, r["answers"]["move"]["choice"])

    # --- 3) 空白图（先验） ---
    blank = np.full_like(img, 128)
    r = ask(blank, args.instruction)
    p_blank = probs_of(r)
    show("纯灰空白（先验）", p_blank, r["answers"]["move"]["choice"])

    # --- 4) 遮掉蓝色目标 ---
    from .color_detector import ColorDetector
    masked = img.copy()
    dets = ColorDetector("blue").detect(img)
    if dets:
        x0, y0, x1, y1 = np.round(dets[0].xyxy).astype(int)
        pad = 12
        masked[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 128
        r = ask(masked, args.instruction)
        p_mask = probs_of(r)
        show("遮掉蓝色目标", p_mask, r["answers"]["move"]["choice"])
    else:
        p_mask = None
        print("  (图里没找到蓝色目标，跳过遮挡对照)")

    # ---------------- 判据 ----------------
    print("\n" + "=" * 72)
    swap = abs(p_real["left"] - p_mir["right"]) + abs(p_real["right"] - p_mir["left"])
    keep = abs(p_real["left"] - p_mir["left"]) + abs(p_real["right"] - p_mir["right"])
    print(f"镜像检验：  |real.L-mir.R|+|real.R-mir.L| = {swap:.4f}   (交换了就该小)")
    print(f"            |real.L-mir.L|+|real.R-mir.R| = {keep:.4f}   (没换就该小)")
    mirror_ok = swap < keep * 0.5
    print(f"            -> {'左右确实跟着图换了' if mirror_ok else '**左右没跟着图换：模型没在读图里的空间关系**'}")

    d_blank = l1(p_real, p_blank)
    print(f"与空白先验的 L1 距离: {d_blank:.4f}   (接近 0 = 图没起作用)")
    if p_mask is not None:
        print(f"遮掉目标后的 L1 变化: {l1(p_real, p_mask):.4f}   (接近 0 = 它看的不是目标)")

    verdict = mirror_ok and d_blank > 0.15
    print("\n结论：" + ("可以继续往闭环接。" if verdict else
                      "**别接闭环。** 这个 checkpoint 对俯视工作台没有有效的空间响应，"
                      "接上去臂只会朝一个固定方向走。"))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
