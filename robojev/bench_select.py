#!/usr/bin/env python3
"""离线比较「选目标」该交给谁：Laya / PlayJev（看标了字母框的真实图）/ CLIP（图文相似度）。

    python -m robojev.bench_select --image scene.png

候选一律来自 YOLO-World（多种说法、按重叠合并）。判据：8 条换了说法的指令，选中的候选对不对。

9/28 第一轮：Laya 3~4/8，而且是按选项位置在猜（「pick up the power strip」选项里就写着
white power strip，它还是选了蓝色套筒）。它是文本决策模型，「这句话指的是哪个实物」超出了它的能力。
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

TESTS = [("pick up the ping pong ball", "ball"), ("grab the table tennis ball", "ball"),
         ("pick up the orange thing", "ball"), ("pick up the blue cylinder", "blue"),
         ("grab the blue sleeve", "blue"), ("pick up the power strip", "power"),
         ("take the water bottle", "bottle"), ("pick up the yellow tape", "yellow")]
PHRASES = ["ping pong ball", "orange ball", "table tennis ball", "blue tape", "yellow tape",
           "white power strip", "plastic bottle", "black cup"]


def truth_index(cands, key):
    for i, c in enumerate(cands):
        if key in c.phrase:
            return i
    return None


def draw_boxes(img, cands):
    import cv2

    vis = img.copy()
    for i, c in enumerate(cands):
        x0, y0, x1, y1 = [int(v) for v in c.xyxy]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, chr(65 + i), (x0 + 3, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def playjev_select(img, cands, instruction):
    from .probe_playjev import ask

    vis = draw_boxes(img, cands)
    crit = {f"box {chr(65 + i)}": f"the object inside the red box labelled {chr(65 + i)}"
            for i in range(len(cands))}
    r = ask(vis, f"The robot was told: \"{instruction}\". Which labelled box contains that object?", crit)
    probs = r["answers"]["move"]["probabilities"]
    p = np.array([probs[f"box {chr(65 + i)}"] for i in range(len(cands))])
    return int(np.argmax(p)), p


class ClipSelect:
    def __init__(self):
        import clip
        import torch

        self.torch = torch
        import os
        weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "clip")
        self.model, self.pre = clip.load("ViT-B/32", device="cuda", download_root=weights)  # 用项目里的权重，不去 ~/.cache/clip
        self.clip = clip

    NEGATIVES = ["a photo of the empty table surface", "a photo of a black robot gripper finger",
                 "a photo of a white wall", "a photo of a cable"]

    def scores(self, img, cands, texts):
        """每个候选框 × 每句文字的 CLIP 相似度，形状 (len(cands), len(texts))。"""
        from PIL import Image

        crops = []
        for c in cands:
            x0, y0, x1, y1 = [int(v) for v in c.xyxy]
            pad = 6
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad, ::-1]
            crops.append(self.pre(Image.fromarray(np.ascontiguousarray(crop))))
        with self.torch.no_grad():
            ie = self.model.encode_image(self.torch.stack(crops).cuda())
            te = self.model.encode_text(self.clip.tokenize(list(texts)).cuda())
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            return (ie @ te.T).float().cpu().numpy()

    def verified(self, img, cand, instruction) -> bool:
        """CLIP 永远会选出一个；这里判断选出来的那个是不是真的是指令说的东西：
        和指令的相似度要高过每一句「背景」描述。第一次在腕部画面里没有球，它选中了
        手指旁边的东西，深度取到手指上，报出一个离夹持中心只有 3cm 的假目标。"""
        s = self.scores(img, [cand], [f"a photo of the object in: {instruction}"] + self.NEGATIVES)[0]
        return bool(s[0] > s[1:].max())

    def __call__(self, img, cands, instruction):
        from PIL import Image

        crops = []
        for c in cands:
            x0, y0, x1, y1 = [int(v) for v in c.xyxy]
            pad = 6
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad, ::-1]
            crops.append(self.pre(Image.fromarray(np.ascontiguousarray(crop))))
        with self.torch.no_grad():
            ie = self.model.encode_image(self.torch.stack(crops).cuda())
            te = self.model.encode_text(self.clip.tokenize([f"a photo of the object in: {instruction}"]).cuda())
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            s = (ie @ te.T).squeeze(1).float().cpu().numpy()
        return int(np.argmax(s)), s


def main() -> int:
    import cv2

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    img = cv2.imread(a.image)

    from .perception import OpenVocab
    cands = OpenVocab().detect(img, PHRASES)
    print("候选:", [f"{chr(65 + i)}={c.phrase}" for i, c in enumerate(cands)])

    judges = {"PlayJev": playjev_select}
    try:
        judges["CLIP"] = ClipSelect()
    except Exception as exc:                                       # noqa: BLE001
        print("CLIP 起不来:", exc)
    for name, fn in judges.items():
        ok, t0 = 0, time.perf_counter()
        for ins, key in TESTS:
            want = truth_index(cands, key)
            i, _ = fn(img, cands, ins)
            ok += i == want
            print(f"  [{name}] {ins:30s} -> {chr(65 + i)}={cands[i].phrase:18s} "
                  f"{'✓' if i == want else '✗ 应为 ' + (cands[want].phrase if want is not None else '?')}")
        print(f"{name}: {ok}/{len(TESTS)}  {(time.perf_counter() - t0) / len(TESTS) * 1000:.0f} ms/条\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
