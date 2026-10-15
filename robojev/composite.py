"""复合策略的各个环节。每个数都来自传感器或模型，不掺手填的估计。

    指令 ──> YOLO-World（多种说法出候选框）──> CLIP（哪个框是指令说的那个）
         ──> 俯视单应性（粗定位 xy，拟合自示范）──> 自动挑最近的示范预抓取姿态
         ──> 腕部 RGB-D（深度 + 相机自报内参 + 手眼标定）──> 目标 3D 位置（含高度）
         ──> PlayJev（Snake 渲染选方向）──> 按测得高度下探 ──> 合爪 ──> 夹爪反馈 + CV 判成败

选目标为什么是 CLIP：9/28 离线比过，8 条换了说法的指令，CLIP 8/8、Laya 3~4/8（按选项位置在猜）、
PlayJev 看标框的真实图 2/8。见 bench_select.py。
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

CALIB = Path(__file__).resolve().parent / "calibration"
EPISODES = Path.home() / "atom_episodes"

# 一个目标用几种说法同时查：YOLO-World 对措辞非常敏感。
SYNONYMS = {
    "ping pong ball": ["ping pong ball", "orange ball", "table tennis ball"],
    "blue sleeve": ["blue tape", "blue cylinder", "blue sleeve", "blue cup"],
    "power strip": ["white power strip", "power strip"],
    "water bottle": ["plastic bottle", "water bottle"],
    "yellow tape": ["yellow tape", "yellow tape roll"],
}
# 场上其他东西也要查出来，当 CLIP 的干扰项 —— 只查目标的话，CLIP 就没得选了。
SCENE_PHRASES = sorted({p for v in SYNONYMS.values() for p in v} | {"black cup"})


@dataclass
class TopTarget:
    cand: object
    pixel: tuple
    xy_coarse: np.ndarray
    clip_scores: list
    phrases: list


class Selector:
    """YOLO-World + CLIP。两个模型只加载一次。"""

    def __init__(self):
        from .bench_select import ClipSelect
        from .perception import OpenVocab

        self.det = OpenVocab()
        self.clip = ClipSelect()
        self._group_cache = {}

    def target_group(self, instruction: str) -> str:
        """指令说的是哪一类物体：CLIP 文字编码器，和每组里**每一种说法**比，取组内最大值再 argmax。

        只和组名比时 8 条里错 2 条（「blue cylinder」明明就在蓝套筒组的说法里，却被归到水瓶）。
        """
        if instruction not in self._group_cache:
            keys = list(SYNONYMS)
            texts = [instruction] + [ph for k in keys for ph in [k] + SYNONYMS[k]]
            owner = [k for k in keys for _ in [k] + SYNONYMS[k]]
            with self.clip.torch.no_grad():
                te = self.clip.model.encode_text(self.clip.clip.tokenize(texts).cuda()).float()
                te = te / te.norm(dim=-1, keepdim=True)
                sim = (te[1:] @ te[0]).cpu().numpy()
            best = {k: max(float(sv) for sv, o in zip(sim, owner) if o == k) for k in keys}
            self._group_cache[instruction] = max(best, key=best.get)
        return self._group_cache[instruction]

    def select(self, img, instruction: str, extra_phrases: Sequence[str] = ()):
        """两个独立信号必须一致才算看到：

        1. 检测器用的说法属于目标那一组（`target_group`）。只看图的话，腕部画面里没有球时，
           CLIP 在唯一的候选（墙上一个被叫成 white power strip 的插座）上也「认」了。
        2. 在剩下的候选里 CLIP 看图选最像的，并且要比背景描述更像（`verified`）。
        """
        phrases = sorted(set(SCENE_PHRASES) | set(extra_phrases))
        cands = self.det.detect(img, phrases)
        if not cands:
            return None, cands, []
        group = set(SYNONYMS.get(self.target_group(instruction), []))
        ok = [c for c in cands if c.phrase in group or any(p in group for p, _ in c.extra.get("also", []))]
        if not ok:
            return None, cands, []
        i, s = self.clip(img, ok, instruction)
        if not self.clip.verified(img, ok[i], instruction):
            return None, cands, s.tolist()          # 选出来的也不像：当作没看到
        return ok[i], ok, s.tolist()


def homography():
    d = json.loads((CALIB / "homography.json").read_text())
    return np.array(d["H"], float), d


def pixel_to_xy(H, u, v) -> np.ndarray:
    p = H @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def in_coverage(meta, u, v, margin: float = 0.0) -> bool:
    (u0, v0), (u1, v1) = meta.get("px_range", [[-1e9, -1e9], [1e9, 1e9]])
    return u0 - margin <= u <= u1 + margin and v0 - margin <= v <= v1 + margin


def choose_pregrasp(xy_target, t_offset, back_s: float = 2.0, min_rise: float = 0.04,
                    max_xy_drift: float = 0.04):
    """从真抓起过乒乓球的示范里，挑抓取点离目标最近的那条，取它合爪前 back_s 秒的关节角。

    只用「画面里球确实被带走了」的片段（demo_ball_episodes.json，由拟合单应性时筛出来）；
    grasp1 的任务标签不可信，很多条写着乒乓球、实际抓的是水瓶。
    """
    import cv2

    good = json.loads((CALIB / "demo_ball_episodes.json").read_text())
    best = None
    for r in good:
        d = np.load(EPISODES / r["f"])
        ee, st, fps, i = d["ee_pose"].astype(float), d["state"].astype(float), float(d["fps"]), r["i"]

        def tip(k):
            R, _ = cv2.Rodrigues(ee[k, 3:6])
            return ee[k, :3] + R @ t_offset

        j = i - int(back_s * fps)
        if j < 0:
            continue
        tg, tj = tip(i), tip(j)
        if tj[2] - tg[2] < min_rise or np.linalg.norm(tj[:2] - tg[:2]) > max_xy_drift:
            continue
        dist = float(np.linalg.norm(tg[:2] - np.asarray(xy_target)))
        if best is None or dist < best["dist"]:
            best = dict(q=st[j, :7].tolist(), src=r["f"], frame=j, tip_pre=tj.tolist(),
                        tip_grasp_demo=tg.tolist(), dist=dist)
    return best
