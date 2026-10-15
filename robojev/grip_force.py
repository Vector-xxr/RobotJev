"""抓取力度和抓取位姿（上/中/下）：CLIP 按语义和物体外观判断，在 PlayJev 伺服之前定好。

力度：只由 CLIP（腕部截图 + 物体名提示词）判出的类别决定，合一次爪就抬，不试抬、不加档。
位姿：CLIP 选上/中/下，再按几何检查能不能做到（固定 31° 斜着接近，夹持点上方最多容下「手指长度」
那么高的物体，再高掌心就碰物体顶），做不到换最近的可行位置。指令里不会说怎么抓。

9/29 选 CLIP 之前测过：Laya / PlayJev 对这两个问题的回答跟物体无关（按选项位置选、或永远同一个答案）；
CLIP 按物体名 12 种里力度类 11 种合理（纸杯判成硬）、位置 12 种都合理，但类间差只有 0.002~0.05 ——
所以同时用「物体名文字」和「腕部截图」两路相似度相加。

TG9801 力位模式的「目标夹取力」是 1~100 的相对档位（不是牛顿，手册没给换算）：夹爪一直合到
阻力到这个档位才停。原来固定 50，对鸡蛋这种东西太大。

CLIP 看腕部画面里的物体框、加物体名文字，和「易碎 / 软 / 硬」三组描述比，取相似度最高的一类，
这一类直接给出力度档位（FORCE）和合爪速度（SPEED）。

9/29 用户要求去掉「滑落检测」：原来是从 1 档起合爪、试抬 15mm 看物体跟没跟着走、滑了就放回加一档。
那一版的实测：乒乓球 / 套筒 / 瓶盖 / 托盘里的球最后都停在 1~2 档；鸡蛋 1~4 档都滑（指尖把蛋挤出去，是几何问题不是力）。
FORCE 取值：硬 = 50（加自适应之前一直固定用 50，套筒、球、胶带都这样抓成功过）；软 15、易碎 5 是 9/29 用户指定的。

2026-09-29 空载实测：档位 1~12 都能合到底（读回与设定一致），速度 200 合满行程 5.8s、1000 是 1.6s。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAGILITY = {
    "易碎": ["a photo of a fragile object that breaks easily", "a photo of an eggshell",
             "a photo of thin glassware"],
    "软": ["a photo of a soft squishy object", "a photo of a sponge", "a photo of a soft fruit"],
    "硬": ["a photo of a hard rigid object", "a photo of a hard plastic container", "a photo of a metal tool"],
}
POSE = {
    "上": ["a photo of an object you pick up by its cap or lid", "a photo of a bottle with a cap on top",
           "a photo of a tall container you grab at the top"],
    "中": ["a photo of an object you grab around its middle", "a photo of a small round object", "a photo of a small box"],
    "下": ["a photo of an object you hold near its base", "a photo of a vase with a heavy base",
           "a photo of a wine glass held at the stem"],
}
FORCE = {"易碎": 5, "软": 15, "硬": 50}         # 目标夹取力（TG9801 力位模式的相对档位 1~100）；软、易碎 9/29 用户指定
SPEED = {"易碎": 200, "软": 500, "硬": 1000}     # 合爪速度：易碎的慢，碰上去的冲击小
PRE_GAP_M = 0.012                                # 先快速合到「比物体宽这么多」，碰不到物体；再慢慢合
FINGER_FLOOR_M = 0.012                           # 两指中间最低离支撑面这么高，指尖不碰桌（抓取高度下限、环境下边界都用它）
                                                 # 开跑时按夹爪标定重设（calibrate_gripper.finger_floor：指尖在 G 下方多深 + 2mm）：
                                                 # 旧夹爪 11.7mm -> 12mm；9/30 换的手指长 -> 26mm


@dataclass
class GripPlan:
    kind: str
    force: int
    speed: int
    scores: dict
    pose: str = "中"
    pose_scores: dict = None


def _text_vec(tf, texts):
    clip = tf.clip
    with clip.torch.no_grad():
        e = clip.model.encode_text(clip.clip.tokenize(list(texts)).cuda())
        return (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()


def _classify(tf, img, box, phrase, groups):
    """每类的「图像相似度均值 + 物体名文字相似度均值」。"""
    names, texts = [], []
    for k, ts in groups.items():
        names += [k] * len(ts); texts += ts
    s_img = tf.clip.scores(img, [box], texts)[0]
    s_txt = _text_vec(tf, texts) @ _text_vec(tf, [f"a photo of a {phrase}"])[0]
    out = {}
    for k in groups:
        idx = [i for i, n in enumerate(names) if n == k]
        out[k] = float(np.mean(s_img[idx]) + np.mean(s_txt[idx]))
    return max(out, key=out.get), out


def judge(tf, img, xyxy, phrase: str = "") -> GripPlan:
    """CLIP 看物体框 + 物体名，判易碎/软/硬（定力度上限、速度）和上/中/下（抓取位姿）。"""
    class Box:
        pass
    box = Box(); box.xyxy = np.asarray(xyxy, float)
    kind, ks = _classify(tf, img, box, phrase or "object", FRAGILITY)
    pose, ps = _classify(tf, img, box, phrase or "object", POSE)
    return GripPlan(kind=kind, force=FORCE[kind], speed=SPEED[kind], scores=ks, pose=pose, pose_scores=ps)


def grasp_height(pose: str, h: float, z_top, band: float, floor: float = None):
    """把「上/中/下」换成抓取高度（离支撑面，米），并检查几何上做不做得到。

    做得到 = 夹持点上方的物体高度 h − z ≤ 手指长度 band（否则掌心碰物体顶），且 z ≥ floor（指尖不碰桌）。
    上 = 顶部那一截（逐层宽度规则算出的 z_top；物体矮于 2.5 倍手指长度时没有「顶部一截」可言）；
    中 = 半高；下 = 1/4 高。模型选的做不到就按就近顺序换。返回 (高度, 实际用的位姿, 说明)。
    """
    floor = FINGER_FLOOR_M if floor is None else floor      # 运行时按夹爪标定设（calibrate_gripper.apply_calibration）
    cand = {"上": z_top if (z_top is not None and h >= 2.5 * band) else None,
            "中": 0.5 * h, "下": max(0.25 * h, floor)}
    order = {"上": ("上", "中", "下"), "中": ("中", "上", "下"), "下": ("下", "中", "上")}[pose]
    why = []
    for p in order:
        z = cand[p]
        if z is None:
            why.append(f"{p}：物体不够高，没有顶部一截")
            continue
        if h - z > band + 0.002:
            why.append(f"{p}：上方还有 {(h - z) * 1000:.0f}mm > 手指长度 {band * 1000:.0f}mm，掌心会碰物体顶")
            continue
        if z < floor:
            why.append(f"{p}：离桌面只有 {z * 1000:.0f}mm，指尖会碰桌")
            continue
        return z, p, why
    return 0.5 * h, "中", why + ["都做不到，退回半高"]
