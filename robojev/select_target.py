"""复合策略的「选目标」：YOLO-World 给候选，Laya 判断哪个是指令要的那个。

分工的理由（都是实测）：
- YOLO-World 能框出东西，但**对措辞极敏感、置信度偏低**，同一个物体换个说法就检不出；
  拿最高分当答案，遇到「给了多个物体、指令又用了别的说法」就会错。
- Laya 是文字上的多选题模型，擅长「这句描述和这条指令是不是一回事」这种判断；
  但它**读不懂数字**（RoboJev 实测：距离从 30cm 变到 1cm，输出不动）。
  所以候选描述里只写**词**：颜色、在桌面哪一片、大小，外加检测器用的是哪个说法。坐标和置信度不写进去。

输出里同时给出 Laya 的选择和 YOLO 最高分的选择，两者不一致时由调用方决定（默认不动臂）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .perception import Candidate, colour_name


def region_words(u: float, v: float, w: int = 640, h: int = 480) -> str:
    horiz = "left" if u < w / 3 else ("right" if u > 2 * w / 3 else "middle")
    vert = "far" if v < h / 3 else ("near" if v > 2 * h / 3 else "centre")
    return {"far": "the far part", "centre": "the centre", "near": "the near part"}[vert] + \
        (" of the table" if horiz == "middle" else f", on the {horiz} side, of the table")


def size_word(area: float) -> str:
    return "small" if area < 3000 else ("medium-sized" if area < 12000 else "large")


def describe(c: Candidate) -> str:
    u, v = c.centre
    col = colour_name(c.hue, c.sat, c.val)
    return (f"a {size_word(c.area)} {col} object in {region_words(u, v)}; "
            f"the detector described it as '{c.phrase}'")


def option_name(c: Candidate) -> str:
    """选项名要**带信息**。第一版用 object_0/object_1，Laya 只能靠位置选，8 条里 4 条全选了第一个。"""
    col = colour_name(c.hue, c.sat, c.val)
    ph = c.phrase
    return ph if col in ph or col in ("grey", "unknown") else f"{col} {ph}"


def laya_select(agent, instruction: str, cands: List[Candidate], n_perm: int = 6,
                seed: int = 0) -> Tuple[Optional[int], Dict]:
    """返回 (选中的下标, 汇总信息)。

    选项顺序打乱 n_perm 次，对每个候选的概率取平均 —— 抵消模型的位置偏好。
    """
    import numpy as np

    if not cands:
        return None, {}
    names = []
    for c in cands:                                   # 名字要唯一
        n = option_name(c)
        while n in names:
            n += " (another one)"
        names.append(n)
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(cands))
    for _ in range(n_perm):
        order = rng.permutation(len(cands))
        criteria = {names[i]: describe(cands[i]) for i in order}
        q = {"target": {"type": "choice",
                        "instructions": f"The robot was told: \"{instruction}\". "
                                        f"Which object on the table should it pick up?",
                        "criteria": criteria}}
        state = {"body": "Objects detected on the table: " +
                 " ".join(f"({k}) {v}." for k, v in criteria.items())}
        probs = agent.predict(state, q)["answers"]["target"]["probabilities"]
        for i in range(len(cands)):
            acc[i] += probs.get(names[i], 0.0)
    acc /= n_perm
    best = int(np.argmax(acc))
    srt = np.sort(acc)[::-1]
    return best, {"probs": {names[i]: round(float(acc[i]), 3) for i in range(len(cands))},
                  "margin": float(srt[0] - (srt[1] if len(srt) > 1 else 0.0)), "names": names}
