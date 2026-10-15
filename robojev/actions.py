"""动作空间：6 个方向 × 2 个幅度 = 12 个硬编码离散动作。

每个动作最终变成 `AR5Env.step()` 吃的 7 维向量 `[dx,dy,dz,drx,dry,drz,gripper]`，
其中 delta **在基座系**（见 `atom_robot/ar5/transforms.py:compose_pose`，平移是直接相加）。

**为什么只有平移，没有旋转**：这一版任务是「把 A 放到 B 里」，夹爪垂直下抓，
姿态全程不变。把 6 个旋转方向也塞进来会让选项数翻倍，而 Laya 在选项多的时候
本来就更不准（见下），收益却是零。要加旋转的时候，照 `AXES` 的样子往下接就行，
`to_delta()` 已经留了 3 个旋转位。

**为什么幅度分两档**：接近阶段需要走得快（20 cm 的距离用 5 mm 步长要 40 步，
每步都要跑一遍 YOLO + Laya），对准阶段需要走得细（粗步长会在目标附近来回超调）。
一档走不出这两种需求。

**幅度为什么是 12 mm / 5 mm**：有两条上限，卡死的是小的那条。

    ArmConfig.max_step_translation = 0.020   # 单步位移上限，超了被 clamp_delta 截断
    AR5Config.max_lead             = 0.015   # 指令点允许领先实测位姿多少，超了被 _leash 拉回

第一版取了 18 mm，在 `max_step_translation` 之下，看起来没问题 —— 结果 sim 跑一轮
`leashed` 计数 46/50。因为臂在一个控制周期里走不完 18 mm，指令点每步都比实测位姿
领先超过 15 mm，于是每一步都被缰绳拉回。表现是「能到但一直在被限速」，不报错，
只有 `env.stats` 里看得见。12 mm 在 `max_lead` 之下，实测 leash 不再触发。

**改这个值之前先看那两个上限**，它们在 `atom_robot/ar5/config.py`。
细档 5 mm 约等于深度估计本身的噪声量级，再细没有意义。

⚠️ **12 个选项正好踩在 Laya 的标定边界上。** 加载 checkpoint 时它会警告
`invalid temperatures ... choice:11+`，意思是 **11 个及以上选项的 choice 题
温度是非法的、confidence 未标定**。所以默认不把 12 个动作塞进一个问题，而是
拆成「哪个方向」(6 选 1) + 「粗还是细」(2 选 1) 两问，两问都在标定区内。
动作空间仍然是这 12 个，只是问法不同。想对比的话 `--single-question` 能切回
一问 12 选 1，但那条路上的 confidence 不要当真。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# 幅度。单位米。COARSE_M 必须 < AR5Config.max_lead (0.015)，理由见模块文档。
COARSE_M = 0.012
FINE_M = 0.005

# 基座系三个轴的朝向说明。这些话会原样进 prompt 给 Laya 看，
# 所以要用「前/后/左/右/上/下」这种它可能在预训练里见过的词，
# 而不是「+X」——它对坐标轴符号没有任何先验。
AXES = [
    (0, +1, "forward", "away from the robot base, further out over the bench"),
    (0, -1, "backward", "back toward the robot base"),
    (1, +1, "left", "to the robot's left"),
    (1, -1, "right", "to the robot's right"),
    (2, +1, "up", "up, away from the bench surface"),
    (2, -1, "down", "down, toward the bench surface"),
]

SCALES = [("big", COARSE_M), ("small", FINE_M)]


@dataclass(frozen=True)
class Action:
    name: str          # 例如 "left_big"
    axis: int          # 0=x 1=y 2=z
    sign: int          # +1 / -1
    scale: str         # "big" / "small"
    metres: float
    blurb: str         # 给 Laya 的 criteria 文本

    def to_delta(self, gripper: float) -> np.ndarray:
        """变成 env.step() 的 7 维动作。gripper: 0=张开 1=闭合。"""
        a = np.zeros(7, dtype=np.float64)
        a[self.axis] = self.sign * self.metres
        a[6] = float(np.clip(gripper, 0.0, 1.0))
        return a


def _build() -> List[Action]:
    out: List[Action] = []
    for axis, sign, word, where in AXES:
        for scale, metres in SCALES:
            out.append(Action(
                name=f"{word}_{scale}",
                axis=axis, sign=sign, scale=scale, metres=metres,
                blurb=f"move the gripper {word} ({where}) by {metres * 1000:.0f} mm",
            ))
    return out


ACTIONS: List[Action] = _build()
BY_NAME: Dict[str, Action] = {a.name: a for a in ACTIONS}

# 拆成两问时用的两套选项。
DIRECTION_CRITERIA: Dict[str, str] = {
    word: f"move the gripper {word} -- {where}" for _, _, word, where in AXES
}
SCALE_CRITERIA: Dict[str, str] = {
    "big": f"a large {COARSE_M * 1000:.0f} mm step -- the gripper is still far from where it needs to be",
    "small": f"a small {FINE_M * 1000:.0f} mm step -- the gripper is close and needs fine alignment",
}
# 一问 12 选 1 时用的。
COMBINED_CRITERIA: Dict[str, str] = {a.name: a.blurb for a in ACTIONS}


def lookup(direction: str, scale: str) -> Action:
    return BY_NAME[f"{direction}_{scale}"]


def hold(gripper: float) -> np.ndarray:
    """原地不动，只管夹爪。抓取和放开的那一步用这个 —— 一边合爪一边移动
    会把已经对准的位置蹭掉。"""
    a = np.zeros(7, dtype=np.float64)
    a[6] = float(np.clip(gripper, 0.0, 1.0))
    return a


def greedy(offset_xyz: np.ndarray, fine_below_m: float = 0.04) -> Optional[Action]:
    """解析 baseline：挑偏差最大的那个轴，往减小它的方向走一步。

    这不是给机器人用的「好」策略，它是**判据**。没有它就无法回答
    「闭环不动是因为 Laya 选错了，还是因为坐标系/标定/执行链本身是错的」——
    同一套 state 喂给它，如果它能把臂开过去而 Laya 不能，问题在模型；
    如果它也开不过去，问题在它下面那几层。
    """
    offset = np.asarray(offset_xyz, dtype=np.float64).reshape(3)
    axis = int(np.argmax(np.abs(offset)))
    if abs(offset[axis]) < 1e-4:
        return None
    sign = 1 if offset[axis] > 0 else -1
    word = next(w for ax, sg, w, _ in AXES if ax == axis and sg == sign)
    # 离得近就换细档，否则会在目标附近反复超调。
    scale = "small" if float(np.linalg.norm(offset)) < fine_below_m else "big"
    return lookup(word, scale)
