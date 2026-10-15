"""特权信息：把感知结果 + 本体状态整理成 Laya 吃的 state，以及要问它的问题。

**为什么要「口语化」而不是直接给数字。**
第一版探针给的是 `Offsets: dx=+0.180 m, dy=0.000 m, dz=0.000 m`，Laya 在 8 道
方向题上只对 1 道。它是个在客服工单上训出来的 BERT 分类头，`dx=-0.180` 对它
就是一串字符，符号和大小都没有意义。所以这里把偏差**翻译成话**：

    "The apple is 18 centimetres to the LEFT of the gripper."

同一件事，但「LEFT」这个词在预训练语料里出现过千万次，而 `dy=+0.180` 没有。
这是让它有机会答对的唯一办法。改成功与否要用 `--policy geometric` 对照着量，
不要凭感觉。

state 里同时保留了数字（`numeric` 段），是给日志和微调数据用的 —— 以后拿这些
轨迹去微调 Laya 时，数字才是可学的特征。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .actions import COMBINED_CRITERIA, DIRECTION_CRITERIA, SCALE_CRITERIA
from .parser import Task

# 分阶段。闭环靠这个决定该问什么、该往哪个目标走。
PHASE_APPROACH = "approach_target"
PHASE_GRASP = "grasp"
PHASE_LIFT = "lift"
PHASE_CARRY = "carry_to_container"
PHASE_RELEASE = "release"
PHASE_DONE = "done"


def _cm(x: float) -> str:
    return f"{abs(x) * 100:.1f}"


def verbalize_offset(offset: np.ndarray) -> str:
    """把 (dx,dy,dz) 说成人话。offset = 目标 - 夹爪，基座系，米。

    只说「有意义」的轴：小于 1 cm 的分量说成 aligned，否则一句一个方向。
    全说出来（包括 0.2 cm 这种）会让句子里塞满噪声，反而淹掉真正大的那一项。
    """
    dx, dy, dz = float(offset[0]), float(offset[1]), float(offset[2])
    bits: List[str] = []
    if abs(dx) >= 0.01:
        bits.append(f"{_cm(dx)} cm {'FORWARD of' if dx > 0 else 'BACKWARD from'} the gripper")
    if abs(dy) >= 0.01:
        bits.append(f"{_cm(dy)} cm to the {'LEFT' if dy > 0 else 'RIGHT'} of the gripper")
    if abs(dz) >= 0.01:
        bits.append(f"{_cm(dz)} cm {'ABOVE' if dz > 0 else 'BELOW'} the gripper")
    if not bits:
        return "exactly at the gripper (aligned within 1 cm on every axis)"
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + ", and " + bits[-1]


@dataclass
class Percept:
    """一帧看到的东西。定不出 3D 位置的用 None，别拿 0 冒充。"""

    seen: bool = False
    cls_name: str = ""
    conf: float = 0.0
    xyz_base: Optional[np.ndarray] = None
    pixel: Optional[tuple] = None


@dataclass
class WorldState:
    task: Task
    phase: str
    ee_xyz: np.ndarray                      # 夹爪当前位置，基座系
    gripper: float                          # 0=张开 1=闭合
    holding: bool                           # 是否认为已经抓住了
    target: Percept = field(default_factory=Percept)
    container: Percept = field(default_factory=Percept)
    step: int = 0

    # ----------------------------------------------------------- 派生量

    @property
    def goal(self) -> Optional[Percept]:
        """当前该往哪个东西走。抓之前是目标物，抓起来之后是容器。"""
        if self.phase in (PHASE_APPROACH, PHASE_GRASP):
            return self.target
        if self.phase in (PHASE_CARRY, PHASE_RELEASE):
            return self.container
        return None

    @property
    def offset(self) -> Optional[np.ndarray]:
        g = self.goal
        if g is None or not g.seen or g.xyz_base is None:
            return None
        return g.xyz_base - self.ee_xyz

    @property
    def distance(self) -> Optional[float]:
        o = self.offset
        return None if o is None else float(np.linalg.norm(o))

    # -------------------------------------------------- 给 Laya 的 state

    def to_laya_state(self) -> Dict[str, Any]:
        t = self.task
        lines: List[str] = [
            f"Robot task: {t.describe()}.",
            f"Stage: {self.phase.replace('_', ' ')}.",
            f"The gripper is currently {'CLOSED and holding something' if self.holding else 'OPEN and empty'}.",
        ]

        if not self.target.seen:
            lines.append(f"The {t.target} is NOT visible in the camera right now.")
        else:
            lines.append(f"The {t.target} is visible (detector confidence {self.target.conf:.2f}).")
        if t.needs_place:
            if not self.container.seen:
                lines.append(f"The {t.container} is NOT visible in the camera right now.")
            else:
                lines.append(f"The {t.container} is visible (detector confidence {self.container.conf:.2f}).")

        goal = self.goal
        goal_name = (t.target if self.phase in (PHASE_APPROACH, PHASE_GRASP) else t.container)
        off = self.offset
        if off is None:
            lines.append(f"The position of the {goal_name} relative to the gripper is unknown.")
        else:
            lines.append(f"The {goal_name} is {verbalize_offset(off)}.")
            lines.append(f"Straight-line distance from the gripper to the {goal_name}: "
                         f"{self.distance * 100:.1f} cm.")

        state = {"body": " ".join(lines)}
        # 数字留着：日志要对账，以后微调要拿它当特征。Laya 读的是 body。
        state["numeric"] = {
            "ee_xyz": [round(float(x), 4) for x in self.ee_xyz],
            "offset": None if off is None else [round(float(x), 4) for x in off],
            "distance": None if self.distance is None else round(self.distance, 4),
            "gripper": round(float(self.gripper), 3),
            "holding": self.holding,
            "phase": self.phase,
            "step": self.step,
        }
        return state


def questions_for(phase: str, task: Task, single_question: bool = False) -> Dict[str, Any]:
    """这一步要问 Laya 什么。按阶段给不同的问题，别每次都问一整套。

    每个问题都要花一次前向（同一批里是并行的，但选项文本会挤占同一个 8192
    的上下文），而且问了不用的答案只会污染日志。
    """
    goal_name = task.target if phase in (PHASE_APPROACH, PHASE_GRASP) else task.container

    q: Dict[str, Any] = {}

    if single_question:
        # 一问 12 选 1。⚠️ 12 >= 11，这个 checkpoint 在这个选项数上的
        # temperature 是非法的，confidence 未标定 —— 只能当对照，别上生产。
        q["move"] = {
            "type": "choice",
            "instructions": f"Which single move brings the gripper closer to the {goal_name}?",
            "criteria": dict(COMBINED_CRITERIA),
        }
    else:
        q["direction"] = {
            "type": "choice",
            "instructions": f"Which direction should the gripper move to get closer to the {goal_name}?",
            "criteria": dict(DIRECTION_CRITERIA),
        }
        q["scale"] = {
            "type": "choice",
            "instructions": "How big should this step be?",
            "criteria": dict(SCALE_CRITERIA),
        }

    if phase in (PHASE_APPROACH, PHASE_GRASP):
        q["grasp_now"] = {
            "type": "noul",
            "instructions": (f"Is the gripper already aligned with the {task.target} closely "
                             f"enough to close the gripper and grasp it right now?"),
        }
    if phase in (PHASE_CARRY, PHASE_RELEASE) and task.needs_place:
        q["release_now"] = {
            "type": "noul",
            "instructions": (f"Is the held object already positioned over the {task.container} "
                             f"closely enough to open the gripper and let it drop in?"),
        }
    return q
