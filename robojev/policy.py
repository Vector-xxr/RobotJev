"""决策层：把 WorldState 变成一个动作。

两个实现，同一个接口：

    LayaPolicy       —— 问 Laya，从 12 个硬编码动作里选。这是目标形态。
    GeometricPolicy  —— 直接算：偏差最大的轴，往减小它的方向走。

`GeometricPolicy` 不是备胎，是**量尺**。闭环不动的时候，唯一能快速分清
「Laya 选错了」和「坐标系/标定/执行链错了」的办法，就是同一套 state 换它跑一遍：
它能开过去 -> 问题在模型；它也开不过去 -> 问题在它下面。
每一步两个都算、只执行一个，差异记进日志（`agreed` 字段），这样跑一轮下来
就有了 Laya 对解析解的逐步符合率，不用另外做评测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np

from . import actions as A
from .state import (PHASE_APPROACH, PHASE_CARRY, PHASE_GRASP, PHASE_RELEASE,
                    WorldState, questions_for)

logger = logging.getLogger(__name__)

# 夹爪认为「已对准可以合爪」的距离。深度估计本身的噪声就有几毫米，
# 定得比这更小会导致永远够不着、原地抖。
GRASP_RADIUS_M = 0.025
RELEASE_RADIUS_M = 0.040
# 距离超过这个值用粗档，否则用细档。
COARSE_ABOVE_M = 0.04
# noul 门限。只在 `--trust-laya-gates` 时才用得上，见下面的实测。
GATE = 0.60

# ---------------------------------------------------------------------------
# 2026-09-23 在本机实测（robojev/bench_policy.py，24 道方向题 + 距离扫描）：
#
#   方向（6 选 1）   79.2% 与解析解一致    随机基线 16.7%   -> **能用**
#   幅度（2 选 1）    0.0% 一致            随机基线 50%     -> 不能用
#   grasp_now(noul)  与距离无关                             -> 不能用
#
# 幅度和门控为什么不能用，扫一遍距离就看得很清楚 —— 它们是**常数**：
#
#     距离      30cm   20cm   12cm   8cm    5cm    3cm    1cm
#     P(big)   0.212  0.232  0.221  0.191  0.114  0.200  0.204
#     grasp    0.655  0.540  0.644  0.638  0.625  0.656  0.646
#
# 距离变了 30 倍，两个输出都没动。这跟 Laya 的形态是一致的：它是个非自回归的
# 分类头，没有数值推理能力，「18 cm 比 5 cm 大」这件事它读不出来。而方向能做到
# 79%，是因为 LEFT / FORWARD / ABOVE 这些词在预训练语料里有极强的信号。
#
# 所以默认分工是：**方向问 Laya，幅度和抓放门控按距离硬判**。
# 动作空间仍然是那 12 个（6 方向 × 2 幅度），Laya 选的是其中的方向那一维。
# 想看纯 Laya 的表现，`--trust-laya-scale` / `--trust-laya-gates` 可以放开，
# 但上面的数字说明那条路不会收敛。
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    action: Optional[A.Action]          # None = 不移动（抓/放/结束那一步）
    gripper: float
    kind: str                           # "move" / "grasp" / "release" / "idle"
    why: str
    confidence: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_env_action(self) -> np.ndarray:
        if self.action is None:
            return A.hold(self.gripper)
        return self.action.to_delta(self.gripper)


class Policy(Protocol):
    name: str

    def decide(self, ws: WorldState) -> Decision: ...


class GeometricPolicy:
    """闭式解。没有模型，不会错，用来给流水线定基准。"""

    name = "geometric"

    def decide(self, ws: WorldState) -> Decision:
        off = ws.offset
        if off is None:
            return Decision(None, ws.gripper, "idle", "目标不可见，原地等")

        dist = float(np.linalg.norm(off))

        if ws.phase in (PHASE_APPROACH, PHASE_GRASP) and not ws.holding:
            if dist <= GRASP_RADIUS_M:
                return Decision(None, 1.0, "grasp",
                                f"距目标 {dist * 100:.1f} cm <= {GRASP_RADIUS_M * 100:.0f} cm，合爪")
        if ws.phase in (PHASE_CARRY, PHASE_RELEASE) and ws.holding:
            if dist <= RELEASE_RADIUS_M:
                return Decision(None, 0.0, "release",
                                f"距容器 {dist * 100:.1f} cm <= {RELEASE_RADIUS_M * 100:.0f} cm，松爪")

        act = A.greedy(off)
        if act is None:
            return Decision(None, ws.gripper, "idle", "偏差已在数值噪声内")
        return Decision(act, ws.gripper, "move",
                        f"最大偏差轴 {'xyz'[act.axis]}={off[act.axis] * 100:+.1f} cm -> {act.name}",
                        confidence=1.0)


def laya_model() -> str:
    """Laya 权重：优先用项目里的 weights/laya（普通文件、单一版本，拷走工作区就能断网用），没有才退回 HuggingFace。"""
    import os
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "laya")
    return local if os.path.isdir(local) else "convaiinnovations/laya"


class LayaPolicy:
    """问 Laya。"""

    name = "laya"

    def __init__(self, agent=None, single_question: bool = False, gate: float = GATE,
                 trust_laya_scale: bool = False, trust_laya_gates: bool = False):
        if agent is None:
            import laya
            agent = laya.load(laya_model())
        self.agent = agent
        self.single_question = single_question
        self.gate = gate
        self.trust_laya_scale = trust_laya_scale
        self.trust_laya_gates = trust_laya_gates

    def decide(self, ws: WorldState) -> Decision:
        off = ws.offset
        if off is None:
            return Decision(None, ws.gripper, "idle", "目标不可见，不问模型")
        dist = float(np.linalg.norm(off))

        # --- 抓 / 放的门控：默认按距离硬判 ---
        # Laya 的 grasp_now 与距离无关（见文件头的实测表），交给它会在任意距离
        # 上随机触发，臂会在半空中合爪。
        if not self.trust_laya_gates:
            if ws.phase in (PHASE_APPROACH, PHASE_GRASP) and not ws.holding \
                    and dist <= GRASP_RADIUS_M:
                return Decision(None, 1.0, "grasp",
                                f"距目标 {dist * 100:.1f} cm <= {GRASP_RADIUS_M * 100:.0f} cm（距离判据）")
            if ws.phase in (PHASE_CARRY, PHASE_RELEASE) and ws.holding \
                    and dist <= RELEASE_RADIUS_M:
                return Decision(None, 0.0, "release",
                                f"距容器 {dist * 100:.1f} cm <= {RELEASE_RADIUS_M * 100:.0f} cm（距离判据）")

        qs = questions_for(ws.phase, ws.task, self.single_question)
        state = ws.to_laya_state()
        # 只把 body 喂给模型：numeric 是给日志的，塞进去只会稀释文本信号。
        res = self.agent.predict({"body": state["body"]}, qs)
        ans = res["answers"]

        if self.trust_laya_gates:
            if "grasp_now" in ans and ans["grasp_now"]["noul"] >= self.gate:
                return Decision(None, 1.0, "grasp",
                                f"grasp_now={ans['grasp_now']['noul']:.2f} >= {self.gate}",
                                confidence=ans["grasp_now"]["confidence"], raw=ans)
            if "release_now" in ans and ans["release_now"]["noul"] >= self.gate:
                return Decision(None, 0.0, "release",
                                f"release_now={ans['release_now']['noul']:.2f} >= {self.gate}",
                                confidence=ans["release_now"]["confidence"], raw=ans)

        # --- 移动 ---
        if self.single_question:
            name = ans["move"]["choice"]
            act = A.BY_NAME[name]
            conf = ans["move"]["confidence"]
            why = f"move={name} (conf {conf:.2f}, 12 选 1，confidence 未标定)"
        else:
            direction = ans["direction"]["choice"]
            if self.trust_laya_scale:
                scale = ans["scale"]["choice"]
                conf = min(ans["direction"]["confidence"], ans["scale"]["confidence"])
                why = (f"dir={direction}({ans['direction']['confidence']:.2f}) "
                       f"scale={scale}({ans['scale']['confidence']:.2f}) 均由模型给")
            else:
                # 幅度是距离的确定函数，不是需要语义判断的东西。
                scale = "big" if dist > COARSE_ABOVE_M else "small"
                conf = ans["direction"]["confidence"]
                why = (f"dir={direction}({conf:.2f}) 由模型 / "
                       f"scale={scale} 由距离 {dist * 100:.1f}cm")
            act = A.lookup(direction, scale)
        return Decision(act, ws.gripper, "move", why, confidence=conf, raw=ans)


def compare(laya_d: Decision, geo_d: Decision) -> Dict[str, Any]:
    """两个策略这一步是否一致。逐步记下来，一轮跑完就是符合率。"""
    same_kind = laya_d.kind == geo_d.kind
    same_action = (laya_d.action.name if laya_d.action else None) == \
                  (geo_d.action.name if geo_d.action else None)
    # 方向对、幅度不对，比方向就错了要好得多，分开记。
    same_axis = (laya_d.action is not None and geo_d.action is not None
                 and laya_d.action.axis == geo_d.action.axis
                 and laya_d.action.sign == geo_d.action.sign)
    return {"agreed": bool(same_kind and same_action),
            "same_kind": bool(same_kind),
            "same_direction": bool(same_axis),
            "laya": laya_d.action.name if laya_d.action else laya_d.kind,
            "geometric": geo_d.action.name if geo_d.action else geo_d.kind}
