"""把「夹爪在哪、目标在哪」画成一帧 Snake，喂给 PlayJev。

**为什么绕这一圈。** PlayJev 零样本看真实俯视图是不行的（`probe_playjev.py`：
镜像不交换、把目标涂掉反而更自信）。它会的是十个 2D 小游戏，其中 Snake 就是
「把白方块开到红方块上」—— 这恰好就是平面抓取里「把夹爪开到目标正上方」。
所以不给它照片，给它一帧**它训练时见过的那种画面**：

    深蓝底 BGR(168,0,0)、白方块 = 夹爪、红方块 = 目标、方块 17 px、格距 19 px

这些数是从 PlayJev 自己的 Snake 截图上量出来的（448x429）。画得越像，越在分布内。

**画面方向是我们定的，所以不需要标定。** 渲染时约定：

    画面上 = 基座 +x（离开底座、往桌子远端）
    画面左 = 基座 +y（机器人的左手边）

模型说 up 就是 +x，说 left 就是 +y，这是**构造出来**的，不是估出来的。
唯一需要标定的是「真实相机里目标的像素 → 基座 xy」，那是 localizer 的事。

只管平面 (x,y)。高度不交给模型：对准之后竖直下探、合爪、抬起是脚本，
和 RoboJev 里「Laya 管方向、距离管门控」同一个分工。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

W, H = 448, 429               # PlayJev Snake 截图尺寸
BG = (168, 0, 0)              # BGR 深蓝
HEAD = (255, 255, 255)        # 白：夹爪
FOOD = (0, 1, 253)            # 红：目标
CELL = 17                     # 方块边长
PITCH = 19                    # 格距
COLS = W // PITCH             # 23
ROWS = H // PITCH             # 22

# Snake 训练时的原样措辞和选项，一个字都不改。
INSTRUCTIONS = "Which move should the player make next?"
OPTIONS: Dict[str, str] = {
    "up": "turn the snake to move up",
    "down": "turn the snake to move down",
    "left": "turn the snake to move left",
    "right": "turn the snake to move right",
}

# 画面方向 -> 基座系平移方向（单位向量，只有 xy）。见模块文档的约定。
SNAKE_TO_BASE: Dict[str, Tuple[float, float]] = {
    "up": (+1.0, 0.0),
    "down": (-1.0, 0.0),
    "left": (0.0, +1.0),
    "right": (0.0, -1.0),
}
# 对应到 actions.py 里 12 个动作的方向名
SNAKE_TO_ACTION: Dict[str, str] = {
    "up": "forward", "down": "backward", "left": "left", "right": "right",
}


@dataclass
class Board:
    """基座系 xy 矩形 -> 格子。默认取工作空间盒的 xy 范围。"""

    x_min: float = 0.20
    x_max: float = 0.66
    y_min: float = -0.516
    y_max: float = 0.43

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """(x,y) 米 -> (col,row)。上 = +x，左 = +y。"""
        u = (self.y_max - y) / (self.y_max - self.y_min)      # 0 在左（+y 大）
        v = (self.x_max - x) / (self.x_max - self.x_min)      # 0 在上（+x 大）
        col = int(np.clip(round(u * (COLS - 1)), 0, COLS - 1))
        row = int(np.clip(round(v * (ROWS - 1)), 0, ROWS - 1))
        return col, row

    @property
    def metres_per_cell(self) -> Tuple[float, float]:
        return ((self.x_max - self.x_min) / (ROWS - 1), (self.y_max - self.y_min) / (COLS - 1))


def draw(head: Tuple[int, int], food: Tuple[int, int]) -> np.ndarray:
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = BG
    for (c, r), colour in ((food, FOOD), (head, HEAD)):     # 蛇头后画，重合时盖在上面
        x0, y0 = 2 + c * PITCH, 2 + r * PITCH
        img[y0:y0 + CELL, x0:x0 + CELL] = colour
    return img


def render(gripper_xy, target_xy, board: Board = Board()) -> Tuple[np.ndarray, tuple, tuple]:
    head = board.to_cell(*gripper_xy)
    food = board.to_cell(*target_xy)
    return draw(head, food), head, food
