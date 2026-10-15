"""把一句自然语言指令拆成 (要抓什么, 放到哪里)，并映射到 YOLO 认得的类别。

    把苹果放到盘子里   ->  Task(target='apple', container='bowl', ...)
    put the apple in the bowl

**为什么不用 Laya 做这一步**：这是纯文本抽取，规则就够，而且必须是确定性的 ——
指令解析错了，后面整条闭环都在追错误的目标，还很难看出来。Laya 在这里也帮不上忙，
它不生成文本，只能在给定选项里选，而「指令里的名词是什么」不是一个固定选项集。

⚠️ **COCO 里没有 plate（盘子）。** YOLO 预训练权重是 COCO 80 类，最接近盘子的是
`bowl`（碗）。说「盘子」会被映射到 `bowl`，实际检测能不能框住你桌上那个盘子，
取决于它长得像不像碗。真要认盘子，得自己标数据微调 YOLO —— 这是 YOLO 的问题，
不是这一层的问题。所有做了近似的映射都会在 Task.notes 里写明，别让它静默发生。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 中文 -> COCO 类名。只收工作台上可能出现的。
# 值是 (coco类名, 是否精确)。不精确的会在 notes 里提醒。
ZH_TO_COCO: Dict[str, Tuple[str, bool]] = {
    "苹果": ("apple", True),
    "香蕉": ("banana", True),
    "橙子": ("orange", True), "橘子": ("orange", True), "桔子": ("orange", True),
    "瓶子": ("bottle", True), "水瓶": ("bottle", True), "矿泉水": ("bottle", True),
    "杯子": ("cup", True), "水杯": ("cup", True), "马克杯": ("cup", True),
    "碗": ("bowl", True),
    "盘子": ("bowl", False),      # COCO 没有 plate
    "碟子": ("bowl", False),
    "盒子": ("book", False),      # COCO 没有 box，book 是最接近的矩形物
    "剪刀": ("scissors", True),
    "勺子": ("spoon", True), "汤匙": ("spoon", True),
    "叉子": ("fork", True),
    "刀": ("knife", True), "小刀": ("knife", True),
    "遥控器": ("remote", True),
    "手机": ("cell phone", True),
    "鼠标": ("mouse", True),
    "键盘": ("keyboard", True),
    "书": ("book", True), "本子": ("book", True),
    "玩偶": ("teddy bear", True), "娃娃": ("teddy bear", True), "毛绒": ("teddy bear", True),
    "花瓶": ("vase", True),
    "时钟": ("clock", True), "钟": ("clock", True),
}

# 英文直接用 COCO 名，这里只收几个常见别名。
EN_ALIASES: Dict[str, Tuple[str, bool]] = {
    "plate": ("bowl", False),
    "box": ("book", False),
    "mug": ("cup", True),
    "water bottle": ("bottle", True),
    "phone": ("cell phone", True),
}

# 「放到 Y 里」的动词/介词。中文按「把 X ... Y ...」的骨架切。
_ZH_PATTERNS = [
    re.compile(r"把\s*(?P<t>.+?)\s*(?:放|摆|搁|移|送)(?:到|进|入|在)\s*(?P<c>.+?)\s*(?:里|中|内|上|里面|上面)?\s*$"),
    re.compile(r"将\s*(?P<t>.+?)\s*(?:放|摆|搁|移|送)(?:到|进|入|在)\s*(?P<c>.+?)\s*(?:里|中|内|上|里面|上面)?\s*$"),
    re.compile(r"^\s*(?:拿起|抓起|抓|拿)\s*(?P<t>.+?)\s*$"),          # 只抓，不放
]
_EN_PATTERNS = [
    re.compile(r"\b(?:put|place|move|drop)\s+(?:the\s+)?(?P<t>.+?)\s+(?:in|into|on|onto|inside)\s+(?:the\s+)?(?P<c>.+?)\s*\.?$", re.I),
    re.compile(r"\b(?:pick\s+up|grasp|grab|take)\s+(?:the\s+)?(?P<t>.+?)\s*\.?$", re.I),
]


class ParseError(ValueError):
    """指令看不懂，或者说到的东西 YOLO 不认识。"""


@dataclass
class Task:
    instruction: str
    target: str                       # COCO 类名，要抓的那个
    container: Optional[str] = None   # COCO 类名，放进去的那个；只抓不放时为 None
    target_said: str = ""             # 用户原话里的词
    container_said: str = ""
    notes: List[str] = field(default_factory=list)   # 做过的近似，要让人看见

    @property
    def needs_place(self) -> bool:
        return self.container is not None

    def describe(self) -> str:
        if self.needs_place:
            return f"pick up the {self.target} and place it in the {self.container}"
        return f"pick up the {self.target}"


def _map_noun(said: str, notes: List[str], coco_names: Optional[List[str]]) -> str:
    """一个名词 -> COCO 类名。映射不精确或类别不存在都要说出来。"""
    said = said.strip().strip("。.！!，,的 ")
    key = said.lower()

    if said in ZH_TO_COCO:
        name, exact = ZH_TO_COCO[said]
    elif key in EN_ALIASES:
        name, exact = EN_ALIASES[key]
    else:
        name, exact = key, True       # 当成 COCO 原名试试

    if not exact:
        notes.append(f"「{said}」COCO 里没有，按最接近的 '{name}' 处理 —— 检测不到就是这个原因")

    if coco_names is not None and name not in coco_names:
        raise ParseError(
            f"「{said}」映射到 '{name}'，但 YOLO 的类别表里没有这个类。\n"
            f"  这个模型认识的 80 类里，和工作台相关的有："
            + ", ".join(n for n in coco_names if n in {
                "apple", "banana", "orange", "bottle", "cup", "bowl", "spoon", "fork",
                "knife", "scissors", "remote", "cell phone", "mouse", "keyboard",
                "book", "teddy bear", "vase", "clock", "sandwich", "donut", "cake"})
            + "\n  要认别的东西就得自己标数据微调 YOLO。"
        )
    return name


def parse(instruction: str, coco_names: Optional[List[str]] = None) -> Task:
    """解析一条指令。`coco_names` 给了就顺便校验类别存在。"""
    text = instruction.strip()
    if not text:
        raise ParseError("指令是空的")

    for pat in _ZH_PATTERNS + _EN_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        notes: List[str] = []
        t_said = m.group("t")
        c_said = m.groupdict().get("c")
        target = _map_noun(t_said, notes, coco_names)
        container = _map_noun(c_said, notes, coco_names) if c_said else None
        return Task(instruction=text, target=target, container=container,
                    target_said=t_said.strip(), container_said=(c_said or "").strip(),
                    notes=notes)

    raise ParseError(
        f"看不懂这条指令：{text!r}\n"
        "  支持的说法：\n"
        "    把<东西>放到<容器>里 / 将<东西>放进<容器>\n"
        "    拿起<东西> / 抓<东西>\n"
        "    put the <thing> in the <container>\n"
        "    pick up the <thing>"
    )
