#!/usr/bin/env python3
"""网页点选抓取：把 click_grasp 的 OpenCV 弹窗换成网页，别的一行不动。

    conda activate laya; cd ~/yuhang_workspace/算法/Jev/RobotJev
    PYTHONPATH= python -m robojev.click_grasp_web --link ~/.atom_agent/robojev_click         # 点选 + 抓取
    PYTHONPATH= python -m robojev.click_grasp_web --link ~/.atom_agent/robojev_click --dry   # 只点选看判定，不连臂

一般不手敲：网页「实时推理测试」页点「开始」，由工作站代理拉起来。

**放在 robojev/ 下，和 click_grasp.py 同一个目录。** 它不改 click_grasp.py，只借用：
  * ClickUI      点 / 框 / 排除点 → SAM 几个粒度 → 当前粒度的判定 → 画图，全部原样
  * Planner      判「能不能抓」，用的就是抓取流程自己的 top335.feasibility
  * grab         开窗那一帧
  * select_from_camera 这个入口：run_composite --click 调它拿点选结果。这里把它换成
    网页版 —— 弹窗换成网页，之后的抓取流程（335 接近 → 腕部接手 → 伺服 → 合爪 → 抬起）
    完全是 run_composite 自己的，一个字没碰。

和网页之间走文件，约定和采集脚本的 --remote 一样（代理负责转给浏览器）：
    control.json  网页 → 脚本  {"seq": 17, "cmd": "click", "x": 312.5, "y": 208.0, "shift": false}
                  按 seq 去重，不靠删文件（删文件有竞态，可能读到写了一半的内容）
    view.jpg      脚本 → 网页  ClickUI.render() 的画面（去掉底下那条文字栏，字由网页自己写）
    state.json    脚本 → 网页  阶段、当前粒度、名字、高 / 宽、判定、acked
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# 网页能发的命令。和 click_grasp.select() 里的鼠标 / 键盘一一对应
COMMANDS = ("click", "drag", "exclude", "cycle", "reset", "resnap", "confirm", "cancel")


class WebLink:
    """脚本和网页之间的三个文件。写都是先写 .tmp 再 rename，网页任何时候读到的都是完整的。"""

    def __init__(self, link_dir):
        self.dir = os.path.expanduser(str(link_dir))
        os.makedirs(self.dir, exist_ok=True)
        self.control_path = os.path.join(self.dir, "control.json")
        self.view_path = os.path.join(self.dir, "view.jpg")
        self.state_path = os.path.join(self.dir, "state.json")
        # 开工时把已有的 seq 记下来：上一轮留下的命令不该在这一轮被执行
        self._seen = self._read_seq()
        self._acked = self._seen
        self._state = {}
        self._view_ts = 0.0

    def _read_seq(self):
        try:
            with open(self.control_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("seq", 0))
        except Exception:                                     # noqa: BLE001
            return 0

    def poll(self):
        """有新命令就返回那条（dict），没有返回 None。"""
        try:
            with open(self.control_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:                                     # noqa: BLE001
            return None                                       # 没有，或者正在被写
        seq = int(doc.get("seq", 0))
        if seq <= self._seen:
            return None
        self._seen = seq
        if str(doc.get("cmd", "")) not in COMMANDS:
            self._acked = seq                                 # 不认识的也回执，别让网页干等
            return None
        doc["_seq"] = seq
        return doc

    def ack(self, seq):
        self._acked = max(self._acked, int(seq))

    def _atomic(self, path, data, binary=False):
        tmp = path + ".tmp"
        with open(tmp, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as f:
            f.write(data)
        os.replace(tmp, path)

    def publish(self, bgr):
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            self._atomic(self.view_path, buf.tobytes(), binary=True)
            self._view_ts = time.time()

    def status(self, **fields):
        """更新并写出 state.json。只给要改的字段，其余沿用上一次的。"""
        self._state.update(fields)
        doc = dict(self._state, acked=self._acked, view_ts=self._view_ts, ts=time.time(),
                   pid=os.getpid())
        try:
            self._atomic(self.state_path, json.dumps(doc, ensure_ascii=False, default=_jsonable))
        except Exception:                                     # noqa: BLE001
            pass                                              # 写不了也不能把抓取带下去


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def describe(ui, frame_hw):
    """当前选中的东西，给网页写字用。ui.current() 在 render() 里已经算过，这里取的是缓存。"""
    h, w = frame_hw
    out = {"frame": [int(w), int(h)], "n": len(ui.masks), "k": 0, "name": "", "ok": False, "why": "",
           "height_mm": None, "width_mm": None, "msg": ui.msg,
           "points": [[float(u), float(v), int(lab)] for (u, v), lab in zip(ui.points, ui.labels)],
           "box": None if ui.box is None else [float(x) for x in ui.box]}
    c = ui.current() if ui.masks else None
    if c is not None:
        o = c["obs"]
        hh = None if o is None else o.extra.get("height")
        ww = None if o is None else o.extra.get("width")
        out.update(k=ui.k % len(ui.masks) + 1, name=c["name"], ok=bool(c["ok"]), why=c["why"],
                   height_mm=None if hh is None else round(float(hh) * 1000),
                   width_mm=None if ww is None else round(float(ww) * 1000))
    return out


def web_select(tf, color, depth, planner, link, regrab=None, dry=False):
    """click_grasp.select() 的网页版：同一个 ClickUI，事件从网页来，画面发给网页。

    返回 {points, labels, box, mask, name, obs, verdict}（和 select() 一样）或 None（退出）。
    """
    from .click_grasp import SCALE, ClickUI

    ui = ClickUI(tf, color, depth, planner)

    def push(phase="ready"):
        img = ui.render()
        # 底下那条 96px 的文字栏不要：名字、粒度、高宽、判定都由网页自己写，字更清楚
        img = img[: int(round(ui.color.shape[0] * SCALE))]
        link.publish(img)
        link.status(phase=phase, dry=dry, error="", **describe(ui, ui.color.shape[:2]))

    push()
    while True:
        cmd = link.poll()
        if cmd is None:
            time.sleep(0.03)                                  # 和原来窗口的 waitKey(30) 一个节奏
            continue
        c, seq = cmd["cmd"], cmd["_seq"]
        link.status(phase="busy", busy=c)                     # SAM 要算一会儿，先让网页知道在算
        try:
            if c == "click":
                ui.click((float(cmd["x"]), float(cmd["y"])), shift=bool(cmd.get("shift")))
            elif c == "drag":
                ui.drag((float(cmd["x0"]), float(cmd["y0"])), (float(cmd["x1"]), float(cmd["y1"])))
            elif c == "exclude":
                ui.exclude((float(cmd["x"]), float(cmd["y"])))
            elif c == "cycle":
                ui.cycle(-1 if int(cmd.get("d", 1)) < 0 else 1)
            elif c == "reset":
                ui.reset()
            elif c == "resnap":
                if regrab is None:
                    ui.msg = "这一帧是存下来的，不能重拍"
                else:
                    color, depth = regrab()
                    ui = ClickUI(tf, color, depth, planner)
                    ui.msg = "重拍了一帧"
            elif c == "cancel":
                link.ack(seq)
                link.status(phase="cancelled", busy="")
                return None
            elif c == "confirm":
                sel = ui.result()                             # 抓不了的会返回 None，并把原因写进 msg
                if sel is not None:
                    link.ack(seq)
                    link.status(phase="confirmed", busy="", **describe(ui, ui.color.shape[:2]))
                    return sel
        except Exception as exc:                              # noqa: BLE001
            # 一次分割 / 判定出错不该把整个会话带走：说出来，让人换个点再试
            ui.msg = "出错了：%s" % exc
            traceback.print_exc()
        link.ack(seq)
        push()
        link.status(busy="")


def make_select_from_camera(link):
    """造一个签名和 click_grasp.select_from_camera 一样的函数，给 run_composite 用。"""
    from .click_grasp import Planner, grab

    def select_from_camera(tf, cam335, full_open=0.098):
        link.status(phase="grabbing")
        color, depth = grab(cam335)
        planner = Planner(full_open)
        print("点选已交给网页（335 这一帧）", flush=True)
        sel = web_select(tf, color, depth, planner, link, regrab=lambda: grab(cam335))
        if sel is not None:
            h, w = sel["obs"].extra.get("height"), sel["obs"].extra.get("width")
            h_txt = "—" if h is None else f"{h * 1000:.0f}mm"
            w_txt = "—" if w is None else f"{w * 1000:.0f}mm"
            print(f"点选：「{sel['name']}」高 {h_txt}，沿夹爪方向宽 {w_txt} -> {sel['verdict']}", flush=True)
            link.status(phase="grasping")
        return sel

    return select_from_camera


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", required=True, help="和网页交换文件的目录（代理拉起时会给）")
    ap.add_argument("--dry", action="store_true", help="只点选、看判定，不连臂")
    ap.add_argument("--frame", default=None, help="（配 --dry）用存下的 npz（color, depth）代替现拍")
    ap.add_argument("--record", default=None, help="录像目录，同 run_composite --record")
    a = ap.parse_args()

    link = WebLink(a.link)
    link.status(phase="loading", dry=a.dry, error="", n=0, k=0, name="", ok=False, why="", msg="")
    code = 0
    try:
        if not a.dry:
            # 把入口换成网页版。两处都换：run_composite 可能是 `from .click_grasp import
            # select_from_camera`（那得换它自己的名字），也可能是用时才去 click_grasp 里取
            from . import click_grasp
            web = make_select_from_camera(link)
            click_grasp.select_from_camera = web
            from . import run_composite
            if hasattr(run_composite, "select_from_camera"):
                run_composite.select_from_camera = web
            sys.argv = ["run_composite", "--click", "--stage", "grasp"] + (
                ["--record", a.record] if a.record else [])
            code = int(run_composite.main() or 0)
            return code

        # ---- --dry：和 click_grasp --dry 同一条路，只是弹窗换成网页 ----
        import warnings
        warnings.filterwarnings("ignore")
        from .calibrate_gripper import apply_calibration
        from .click_grasp import CALIB, Planner, grab
        from .objects import TargetFinder

        tf = TargetFinder("pick up the object")
        apply_calibration(np.asarray(json.loads((CALIB / "wrist_J.json").read_text())["J"], float))
        if a.frame:
            d = np.load(a.frame)
            sel = web_select(tf, d["color"], d["depth"], Planner(), link, dry=True)
        else:
            from .camera import TOP_SERIAL, OrbbecRGBD

            cam = OrbbecRGBD(serial=TOP_SERIAL)
            cam.connect()
            try:
                link.status(phase="grabbing")
                color, depth = grab(cam)
                sel = web_select(tf, color, depth, Planner(), link, regrab=lambda: grab(cam), dry=True)
            finally:
                cam.disconnect()
        if sel is None:
            print("没选，退出", flush=True)
        else:
            print(f"选中「{sel['name']}」：{sel['verdict']}（--dry，不抓）", flush=True)
        return 0
    except SystemExit as exc:
        # run_composite 可能直接 sys.exit(n) 收尾 —— 那是正常退出，不是出错
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            link.status(phase="error", error="run_composite 退出码 %s" % code)
        raise
    except BaseException as exc:                              # noqa: BLE001
        code = 1
        link.status(phase="error", error="%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        st = link._state.get("phase")
        if st not in ("error",):
            link.status(phase="done", exit_code=code, busy="")


if __name__ == "__main__":
    raise SystemExit(main())
