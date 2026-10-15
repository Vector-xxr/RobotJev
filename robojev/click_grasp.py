#!/usr/bin/env python3
"""鼠标点选抓取：335 拍一帧 -> 点要抓的物体 -> SAM 分割出框和边缘、量高 / 宽、判能不能抓 -> Enter -> 臂去抓。

    conda activate laya; cd ~/yuhang_workspace/算法/Jev/RoboJev
    PYTHONPATH= python -m robojev.click_grasp                         # 点选 + 抓取
    PYTHONPATH= python -m robojev.click_grasp --record /tmp/rec       # 同上，并录像
    PYTHONPATH= python -m robojev.click_grasp --dry                   # 只点选、看判定，不连臂
    PYTHONPATH= python -m robojev.click_grasp --dry --frame f.npz     # 用存下的一帧（color/depth），不开相机

窗口里：
    左键点物体               分割出这个物体（按住拖 = 框选；Shift+左键 = 再加一个点）
    右键                     排除点：点到的东西不要（分多了时用）
    Tab / 空格 / 滚轮        换分割粒度（一个点可能对应「球」「球 + 瓶颈」「球 + 整瓶」）
    r                        重来
    Enter                    抓当前选中的（只有判定「能抓」时才接受）
    Esc / q                  退出，不抓

画面只用开窗那一帧，不实时刷新。确认后才连臂；臂回 0 位后按同样的提示在新画面上重新分割，
和点选时的掩码对不上（物体被挪了 / 被挡了）就不抓。之后的流程和按指令抓完全一样
（335 接近 -> 腕部接手 -> 伺服 -> 按 CLIP 判的力度合爪 -> 抬起判成败），只是目标不按名字找：
腕部交接按 335 测的位置投影到腕部画面，伺服中每帧在上一次的位置给 SAM 点提示（见 TargetFinder.track_px）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"))

ROOT = Path(__file__).resolve().parents[4] / "数据_硬件" / "SO101"
CALIB = Path(__file__).resolve().parent / "calibration"
WIN = "click to grasp"
SCALE = 1.5                      # 窗口放大倍数（640x480 太小，点不准）
FONT_PATHS = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
HELP = "左键点物体（拖=框选，Shift=加点） 右键=排除  Tab/滚轮=换粒度  r=重来  Enter=抓  Esc=退出"


def _font(size):
    from PIL import ImageFont

    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def name_of(tf, color, mask):
    """CLIP 给选中的物体起个名字（COCO 80 类里最像的）：只给抓取力度 / 位姿的判断当参考，腕部交接不靠它。
    9/29 实测：球 -> sports ball、瓶子 -> bottle 对；泡沫板这种词表外的东西会起成 surfboard 之类。"""
    from .objects import coco_names

    class Box:
        pass

    ys, xs = np.nonzero(mask)
    b = Box()
    b.xyxy = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    names = coco_names()
    s = tf.clip.scores(color, [b], [f"a photo of a {n}" for n in names])[0]
    return names[int(np.argmax(s))]


class Planner:
    """点选界面里判「能不能抓」要的东西：外参、J、运动学（缓存的 DH，还没连臂）、起点（0 位）。
    判定用的就是抓取流程自己的 top335.feasibility，所以界面说能抓的，流程一定接受（除非物体又被挪了）。"""

    def __init__(self, full_open=0.098):
        from atom_robot.ar5.config import AR5Config
        from atom_robot.ar5.kinematics import AR5Kinematics

        from .objects import CameraModel
        from .top335 import load_extrinsic

        cfg = AR5Config().arm
        dh = json.loads((ROOT / "atom_robot" / "ar5" / "ar5_dh.json").read_text())["dh"]
        self.kin = AR5Kinematics(np.array(dh).reshape(-1, 4), cfg.num_joints, cfg.joint_min, cfg.joint_max)
        self.R, self.t, self.meta = load_extrinsic()
        self.q_ref = np.asarray(self.meta["q_ref"], float)
        self.J = np.asarray(json.loads((CALIB / "wrist_J.json").read_text())["J"], float)
        centre = np.asarray(json.loads((ROOT / "atom_robot" / "ar5" / "joint_calibration.json").read_text())
                            ["centre_joints"], float)
        self.f_start = self.kin.fk(centre)[:3, 3]            # 抓取流程先回 0 位再出发
        self.cm = CameraModel(CALIB / "orbbec335_params.json")
        self.full_open = full_open

    def judge(self, tf, depth, mask):
        """-> (能抓?, 说明, ObjectObs 或 None)"""
        from .top335 import feasibility, grip_width, measure_mask

        r = measure_mask(tf, depth, mask, self.cm, self.R, self.t)
        if r is None:
            return False, "这块区域没有有效深度，量不出几何", None
        obs, plane = r
        obs.extra["width"] = grip_width(obs.extra.get("grasp_pts"), self.J, self.R)
        why, p = feasibility(obs, plane, self.R, self.t, self.J, self.kin, self.q_ref, self.cm, self.f_start,
                             self.full_open, need_height=True)
        if why:
            return False, why, obs
        return True, "能抓" + (p["note"] or f"（平移高度 {p['f_above'][2] * 1000:.0f}mm）"), obs


class ClickUI:
    """点选的状态和画面（不含窗口）：点 / 框 / 排除点 -> SAM 几个粒度 -> 当前粒度的判定 -> 画图。
    select() 把 OpenCV 窗口的鼠标、键盘接到这里；离线测试直接调这些方法。"""

    def __init__(self, tf, color, depth, planner):
        self.tf, self.color, self.depth, self.planner = tf, color, depth, planner
        self.points, self.labels, self.box = [], [], None
        self.masks, self.k, self.cache = [], 0, {}
        self.drag0 = self.drag1 = None
        self.msg = ""
        self.f_big, self.f_small = _font(20), _font(15)

    # ---- 输入
    def click(self, p, shift=False):
        if shift and self.points:
            self.points.append(p); self.labels.append(1)          # 同一个物体再加一个点
        else:
            self.points, self.labels, self.box = [p], [1], None    # 新的一次点选
        self._reprompt()

    def drag(self, p0, p1):
        (x0, y0), (x1, y1) = p0, p1
        self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.points, self.labels = [], []
        self._reprompt()

    def exclude(self, p):
        if self.points or self.box is not None:
            self.points.append(p); self.labels.append(0)
            self._reprompt()

    def cycle(self, d=1):
        if self.masks:
            self.k += d

    def reset(self):
        self.points, self.labels, self.box = [], [], None
        self._reprompt()

    def _reprompt(self):
        self.cache, self.k = {}, 0
        if not self.points and self.box is None:
            self.masks = []
            return
        t0 = time.time()
        self.masks = self.tf.segment_prompt(self.color, points=self.points or None, labels=self.labels or None,
                                            box=self.box, multimask=True)
        self.msg = f"分割出 {len(self.masks)} 个粒度，{(time.time() - t0) * 1000:.0f}ms"

    # ---- 当前选中的
    def current(self):
        if not self.masks:
            return None
        k = self.k % len(self.masks)
        if k not in self.cache:
            m = self.masks[k]
            ok, why, obs = self.planner.judge(self.tf, self.depth, m)
            self.cache[k] = dict(mask=m, ok=ok, why=why, obs=obs, name=name_of(self.tf, self.color, m))
        return self.cache[k]

    def result(self):
        """Enter：当前粒度能抓就返回点选结果，否则 None。"""
        c = self.current()
        if c is None or not c["ok"]:
            self.msg = "这个抓不了，换一个物体或换粒度"
            return None
        return dict(points=list(self.points), labels=list(self.labels), box=self.box,
                    mask=c["mask"], name=c["name"], obs=c["obs"], verdict=c["why"])

    # ---- 画面
    def render(self):
        import cv2
        from PIL import Image, ImageDraw

        img = self.color.copy()
        c = self.current()
        if c is not None:
            col = (0, 210, 0) if c["ok"] else (0, 0, 255)
            m = c["mask"]
            img[m] = (0.6 * img[m] + 0.4 * np.array(col)).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, col, 2)
            ys, xs = np.nonzero(m)
            cv2.rectangle(img, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (255, 255, 0), 1)
        for (u, v), lab in zip(self.points, self.labels):
            cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0) if lab else (0, 0, 255),
                           cv2.MARKER_CROSS if lab else cv2.MARKER_TILTED_CROSS, 14, 2)
        if self.box is not None:
            b = [int(v) for v in self.box]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 1)
        if self.drag0 is not None and self.drag1 is not None:
            (x0, y0), (x1, y1) = self.drag0, self.drag1
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 1)
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        H, W = img.shape[:2]
        img = np.vstack([img, np.full((96, W, 3), 30, np.uint8)])
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        dr = ImageDraw.Draw(pil)
        if c is not None:
            o = c["obs"]
            h = None if o is None else o.extra.get("height")
            w = None if o is None else o.extra.get("width")
            h_txt = "—" if h is None else f"{h * 1000:.0f}mm"
            w_txt = "—" if w is None else f"{w * 1000:.0f}mm"
            dr.text((10, H + 6), f"「{c['name']}」  粒度 {self.k % len(self.masks) + 1}/{len(self.masks)}   "
                                 f"高 {h_txt}   沿夹爪方向宽 {w_txt}", font=self.f_big, fill=(255, 255, 255))
            why = ("可以抓：" if c["ok"] else "抓不了：") + c["why"] + ("    按 Enter 抓" if c["ok"] else "")
            dr.text((10, H + 34), why[:95], font=self.f_small, fill=(90, 230, 90) if c["ok"] else (255, 110, 110))
        else:
            dr.text((10, H + 6), "点一下要抓的物体", font=self.f_big, fill=(255, 255, 255))
        dr.text((10, H + 60), HELP, font=self.f_small, fill=(180, 180, 180))
        if self.msg:
            dr.text((W - 300, H + 6), self.msg, font=self.f_small, fill=(160, 160, 160))
        return np.asarray(pil)[:, :, ::-1].copy()


def select(tf, color, depth, planner, window=WIN):
    """弹窗点选。返回 {points, labels, box, mask, name, obs, verdict} 或 None（退出）。"""
    import cv2

    ui = ClickUI(tf, color, depth, planner)
    dirty = [True]

    def on_mouse(ev, x, y, flags, _):
        p = (x / SCALE, y / SCALE)
        if p[1] >= color.shape[0]:
            return                                              # 点在下面的文字栏里
        if ev == cv2.EVENT_LBUTTONDOWN:
            ui.drag0 = ui.drag1 = p
        elif ev == cv2.EVENT_MOUSEMOVE and ui.drag0 is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            ui.drag1 = p; dirty[0] = True
        elif ev == cv2.EVENT_LBUTTONUP and ui.drag0 is not None:
            p0 = ui.drag0
            ui.drag0 = ui.drag1 = None
            if abs(p[0] - p0[0]) > 6 and abs(p[1] - p0[1]) > 6:
                ui.drag(p0, p)
            else:
                ui.click(p, shift=bool(flags & cv2.EVENT_FLAG_SHIFTKEY))
            dirty[0] = True
        elif ev == cv2.EVENT_RBUTTONDOWN:
            ui.exclude(p); dirty[0] = True
        elif ev == cv2.EVENT_MOUSEWHEEL:
            ui.cycle(1 if cv2.getMouseWheelDelta(flags) < 0 else -1); dirty[0] = True

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            if dirty[0]:
                cv2.imshow(window, ui.render())
                dirty[0] = False
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                return None
            if key in (9, 32, ord("n")):
                ui.cycle(1); dirty[0] = True
            if key == ord("r"):
                ui.reset(); dirty[0] = True
            if key in (13, 10):
                sel = ui.result()
                if sel is not None:
                    return sel
                dirty[0] = True
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return None                                 # 窗口被关掉
            except cv2.error:
                return None
    finally:
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
        except cv2.error:
            pass


def grab(cam, n_warm=8):
    """开窗用的那一帧：先丢几帧让曝光稳定。"""
    f = None
    for _ in range(n_warm):
        f = cam.read()
    for _ in range(20):
        if f is not None and f.ok and f.depth is not None:
            return f.color.copy(), f.depth.copy()
        time.sleep(0.05)
        f = cam.read()
    raise RuntimeError("335 读不到帧")


def select_from_camera(tf, cam335, full_open=0.098):
    """run_composite --click 调用：拍一帧、点选。返回 select 的结果（None = 退出）。"""
    color, depth = grab(cam335)
    planner = Planner(full_open)
    print("点选窗口已打开（335 这一帧）：" + HELP)
    sel = select(tf, color, depth, planner)
    if sel is not None:
        h, w = sel["obs"].extra.get("height"), sel["obs"].extra.get("width")
        h_txt = "—" if h is None else f"{h * 1000:.0f}mm"
        w_txt = "—" if w is None else f"{w * 1000:.0f}mm"
        print(f"点选：「{sel['name']}」高 {h_txt}，沿夹爪方向宽 {w_txt} -> {sel['verdict']}")
    return sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="只点选、看判定，不连臂")
    ap.add_argument("--frame", default=None, help="（配 --dry）用存下的 npz（color, depth）代替现拍")
    ap.add_argument("--record", default=None, help="录像目录，同 run_composite --record")
    a = ap.parse_args()
    if not a.dry:
        from . import run_composite
        argv = ["run_composite", "--click", "--stage", "grasp"] + (["--record", a.record] if a.record else [])
        sys.argv = argv
        return run_composite.main()

    import warnings
    warnings.filterwarnings("ignore")
    from .objects import TargetFinder

    tf = TargetFinder("pick up the object")
    from .calibrate_gripper import apply_calibration
    apply_calibration(np.asarray(json.loads((CALIB / "wrist_J.json").read_text())["J"], float))
    if a.frame:
        d = np.load(a.frame)
        color, depth = d["color"], d["depth"]
        sel = select(tf, color, depth, Planner())
    else:
        from .camera import TOP_SERIAL, OrbbecRGBD

        cam = OrbbecRGBD(serial=TOP_SERIAL)
        cam.connect()
        try:
            sel = select_from_camera(tf, cam)
        finally:
            cam.disconnect()
    if sel is None:
        print("没选，退出")
        return 0
    print(f"选中「{sel['name']}」：{sel['verdict']}（--dry，不抓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
