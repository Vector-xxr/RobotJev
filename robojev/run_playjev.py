#!/usr/bin/env python3
"""PlayJev 控制机械臂抓取：夹爪和目标画成一帧 Snake -> PlayJev 选方向 -> 臂走一步。

    # mock 臂 + 合成目标，整条链跑一遍，不碰任何硬件
    python -m robojev.run_playjev --sim-target 0.50 0.05

    # 真相机找蓝色目标 + mock 臂（要先有 homography 标定）
    python -m robojev.run_playjev --target-color blue

    # 真机（要 homography 标定 + 显式给抓取高度）
    python -m robojev.run_playjev --target-color blue --real --grasp-z 0.12

分三段，只有第一段交给模型：

    ALIGN    PlayJev 看 Snake 画面选 up/down/left/right，臂在当前高度平移
    DESCEND  平面距离 < ALIGN_M 后竖直下探到 --grasp-z        （脚本）
    GRASP    合爪、抬起                                          （脚本）

为什么这么切：`bench_snake.py` 量过，PlayJev 在渲染画面上选方向是 100%，
但画面里没有高度，而「什么时候该停、该合爪」在 RobotJev 里已经量过 ——
这类门控交给模型不收敛，按距离判才稳。

**画面是自适应缩放的。** 固定用整个工作空间当棋盘，y 方向一格就是 4.3 cm，
最后几厘米根本对不准。所以每一步都以夹爪和目标的中点为中心、按两者距离决定
画面范围：离得远看全局，离得近放大。模型每一步看到的都是「白块和红块隔着几格」，
方向信息保留，精度随距离提高。

位置一律用**指尖**（法兰 + 示范里拟合出的偏移），`--grasp-z` 是指尖高度。
示范里抓套筒上的乒乓球时指尖在 z≈0.211，套筒顶约在 0.19。真机不给这个参数不让跑。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "算法" / "DexGraspVLA-" / "SO101"))

from . import actions as A                                        # noqa: E402
from .probe_playjev import ask                                     # noqa: E402
from .snake_render import (INSTRUCTIONS, OPTIONS, SNAKE_TO_ACTION,  # noqa: E402
                           Board, render)

logger = logging.getLogger("robojev.playjev")

CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
ALIGN_M = 0.008          # 平面距离小于这个就算对准，开始下探
MIN_HALF_M = 0.03        # 自适应画面的最小半边长，太小了两块会叠在一起
# 单轴死区 = 半个细步。小于它的轴向偏差在画面里画成「同一行/同一列」。
# 不加这个会卡死：第一版在 d=1.6cm 处来回抖了 100 多步 —— x 只剩 2.5mm，
# 模型选 up/down（也算对，都能缩短距离），5mm 一步走过头，下一帧又翻到另一边，
# 于是永远在 x 上来回，y 那 1.6cm 一直没收。臂分辨不了半步以下的差，就告诉模型
# 这个轴已经对齐了，画面里只剩一个正确方向。
AXIS_DEADBAND_M = A.FINE_M / 2


def settle(env, gripper: float = 0.0, timeout_s: float = 2.0, tol_m: float = 0.001) -> bool:
    """每 50ms 重发同一个目标，直到臂**真的到了、停了**，再去问模型。

    RT 发送线程的规矩：**臂在动的时候，0.25 s 收不到新目标就刹车并锁存**，之后所有目标
    一律拒收（env.stats 里记成 unreachable，很容易误判成 IK 问题）。PlayJev 在 CPU 上
    想一次 250ms，一步 12mm 又走不完 0.25s，所以每步之后必须先让臂停稳。

    踩过两次：
    1. 什么都不做 -> 第一步之后就锁存，40 步 39 步被拒。
    2. 只调一次 `hold_until_rest` -> 目标刚发出去时跟踪器速度还是 0，它当场判「已停稳」
       返回，臂随后加速，照样锁存。
    所以这里判的是「实测位置追上了指令位置」，并且在等的过程中持续重发同一目标
    （零增量 env.step，目标不变，走的是 env 的全部安全检查）。
    """
    if not getattr(env.arm, "_rt_active", False):
        return True
    from . import actions as _A

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        time.sleep(0.05)
        env.step(_A.hold(gripper))
        tgt = env._target_pose
        cur = np.asarray(env.arm.get_ee_pose(), dtype=np.float64)
        v = getattr(env.arm, "_v_cmd", None)
        still = v is None or float(np.max(np.abs(v))) < 1e-3
        if tgt is not None and np.linalg.norm(cur[:3] - tgt[:3]) < tol_m and still:
            return True
    logger.warning("settle: 2s 内没停稳")
    return False


def adaptive_board(g_xy, t_xy) -> Board:
    """以两点中点为中心，范围随距离缩放，保证两点都在画面里且隔开几格。"""
    c = (np.asarray(g_xy) + np.asarray(t_xy)) / 2.0
    half = max(float(np.max(np.abs(np.asarray(g_xy) - np.asarray(t_xy)))) * 0.75, MIN_HALF_M)
    return Board(x_min=c[0] - half, x_max=c[0] + half, y_min=c[1] - half, y_max=c[1] + half)


class GripperXYZ:
    """夹爪**指尖**（两指夹持中心）的位置。真机：读关节自己算 FK 再加指尖偏移。

    指尖 = 法兰 + R_flange @ t。t 是从示范里拟合出来的（homography.json 的
    tip_offset_flange，约 [12,-40,108] mm）。夹爪斜 50~60° 装着接近物体，指尖比法兰
    往前伸十几厘米 —— 用法兰去对准目标会整整偏一个夹爪长度。
    mock：mock 臂没有 DH 表，用它报的位姿，不加偏移。
    """

    def __init__(self, env, mock: bool):
        self.env = env
        self.kin = None
        self.t = np.zeros(3)
        if not mock:
            from atom_robot.ar5.kinematics import AR5Kinematics
            self.kin = AR5Kinematics.from_robot(env.arm)
            if CALIB.exists():
                self.t = np.asarray(json.loads(CALIB.read_text()).get("tip_offset_flange", [0, 0, 0]), float)

    def __call__(self) -> np.ndarray:
        if self.kin is not None:
            q = np.asarray(self.env.arm.get_joint_positions(), dtype=np.float64)
            T = self.kin.fk(q)
            return T[:3, 3] + T[:3, :3] @ self.t
        return np.asarray(self.env.arm.get_ee_pose(), dtype=np.float64)[:3]


class CameraTarget:
    """俯视相机 + 颜色分割 + 单应性，给出目标在基座系的 xy。"""

    def __init__(self, color: str, pixel_offset=(0.0, 0.0), n_frames: int = 15):
        """开始时连看 n_frames 帧取中位数，之后**冻结**。

        为什么冻结：夹爪接近、下探时会挡住俯视画面里的目标（示范的合爪帧里，套筒
        整个被夹爪盖住）。物体在被抓之前不会动，所以一开始定一次就够了。
        `pixel_offset`：单应性是拿「乒乓球中心」的像素拟合的，而球放在套筒上。
        对套筒本身，要先加上球相对套筒的像素偏移（示范中位 (7.5,-22.5)）再映射。
        """
        from .camera import TopCamera
        from .color_detector import ColorDetector

        if not CALIB.exists():
            raise SystemExit(
                f"没有 {CALIB}。像素 -> 基座 xy 的映射没标定，目标位置是算不出来的。")
        d = json.loads(CALIB.read_text())
        self.Hm = np.array(d["H"], dtype=np.float64)
        det = ColorDetector(color)
        pts = []
        with TopCamera() as cam:
            time.sleep(1.0)
            for _ in range(n_frames * 3):
                f = cam.read()
                dets = det.detect(f.color) if f.ok else []
                if dets:
                    pts.append(dets[0].centre)
                if len(pts) >= n_frames:
                    break
                time.sleep(0.05)
        if len(pts) < 3:
            raise SystemExit(f"俯视相机里只看到 {len(pts)} 次 '{color}'，目标定不下来")
        uv = np.median(np.array(pts), axis=0) + np.asarray(pixel_offset, float)
        p = self.Hm @ np.array([uv[0], uv[1], 1.0])
        self.xy = p[:2] / p[2]
        self.uv = uv
        self.spread_px = float(np.max(np.std(np.array(pts), axis=0)))

    def __call__(self):
        return self.xy

    def close(self):
        pass


def run(a) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(message)s")
    if a.real and a.grasp_z is None:
        print("真机必须显式给 --grasp-z（指尖高度，米）。没有安全的默认值。")
        return 2

    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    cfg = AR5Config()
    cfg.mock = not a.real
    cfg.real_cameras_in_mock = False
    cfg.cameras.sources = {}
    if a.real:
        # 只在这次运行里开。config 里配好的是 suppleStop + 最不灵敏 0.01，触发了也不掉电。
        cfg.arm.collision_detection_enabled = True

    # **先定目标，再连臂。** 定目标要看 15 帧、约 2 秒；臂一连上 RT 就开始计时，
    # 刚连上时跟踪器还在收敛（有残余速度），这 2 秒不发目标就会锁存 ——
    # 第二版真机跑就是这样：循环还没开始就锁了，40 步里 39 步被拒。
    target_src = None
    if a.sim_target:
        fixed = np.array(a.sim_target, dtype=np.float64)
        get_target = lambda: fixed                                    # noqa: E731
        print(f"目标 : 合成 xy={fixed.tolist()}")
    else:
        target_src = CameraTarget(a.target_color, pixel_offset=a.pixel_offset)
        get_target = target_src
        print(f"目标 : 俯视相机 '{a.target_color}' 像素{np.round(target_src.uv).tolist()} "
              f"(抖动 {target_src.spread_px:.1f}px) -> 指尖要到 xy={np.round(target_src.xy, 3).tolist()}（已冻结）")

    env = AR5Env(cfg)
    env.connect()
    settle(env)

    gxyz = GripperXYZ(env, mock=not a.real)
    settle(env)
    if getattr(env.arm, "latched", False):
        print("RT 已经锁存（连上之后就没收到目标），不往下跑。")
        env.close()
        return 3
    grasp_z = a.grasp_z if a.grasp_z is not None else 0.15
    g0 = gxyz()
    print(f"机械臂: {'**真机**' if a.real else 'mock'}   当前指尖 {np.round(g0, 3).tolist()}   抓取指尖高度 {grasp_z:.3f} m\n")

    out_dir = Path(a.frames) if a.frames else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    phase = "ALIGN"
    t0 = time.perf_counter()
    try:
        for step in range(a.max_steps):
            g = gxyz()
            t = get_target()
            if t is None:
                print(f"[{step:3d}] 目标没看到，原地等")
                time.sleep(a.period)
                continue
            off = t - g[:2]
            dist = float(np.linalg.norm(off))

            if phase == "ALIGN":
                if dist < ALIGN_M:
                    print(f"[{step:3d}] 平面距离 {dist * 1000:.1f} mm < {ALIGN_M * 1000:.0f} mm，对准了")
                    if a.align_only:
                        phase = "ALIGNED"
                        break
                    phase = "DESCEND"
                    print("      开始下探")
                    continue
                t_draw = t.copy()
                for k in range(2):
                    if abs(off[k]) < AXIS_DEADBAND_M:
                        t_draw[k] = g[k]
                board = adaptive_board(g[:2], t_draw)
                img, head, food = render(g[:2], t_draw, board)
                r = ask(img, INSTRUCTIONS, OPTIONS)
                ans = r["answers"]["move"]
                move = ans["choice"]
                # 粗细档看「选中那个轴」上的误差，不看总距离。按总距离选的第一版在
                # (0.35,0.30) 卡死：总距离 4.8cm 走 12mm 粗档，但模型选的 x 轴只差 6mm，
                # 一步翻到另一侧 6mm，下一帧反过来，y 上 14 格永远没人收。
                od = t_draw - g[:2]
                axis_err = abs(od[0]) if move in ("up", "down") else abs(od[1])
                scale = "big" if axis_err > A.COARSE_M else "small"
                act = A.lookup(SNAKE_TO_ACTION[move], scale)
                env.step(act.to_delta(gripper=0.0))
                settle(env, gripper=0.0)
                ok = ((move == "up" and od[0] > 0) or (move == "down" and od[0] < 0)
                      or (move == "left" and od[1] > 0) or (move == "right" and od[1] < 0))
                print(f"[{step:3d}] ALIGN  d={dist * 100:5.1f}cm  蛇头{head} 食物{food}  "
                      f"PlayJev={move:5s} p={max(ans['probabilities'].values()):.2f}  "
                      f"-> {act.name:14s} {'✓' if ok else '✗ 方向错'}  {r['timing']['forward_ms']:.0f}ms")
                if out_dir:
                    import cv2
                    cv2.imwrite(str(out_dir / f"{step:03d}_{move}.png"), img)
                log.append({"step": step, "phase": phase, "gripper": g.tolist(), "target": t.tolist(),
                            "dist": dist, "move": move, "probs": ans["probabilities"], "correct": bool(ok)})

            elif phase == "DESCEND":
                if g[2] <= grasp_z + 0.002:
                    phase = "GRASP"
                    continue
                dz = max(-A.COARSE_M, grasp_z - g[2])
                env.step(np.array([0, 0, dz, 0, 0, 0, 0.0]))
                settle(env, gripper=0.0)
                print(f"[{step:3d}] DESCEND z={g[2] * 100:.1f}cm -> 目标 {grasp_z * 100:.1f}cm")

            elif phase == "GRASP":
                # 原地合爪。TG9801 手册：闭合要 **1.6 s**。第一版只等了 0.4s 就开始抬，
                # 爪还没合上臂就走了。等 2s，期间每 50ms 重发同一目标，别让 RT 锁存。
                for _ in range(40):
                    env.step(A.hold(1.0))
                    time.sleep(0.05)
                print(f"[{step:3d}] GRASP  合爪，开度 {env.hand.get_opening():.2f}")
                for _ in range(6):
                    env.step(A.lookup("up", "big").to_delta(gripper=1.0))
                    settle(env, gripper=1.0)
                print(f"[{step:3d}] LIFT   抬起 {6 * A.COARSE_M * 100:.0f} cm")
                phase = "DONE"
                break

            time.sleep(a.period)
    finally:
        if target_src is not None:
            target_src.close()
        env.close()

    aligned = [x for x in log if x["phase"] == "ALIGN"]
    n_ok = sum(x["correct"] for x in aligned)
    print(f"\n结束阶段 {phase}，{len(log)} 步 PlayJev 决策 / {time.perf_counter() - t0:.1f}s")
    if aligned:
        print(f"PlayJev 方向正确 {n_ok}/{len(aligned)} ({100 * n_ok / len(aligned):.0f}%)")
    print(f"env 统计: {env.stats}")
    if a.log:
        Path(a.log).write_text(json.dumps(log, ensure_ascii=False, indent=2))
    return 0 if phase in ("DONE", "ALIGNED") else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-target", type=float, nargs=2, metavar=("X", "Y"), help="合成目标 xy（米）")
    p.add_argument("--target-color", default="blue")
    p.add_argument("--real", action="store_true", help="驱动真臂")
    p.add_argument("--grasp-z", type=float, default=None, help="抓取时的指尖高度（米）；真机必须给")
    p.add_argument("--pixel-offset", type=float, nargs=2, default=[7.5, -22.5],
                   help="目标像素加这个偏移再映射。默认=球相对套筒的偏移（单应性是拿球心拟合的）")
    p.add_argument("--align-only", action="store_true", help="只做平面对准，对准后停下（不下探不合爪）")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--period", type=float, default=0.05)
    p.add_argument("--frames", default="", help="把每一步喂给 PlayJev 的画面存到这个目录")
    p.add_argument("--log", default="")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
