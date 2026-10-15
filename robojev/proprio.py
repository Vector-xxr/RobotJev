"""本体特权信息：读关节角，自己算 FK 得到夹爪的笛卡尔坐标。

为什么不直接用 `arm.get_ee_pose()`：那是控制器算好报上来的，而这台臂的逆解
是我们自己用 PoE 解的（xMate 模型库不支持 AR5）。自己再算一遍 FK 并和控制器
对账，是确认「我们认为的末端位置」和「臂实际认为的」是同一个东西的唯一办法 ——
两者不一致时，后面喂给模型的坐标就是错的，而且错得很安静。

`from_robot()` 读的是控制器里**标定过**的 DH 表，不是出厂名义值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Proprio:
    q_deg: np.ndarray          # 7 个关节角，度
    xyz: np.ndarray            # 法兰原点，基座系，米（自己算的 FK）
    z_axis: np.ndarray         # 法兰 z 轴在基座系的朝向（夹爪指向）
    ctrl_xyz: np.ndarray       # 控制器报的，用来对账
    gripper: float             # 0=张开 1=闭合

    @property
    def fk_error_mm(self) -> float:
        return float(np.linalg.norm(self.xyz - self.ctrl_xyz) * 1000)

    def verbalize(self) -> str:
        """写成一句话。数字保留，但配上方向词 —— Laya 那边量过，纯数字它读不懂。"""
        x, y, z = self.xyz
        down = self.z_axis[2] < -0.9
        return (f"The gripper is at x={x * 100:.1f} cm (forward from the robot base), "
                f"y={y * 100:+.1f} cm ({'left' if y > 0 else 'right'} of the base), "
                f"z={z * 100:.1f} cm (height above the base). "
                f"It is {'pointing straight down' if down else 'tilted, not pointing straight down'} "
                f"and {'open' if self.gripper < 0.5 else 'closed'}.")


class ProprioReader:
    def __init__(self, env):
        from atom_robot.ar5.kinematics import AR5Kinematics

        self.env = env
        self.kin = AR5Kinematics.from_robot(env.arm)

    def read(self) -> Proprio:
        q = np.asarray(self.env.arm.get_joint_positions(), dtype=np.float64)
        T = self.kin.fk(q)
        return Proprio(q_deg=np.rad2deg(q), xyz=T[:3, 3].copy(), z_axis=T[:3, 2].copy(),
                       ctrl_xyz=np.asarray(self.env.arm.get_ee_pose(), dtype=np.float64)[:3],
                       gripper=float(self.env.hand.get_opening()))
