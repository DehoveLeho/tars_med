from dataclasses import dataclass
from typing import List
import numpy as np
import mujoco


@dataclass(frozen=True)
class IKConfig:
    damping: float = 0.015
    step_size: float = 0.8
    max_dq: float = 0.06
    tolerance: float = 0.002
    max_iterations: int = 30
    ee_offset_x: float = 0.06
    ee_offset_y: float = 0.0
    ee_offset_z: float = 0.0


@dataclass(frozen=True)
class JointLimits:
    min_pos: float
    max_pos: float


class DampedLeastSquaresIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joint_names: List[str],
        ee_body_name: str,
        config: IKConfig,
    ) -> None:
        self._model: mujoco.MjModel = model
        self._data: mujoco.MjData = data
        self._config: IKConfig = config
        self._ee_offset: np.ndarray = np.array(
            [config.ee_offset_x, config.ee_offset_y, config.ee_offset_z],
            dtype=np.float64,
        )
        self._ee_body_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )
        if self._ee_body_id < 0:
            raise ValueError(f"Body {ee_body_name} not found in model")

        self._joint_ids: List[int] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in arm_joint_names
        ]
        for name, jid in zip(arm_joint_names, self._joint_ids):
            if jid < 0:
                raise ValueError(f"Joint {name} not found in model")

        self._qpos_indices: List[int] = [
            int(model.jnt_qposadr[jid]) for jid in self._joint_ids
        ]
        self._dof_indices: List[int] = [
            int(model.jnt_dofadr[jid]) for jid in self._joint_ids
        ]
        self._joint_limits: List[JointLimits] = [
            JointLimits(
                min_pos=float(model.jnt_range[jid][0]),
                max_pos=float(model.jnt_range[jid][1]),
            )
            for jid in self._joint_ids
        ]
        self._q_nominal: np.ndarray = np.array(
            [-0.35, -0.25, 0.0, 0.70, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._k_null: np.ndarray = np.array(
            [0.15, 0.15, 0.10, 0.10, 0.04, 0.04, 0.04],
            dtype=np.float64,
        )
        self._eye3: np.ndarray = np.eye(3, dtype=np.float64)
        self._eye7: np.ndarray = np.eye(7, dtype=np.float64)
        self._jacp: np.ndarray = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr: np.ndarray = np.zeros((3, model.nv), dtype=np.float64)

    def get_ee_position(self) -> np.ndarray:
        r_mat: np.ndarray = self._data.xmat[self._ee_body_id].reshape((3, 3))
        return np.array(
            self._data.xpos[self._ee_body_id] + r_mat @ self._ee_offset,
            dtype=np.float64,
        )

    def get_qpos_indices(self) -> List[int]:
        return list(self._qpos_indices)

    def get_joint_ids(self) -> List[int]:
        return list(self._joint_ids)

    def solve(
        self,
        target_pos_world: np.ndarray,
        current_q_des: np.ndarray,
    ) -> np.ndarray:
        q_result: np.ndarray = np.array(current_q_des, dtype=np.float64).copy()
        lambda_sq: float = self._config.damping ** 2

        for _ in range(self._config.max_iterations):
            for i, q_idx in enumerate(self._qpos_indices):
                self._data.qpos[q_idx] = q_result[i]
            mujoco.mj_forward(self._model, self._data)

            current_pos: np.ndarray = self.get_ee_position()
            error: np.ndarray = target_pos_world - current_pos
            if float(np.linalg.norm(error)) < self._config.tolerance:
                break

            mujoco.mj_jac(
                self._model,
                self._data,
                self._jacp,
                self._jacr,
                current_pos,
                self._ee_body_id,
            )
            j_arm: np.ndarray = self._jacp[:, self._dof_indices]
            j_jt: np.ndarray = j_arm @ j_arm.T + lambda_sq * self._eye3
            j_pinv: np.ndarray = j_arm.T @ np.linalg.solve(j_jt, self._eye3)

            dq_null: np.ndarray = self._k_null * (self._q_nominal - q_result)
            dq: np.ndarray = (
                j_pinv @ error * self._config.step_size
                + (self._eye7 - j_pinv @ j_arm) @ dq_null
            )

            dq_clamped: np.ndarray = np.clip(
                dq, -self._config.max_dq, self._config.max_dq
            )
            for i in range(len(q_result)):
                q_val: float = float(q_result[i] + dq_clamped[i])
                limits: JointLimits = self._joint_limits[i]
                q_result[i] = float(
                    np.clip(q_val, limits.min_pos, limits.max_pos)
                )

        return q_result
