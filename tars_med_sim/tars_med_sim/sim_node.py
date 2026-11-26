import contextlib
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from projet_interfaces.action import ExecuteGrasp
from projet_interfaces.msg import ErrorCode, ObjectPose
from rclpy._rclpy_pybind11 import RCLError
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from tars_med_sim.ik_solver import DampedLeastSquaresIK, IKConfig


@dataclass(frozen=True)
class SimNodeConfig:
    scene_xml_path: str = ""
    enable_viewer: bool = True
    sim_rate_hz: float = 100.0
    state_publish_rate_hz: float = 50.0
    substeps_per_cycle: int = 5
    base_body_name: str = "pelvis"
    ee_body_name: str = "right_wrist_yaw_link"
    kp_lower_body: float = 300.0
    kd_lower_body: float = 12.0
    kp_waist: float = 220.0
    kd_waist: float = 10.0
    kp_left_arm: float = 100.0
    kd_left_arm: float = 5.0
    kp_right_arm: float = 150.0
    kd_right_arm: float = 8.0


@dataclass(frozen=True)
class ActuatorControlMapping:
    actuator_id: int
    joint_id: int
    qpos_adr: int
    dof_adr: int
    kp: float
    kd: float
    ctrl_min: float
    ctrl_max: float
    default_qpos: float
    is_right_arm: bool
    right_arm_index: int


def _matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    tr: float = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s: float = float(np.sqrt(tr + 1.0) * 2.0)
        qw: float = 0.25 * s
        qx: float = (r[2, 1] - r[1, 2]) / s
        qy: float = (r[0, 2] - r[2, 0]) / s
        qz: float = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = float(np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0)
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = float(np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0)
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0)
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q_norm: float = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if q_norm > 1e-9:
        qw /= q_norm
        qx /= q_norm
        qy /= q_norm
        qz /= q_norm
    return np.array([qx, qy, qz, qw], dtype=np.float64)


class SimNode(Node):
    def __init__(self, config: SimNodeConfig, ik_config: IKConfig) -> None:
        super().__init__("sim_node")
        self._config: SimNodeConfig = config
        self._ik_config: IKConfig = ik_config
        self._lock: threading.Lock = threading.Lock()
        self._viewer: Optional[Any] = None

        scene_path: str = self._resolve_scene_path(config.scene_xml_path)
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(scene_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self._arm_joint_names: List[str] = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self._base_body_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, self._config.base_body_name
        )
        if self._base_body_id < 0:
            raise ValueError(f"Base body {self._config.base_body_name} not found")

        self._ee_body_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, self._config.ee_body_name
        )
        if self._ee_body_id < 0:
            raise ValueError(f"EE body {self._config.ee_body_name} not found")

        self._ee_geom_ids: List[int] = [
            i
            for i in range(self._model.ngeom)
            if self._model.geom_bodyid[i] == self._ee_body_id
        ]

        self._proxy_object_names: List[str] = ["scalpel", "tourniquet", "bottle"]
        self._proxy_object_ids: Dict[str, int] = {}
        for name in self._proxy_object_names:
            oid: int = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if oid >= 0:
                self._proxy_object_ids[name] = oid

        self._weld_ids: Dict[str, int] = {}
        for name in self._proxy_object_names:
            wid: int = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{name}"
            )
            if wid >= 0:
                self._weld_ids[name] = wid
        self._grasped_object_id: Optional[str] = None

        self._actuator_mappings: List[ActuatorControlMapping] = (
            self._setup_actuator_mappings()
        )

        self._ik_solver: DampedLeastSquaresIK = DampedLeastSquaresIK(
            model=self._model,
            data=self._data,
            arm_joint_names=self._arm_joint_names,
            ee_body_name=self._config.ee_body_name,
            config=self._ik_config,
        )

        initial_ee_pos: np.ndarray = self._ik_solver.get_ee_position()
        self._target_ee_pos_world: np.ndarray = initial_ee_pos.copy()

        self._right_arm_q_des: np.ndarray = np.zeros(
            len(self._arm_joint_names), dtype=np.float64
        )
        for i, jname in enumerate(self._arm_joint_names):
            jid: int = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, jname
            )
            qadr: int = int(self._model.jnt_qposadr[jid])
            self._right_arm_q_des[i] = float(self._data.qpos[qadr])

        self._gripper_ratio: float = 0.0
        self._measured_force: float = 0.0

        self._key_queue: queue.Queue[int] = queue.Queue(maxsize=32)
        self._demo_active: bool = False
        self._demo_waypoints: List[Tuple[float, float, float, float]] = []
        self._demo_wp_idx: int = 0
        self._demo_step: int = 0
        self._demo_steps_per_wp: int = 40
        self._demo_start_pt: np.ndarray = np.zeros(3, dtype=np.float64)
        self._demo_start_grip: float = 0.0

        self._joint_names_all: List[str] = [
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, m.joint_id)
            for m in self._actuator_mappings
        ]

        if self._config.enable_viewer:
            try:
                self._viewer = mujoco.viewer.launch_passive(
                    self._model, self._data, key_callback=self._on_key
                )
                self._setup_viewer_overlay()
            except (RuntimeError, ValueError) as viewer_err:
                self.get_logger().warn(
                    f"Passive viewer launch skipped: {viewer_err}"
                )
                self._viewer = None

        self._action_cb_group: ReentrantCallbackGroup = ReentrantCallbackGroup()
        self._sub_cb_group: MutuallyExclusiveCallbackGroup = (
            MutuallyExclusiveCallbackGroup()
        )

        self._ee_target_sub = self.create_subscription(
            PointStamped,
            "/teleop/ee_target",
            self._ee_target_callback,
            10,
            callback_group=self._sub_cb_group,
        )
        self._gripper_cmd_sub = self.create_subscription(
            Float32,
            "/teleop/gripper_cmd",
            self._gripper_cmd_callback,
            10,
            callback_group=self._sub_cb_group,
        )

        self._joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10
        )
        self._object_pose_pub = self.create_publisher(
            ObjectPose, "/perception/object_pose", 10
        )

        self._action_server: ActionServer = ActionServer(
            self,
            ExecuteGrasp,
            "/manipulation/execute_grasp",
            execute_callback=self._execute_grasp_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._action_cb_group,
        )

        sim_period: float = 1.0 / self._config.sim_rate_hz
        self._sim_timer = self.create_timer(
            sim_period, self._sim_step_timer_callback
        )

        publish_period: float = 1.0 / self._config.state_publish_rate_hz
        self._pub_timer = self.create_timer(
            publish_period, self._publish_states_timer_callback
        )

    @contextlib.contextmanager
    def _guard_data(self) -> Generator[None, None, None]:
        v_lock = (
            self._viewer.lock()
            if self._viewer is not None
            else contextlib.nullcontext()
        )
        with self._lock, v_lock:
            yield

    def _resolve_scene_path(self, scene_xml_arg: str) -> str:
        if scene_xml_arg and os.path.exists(scene_xml_arg):
            return scene_xml_arg

        ws_scene: str = os.path.expanduser(
            "~/tars_med_ws/src/tars_med/tars_med_sim/models/tars_med_scene.xml"
        )
        if os.path.exists(ws_scene):
            self._ensure_meshes_symlink(os.path.dirname(ws_scene))
            return ws_scene

        raise FileNotFoundError(f"Could not find scene XML at {ws_scene}")

    def _ensure_meshes_symlink(self, models_dir: str) -> None:
        target_meshes: str = os.path.expanduser(
            "~/tars_med_ws/src/unitree_mujoco/unitree_robots/g1/meshes"
        )
        link_path: str = os.path.join(models_dir, "meshes")
        if not os.path.exists(link_path) and os.path.exists(target_meshes):
            try:
                os.symlink(target_meshes, link_path)
            except (OSError, FileExistsError):
                pass

    def _setup_actuator_mappings(self) -> List[ActuatorControlMapping]:
        mappings: List[ActuatorControlMapping] = []
        for a_id in range(self._model.nu):
            j_id: int = int(self._model.actuator_trnid[a_id, 0])
            j_name: str = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, j_id
            )
            qpos_adr: int = int(self._model.jnt_qposadr[j_id])
            dof_adr: int = int(self._model.jnt_dofadr[j_id])
            ctrl_min: float = float(self._model.actuator_ctrlrange[a_id, 0])
            ctrl_max: float = float(self._model.actuator_ctrlrange[a_id, 1])
            default_q: float = float(self._data.qpos[qpos_adr])

            is_right: bool = j_name in self._arm_joint_names
            right_idx: int = (
                self._arm_joint_names.index(j_name) if is_right else -1
            )

            kp: float = self._config.kp_lower_body
            kd: float = self._config.kd_lower_body
            if "waist" in j_name:
                kp = self._config.kp_waist
                kd = self._config.kd_waist
            elif "left_shoulder" in j_name or "left_elbow" in j_name or "left_wrist" in j_name:
                kp = self._config.kp_left_arm
                kd = self._config.kd_left_arm
            elif is_right:
                kp = self._config.kp_right_arm
                kd = self._config.kd_right_arm

            mappings.append(
                ActuatorControlMapping(
                    actuator_id=a_id,
                    joint_id=j_id,
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                    kp=kp,
                    kd=kd,
                    ctrl_min=ctrl_min,
                    ctrl_max=ctrl_max,
                    default_qpos=default_q,
                    is_right_arm=is_right,
                    right_arm_index=right_idx,
                )
            )
        return mappings

    def _ee_target_callback(self, msg: PointStamped) -> None:
        with self._guard_data():
            p_in: np.ndarray = np.array(
                [msg.point.x, msg.point.y, msg.point.z], dtype=np.float64
            )
            if msg.header.frame_id == self._config.base_body_name:
                p_base: np.ndarray = self._data.xpos[self._base_body_id]
                r_base: np.ndarray = self._data.xmat[
                    self._base_body_id
                ].reshape((3, 3))
                self._target_ee_pos_world = p_base + r_base @ p_in
            else:
                self._target_ee_pos_world = p_in

    def _gripper_cmd_callback(self, msg: Float32) -> None:
        with self._guard_data():
            self._gripper_ratio = float(np.clip(msg.data, 0.0, 1.0))

    def _sim_step_timer_callback(self) -> None:
        if not rclpy.ok():
            return
        with self._guard_data():
            self._process_keys()
            if self._demo_active:
                self._step_demo()

            target_pos: np.ndarray = self._target_ee_pos_world.copy()
            current_q_des: np.ndarray = self._right_arm_q_des.copy()

            solved_q: np.ndarray = self._ik_solver.solve(target_pos, current_q_des)
            self._right_arm_q_des = solved_q.copy()

            for _ in range(self._config.substeps_per_cycle):
                for m in self._actuator_mappings:
                    q_cur: float = float(self._data.qpos[m.qpos_adr])
                    v_cur: float = float(self._data.qvel[m.dof_adr])
                    target_q: float = (
                        self._right_arm_q_des[m.right_arm_index]
                        if m.is_right_arm
                        else m.default_qpos
                    )
                    tau: float = m.kp * (target_q - q_cur) - m.kd * v_cur
                    clamped_tau: float = float(
                        np.clip(tau, m.ctrl_min, m.ctrl_max)
                    )
                    self._data.ctrl[m.actuator_id] = clamped_tau

                mujoco.mj_step(self._model, self._data)

            self._measured_force = self._compute_ee_contact_force()

            if self._gripper_ratio > 0.6 and self._grasped_object_id is None:
                cur_ee_pos: np.ndarray = self._ik_solver.get_ee_position()
                best_name: Optional[str] = None
                best_dist: float = 0.12
                for name, oid in self._proxy_object_ids.items():
                    d_obj: float = float(np.linalg.norm(cur_ee_pos - self._data.xpos[oid]))
                    if d_obj < best_dist:
                        best_dist = d_obj
                        best_name = name
                if best_name is not None:
                    self._attach_object(best_name)
            elif self._gripper_ratio < 0.2 and self._grasped_object_id is not None:
                self._detach_object()
            elif self._grasped_object_id is not None and self._measured_force > 18.0:
                self._detach_object()
                self._gripper_ratio = 0.0

        if rclpy.ok() and self._viewer is not None and self._viewer.is_running():
            try:
                self._viewer.sync()
            except (RuntimeError, ValueError, OSError):
                pass

    def _compute_ee_contact_force(self) -> float:
        total_force: float = 0.0
        c_force: np.ndarray = np.zeros(6, dtype=np.float64)
        for i in range(self._data.ncon):
            con = self._data.contact[i]
            if (
                con.geom1 in self._ee_geom_ids
                or con.geom2 in self._ee_geom_ids
            ):
                mujoco.mj_contactForce(self._model, self._data, i, c_force)
                total_force += float(np.abs(c_force[0]))
        if self._gripper_ratio > 0.05:
            ee_pos: np.ndarray = self._ik_solver.get_ee_position()
            for oid in self._proxy_object_ids.values():
                obj_pos: np.ndarray = self._data.xpos[oid]
                dist: float = float(np.linalg.norm(ee_pos - obj_pos))
                if dist < 0.12:
                    total_force += self._gripper_ratio * 12.0
                    break
        return total_force

    def _publish_states_timer_callback(self) -> None:
        now = self.get_clock().now().to_msg()
        with self._guard_data():
            js_msg = JointState()
            js_msg.header.stamp = now
            js_msg.name = list(self._joint_names_all)
            positions: List[float] = []
            velocities: List[float] = []
            efforts: List[float] = []
            for m in self._actuator_mappings:
                positions.append(float(self._data.qpos[m.qpos_adr]))
                velocities.append(float(self._data.qvel[m.dof_adr]))
                efforts.append(float(self._data.ctrl[m.actuator_id]))
            js_msg.position = positions
            js_msg.velocity = velocities
            js_msg.effort = efforts
            self._joint_state_pub.publish(js_msg)

            p_base: np.ndarray = self._data.xpos[self._base_body_id].copy()
            r_base: np.ndarray = self._data.xmat[
                self._base_body_id
            ].reshape((3, 3)).copy()

            for name, oid in self._proxy_object_ids.items():
                p_obj: np.ndarray = self._data.xpos[oid]
                r_obj: np.ndarray = self._data.xmat[oid].reshape((3, 3))

                p_rel: np.ndarray = r_base.T @ (p_obj - p_base)
                r_rel: np.ndarray = r_base.T @ r_obj
                q_rel: np.ndarray = _matrix_to_quaternion(r_rel)

                obj_msg = ObjectPose()
                obj_msg.object_id = name
                obj_msg.confidence = 1.0
                obj_msg.pose.header.stamp = now
                obj_msg.pose.header.frame_id = self._config.base_body_name
                obj_msg.pose.pose.position.x = float(p_rel[0])
                obj_msg.pose.pose.position.y = float(p_rel[1])
                obj_msg.pose.pose.position.z = float(p_rel[2])
                obj_msg.pose.pose.orientation.x = float(q_rel[0])
                obj_msg.pose.pose.orientation.y = float(q_rel[1])
                obj_msg.pose.pose.orientation.z = float(q_rel[2])
                obj_msg.pose.pose.orientation.w = float(q_rel[3])
                self._object_pose_pub.publish(obj_msg)

    def _attach_object(self, object_id: str) -> None:
        if object_id not in self._weld_ids or object_id not in self._proxy_object_ids:
            return
        eq_id: int = self._weld_ids[object_id]
        oid: int = self._proxy_object_ids[object_id]
        p1: np.ndarray = self._data.xpos[self._ee_body_id]
        p2: np.ndarray = self._data.xpos[oid]
        r1: np.ndarray = self._data.xmat[self._ee_body_id].reshape((3, 3))
        r2: np.ndarray = self._data.xmat[oid].reshape((3, 3))
        rel_p: np.ndarray = r1.T @ (p2 - p1)
        rel_r: np.ndarray = r1.T @ r2
        q_rel: np.ndarray = _matrix_to_quaternion(rel_r)
        self._model.eq_data[eq_id, 3:6] = rel_p
        self._model.eq_data[eq_id, 6:10] = np.array(
            [q_rel[3], q_rel[0], q_rel[1], q_rel[2]], dtype=np.float64
        )
        self._data.eq_active[eq_id] = 1
        self._grasped_object_id = object_id

    def _detach_object(self) -> None:
        for eq_id in self._weld_ids.values():
            self._data.eq_active[eq_id] = 0
        self._grasped_object_id = None

    def _setup_viewer_overlay(self) -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        ctrl_line: str = (
            "Arrows: X/Y | W/S: Z | Space: Grip | D: Handover Demo | R: Release | H: Home"
        )
        try:
            self._viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "T.A.R.S.-Med Teleop",
                        ctrl_line,
                    )
                ]
            )
        except (RuntimeError, ValueError):
            pass

    def _on_key(self, keycode: int) -> None:
        try:
            self._key_queue.put_nowait(keycode)
        except queue.Full:
            pass

    def _start_demo(self) -> None:
        p_base: np.ndarray = self._data.xpos[self._base_body_id]
        r_base: np.ndarray = self._data.xmat[self._base_body_id].reshape((3, 3))
        p_rel: np.ndarray = r_base.T @ (self._target_ee_pos_world - p_base)
        self._demo_waypoints = [
            (0.34, -0.26, 0.16, 0.0),
            (0.34, -0.26, 0.03, 0.0),
            (0.34, -0.26, 0.03, 1.0),
            (0.34, -0.26, 0.20, 1.0),
            (0.42, -0.12, 0.16, 1.0),
            (0.42, -0.12, 0.16, 1.0),
            (0.42, -0.12, 0.16, 0.0),
            (0.32, -0.16, 0.08, 0.0),
        ]
        self._demo_wp_idx = 0
        self._demo_step = 0
        self._demo_start_pt = p_rel.copy()
        self._demo_start_grip = self._gripper_ratio
        self._demo_active = True

    def _step_demo(self) -> None:
        if not self._demo_active or self._demo_wp_idx >= len(self._demo_waypoints):
            self._demo_active = False
            return
        tx, ty, tz, grip = self._demo_waypoints[self._demo_wp_idx]
        self._demo_step += 1
        alpha: float = float(self._demo_step / self._demo_steps_per_wp)
        nx: float = float(self._demo_start_pt[0] + alpha * (tx - self._demo_start_pt[0]))
        ny: float = float(self._demo_start_pt[1] + alpha * (ty - self._demo_start_pt[1]))
        nz: float = float(self._demo_start_pt[2] + alpha * (tz - self._demo_start_pt[2]))
        ngrip: float = float(self._demo_start_grip + alpha * (grip - self._demo_start_grip))

        p_base: np.ndarray = self._data.xpos[self._base_body_id]
        r_base: np.ndarray = self._data.xmat[self._base_body_id].reshape((3, 3))
        self._target_ee_pos_world = p_base + r_base @ np.array([nx, ny, nz], dtype=np.float64)
        self._gripper_ratio = ngrip

        if self._demo_step >= self._demo_steps_per_wp:
            self._demo_wp_idx += 1
            self._demo_step = 0
            self._demo_start_pt = np.array([tx, ty, tz], dtype=np.float64)
            self._demo_start_grip = grip
            if self._demo_wp_idx >= len(self._demo_waypoints):
                self._demo_active = False

    def _process_keys(self) -> None:
        step: float = 0.02
        while not self._key_queue.empty():
            try:
                keycode: int = self._key_queue.get_nowait()
            except queue.Empty:
                break
            if keycode == 68:
                self._start_demo()
                continue

            self._demo_active = False
            p_base: np.ndarray = self._data.xpos[self._base_body_id]
            r_base: np.ndarray = self._data.xmat[self._base_body_id].reshape((3, 3))
            p_rel: np.ndarray = r_base.T @ (self._target_ee_pos_world - p_base)

            if keycode == 265:
                p_rel[0] = float(np.clip(p_rel[0] + step, 0.20, 0.46))
            elif keycode == 264:
                p_rel[0] = float(np.clip(p_rel[0] - step, 0.20, 0.46))
            elif keycode in (263, 65):
                p_rel[1] = float(np.clip(p_rel[1] + step, -0.36, 0.05))
            elif keycode in (262, 68):
                p_rel[1] = float(np.clip(p_rel[1] - step, -0.36, 0.05))
            elif keycode in (266, 87):
                p_rel[2] = float(np.clip(p_rel[2] + step, 0.01, 0.30))
            elif keycode in (267, 83):
                p_rel[2] = float(np.clip(p_rel[2] - step, 0.01, 0.30))
            elif keycode == 32:
                self._gripper_ratio = 1.0 if self._gripper_ratio < 0.5 else 0.0
            elif keycode == 72:
                p_rel = np.array([0.32, -0.16, 0.08], dtype=np.float64)
                self._gripper_ratio = 0.0
            elif keycode == 82:
                self._detach_object()
                self._gripper_ratio = 0.0
                p_rel = np.array([0.32, -0.16, 0.08], dtype=np.float64)
            elif keycode == 71:
                closest_obj_id: int = -1
                min_d: float = 1e9
                cur_ee: np.ndarray = self._ik_solver.get_ee_position()
                for oid in self._proxy_object_ids.values():
                    d: float = float(np.linalg.norm(cur_ee - self._data.xpos[oid]))
                    if d < min_d:
                        min_d = d
                        closest_obj_id = oid
                if closest_obj_id >= 0:
                    tgt_world: np.ndarray = self._data.xpos[closest_obj_id].copy()
                    p_rel = r_base.T @ (tgt_world - p_base)
                    p_rel[2] += 0.04
                    self._gripper_ratio = 1.0

            self._target_ee_pos_world = p_base + r_base @ p_rel

    def _goal_callback(
        self, goal_request: ExecuteGrasp.Goal
    ) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(
        self, goal_handle: ServerGoalHandle
    ) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_grasp_callback(
        self, goal_handle: ServerGoalHandle
    ) -> ExecuteGrasp.Result:
        result = ExecuteGrasp.Result()
        req: ExecuteGrasp.Goal = goal_handle.request

        with self._guard_data():
            p_base: np.ndarray = self._data.xpos[self._base_body_id].copy()
            r_base: np.ndarray = self._data.xmat[
                self._base_body_id
            ].reshape((3, 3)).copy()
            cur_ee: np.ndarray = self._ik_solver.get_ee_position().copy()

        tgt_pt: np.ndarray = np.array(
            [
                req.target_pose.pose.position.x,
                req.target_pose.pose.position.y,
                req.target_pose.pose.position.z,
            ],
            dtype=np.float64,
        )

        if float(np.linalg.norm(tgt_pt)) < 1e-4 and req.object_id in self._proxy_object_ids:
            with self._guard_data():
                tgt_pt_world = self._data.xpos[
                    self._proxy_object_ids[req.object_id]
                ].copy()
        elif req.target_pose.header.frame_id == self._config.base_body_name:
            tgt_pt_world = p_base + r_base @ tgt_pt
        else:
            tgt_pt_world = tgt_pt

        pre_grasp_world: np.ndarray = tgt_pt_world.copy()
        pre_grasp_world[2] += 0.06

        steps_pre: int = 25
        for step in range(steps_pre):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.error.code = ErrorCode.NONE
                result.error.source_block = "manipulation"
                result.error.message = "Grasp cancelled by client"
                return result

            alpha: float = float((step + 1) / steps_pre)
            interp_target: np.ndarray = (1.0 - alpha) * cur_ee + alpha * pre_grasp_world
            with self._guard_data():
                self._target_ee_pos_world = interp_target.copy()
                current_force = self._measured_force
                current_gripper = self._gripper_ratio

            feedback = ExecuteGrasp.Feedback()
            feedback.current_force_newton = float(current_force)
            feedback.gripper_closure_ratio = float(current_gripper)
            feedback.status_message = "Approaching pre-grasp pose"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.04)

        steps_lower: int = 15
        for step in range(steps_lower):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.error.code = ErrorCode.NONE
                result.error.source_block = "manipulation"
                result.error.message = "Grasp cancelled by client"
                return result

            alpha = float((step + 1) / steps_lower)
            interp_target = (1.0 - alpha) * pre_grasp_world + alpha * tgt_pt_world
            with self._guard_data():
                self._target_ee_pos_world = interp_target.copy()
                current_force = self._measured_force
                current_gripper = self._gripper_ratio

            feedback = ExecuteGrasp.Feedback()
            feedback.current_force_newton = float(current_force)
            feedback.gripper_closure_ratio = float(current_gripper)
            feedback.status_message = "Lowering to grasp pose"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.04)

        steps_grasp: int = 20
        max_f: float = max(5.0, float(req.max_force_newton))
        for step in range(steps_grasp):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.error.code = ErrorCode.NONE
                result.error.source_block = "manipulation"
                result.error.message = "Grasp cancelled by client"
                return result

            ratio: float = float((step + 1) / steps_grasp)
            with self._guard_data():
                self._gripper_ratio = ratio
                current_force = self._measured_force

            feedback = ExecuteGrasp.Feedback()
            feedback.current_force_newton = float(current_force)
            feedback.gripper_closure_ratio = float(ratio)
            feedback.status_message = "Closing gripper"
            goal_handle.publish_feedback(feedback)

            if current_force >= max_f:
                break
            time.sleep(0.04)

        with self._guard_data():
            if req.object_id in self._proxy_object_ids:
                self._attach_object(req.object_id)
            elif self._grasped_object_id is None:
                cur_ee_now: np.ndarray = self._ik_solver.get_ee_position()
                for name, oid in self._proxy_object_ids.items():
                    if float(np.linalg.norm(cur_ee_now - self._data.xpos[oid])) < 0.12:
                        self._attach_object(name)
                        break

        lift_target: np.ndarray = tgt_pt_world.copy()
        lift_target[2] += 0.12
        steps_lift: int = 20
        for step in range(steps_lift):
            if goal_handle.is_cancel_requested:
                with self._guard_data():
                    self._detach_object()
                    self._gripper_ratio = 0.0
                goal_handle.canceled()
                result.success = False
                result.error.code = ErrorCode.NONE
                result.error.source_block = "manipulation"
                result.error.message = "Grasp cancelled by client"
                return result
            alpha = float((step + 1) / steps_lift)
            interp_target = (1.0 - alpha) * tgt_pt_world + alpha * lift_target
            with self._guard_data():
                self._target_ee_pos_world = interp_target.copy()
            feedback = ExecuteGrasp.Feedback()
            feedback.current_force_newton = float(self._measured_force)
            feedback.gripper_closure_ratio = 1.0
            feedback.status_message = "Lifting instrument off table"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.03)

        handover_target: np.ndarray = p_base + r_base @ np.array([0.42, -0.12, 0.16], dtype=np.float64)
        steps_handover: int = 25
        for step in range(steps_handover):
            if goal_handle.is_cancel_requested:
                with self._guard_data():
                    self._detach_object()
                    self._gripper_ratio = 0.0
                goal_handle.canceled()
                result.success = False
                result.error.code = ErrorCode.NONE
                result.error.source_block = "manipulation"
                result.error.message = "Grasp cancelled by client"
                return result
            alpha = float((step + 1) / steps_handover)
            interp_target = (1.0 - alpha) * lift_target + alpha * handover_target
            with self._guard_data():
                self._target_ee_pos_world = interp_target.copy()
            feedback = ExecuteGrasp.Feedback()
            feedback.current_force_newton = float(self._measured_force)
            feedback.gripper_closure_ratio = 1.0
            feedback.status_message = "Delivering instrument forward for handover"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.03)

        steps_hold: int = 60
        for _ in range(steps_hold):
            if goal_handle.is_cancel_requested:
                break
            with self._guard_data():
                current_f = self._measured_force
                has_grasped = self._grasped_object_id is not None
            if current_f > 15.0 or not has_grasped:
                break
            feedback = ExecuteGrasp.Feedback()
            feedback.current_force_newton = float(current_f)
            feedback.gripper_closure_ratio = 1.0
            feedback.status_message = "Holding instrument forward, waiting for handover pull"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.05)

        with self._guard_data():
            self._detach_object()
            self._gripper_ratio = 0.0

        home_target: np.ndarray = p_base + r_base @ np.array([0.32, -0.16, 0.08], dtype=np.float64)
        steps_home: int = 20
        for step in range(steps_home):
            alpha = float((step + 1) / steps_home)
            interp_target = (1.0 - alpha) * handover_target + alpha * home_target
            with self._guard_data():
                self._target_ee_pos_world = interp_target.copy()
            time.sleep(0.03)

        with self._guard_data():
            final_ee: np.ndarray = self._ik_solver.get_ee_position().copy()
            final_force: float = self._measured_force
            r_rel: np.ndarray = r_base.T @ self._data.xmat[
                self._ee_body_id
            ].reshape((3, 3))
            p_rel: np.ndarray = r_base.T @ (final_ee - p_base)
            q_rel: np.ndarray = _matrix_to_quaternion(r_rel)

        actual_pose = PoseStamped()
        actual_pose.header.stamp = self.get_clock().now().to_msg()
        actual_pose.header.frame_id = self._config.base_body_name
        actual_pose.pose.position.x = float(p_rel[0])
        actual_pose.pose.position.y = float(p_rel[1])
        actual_pose.pose.position.z = float(p_rel[2])
        actual_pose.pose.orientation.x = float(q_rel[0])
        actual_pose.pose.orientation.y = float(q_rel[1])
        actual_pose.pose.orientation.z = float(q_rel[2])
        actual_pose.pose.orientation.w = float(q_rel[3])

        result.success = True
        result.error.code = ErrorCode.NONE
        result.error.source_block = "manipulation"
        result.error.message = "Grasp executed successfully"
        result.actual_pose = actual_pose
        result.measured_force_newton = float(final_force)

        goal_handle.succeed()
        return result

    def destroy_node(self) -> None:
        self._sim_timer.cancel()
        self._pub_timer.cancel()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except (RuntimeError, ValueError):
                pass
            time.sleep(0.3)
            self._viewer = None
        self._action_server.destroy()
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)

    sim_config = SimNodeConfig()
    ik_config = IKConfig()

    node = SimNode(config=sim_config, ik_config=ik_config)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        executor.shutdown(timeout_sec=0.5)
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (RuntimeError, RCLError):
                pass


if __name__ == "__main__":
    main()
