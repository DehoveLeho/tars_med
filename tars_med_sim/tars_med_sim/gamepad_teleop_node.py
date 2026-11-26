import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32


@dataclass(frozen=True)
class TeleopConfig:
    axis_left_x: int = 0
    axis_left_y: int = 1
    axis_right_y: int = 4
    axis_trigger: int = 5
    button_reset: int = 0
    deadzone: float = 0.05
    speed_scale_xy: float = 0.004
    speed_scale_z: float = 0.004
    publish_rate_hz: float = 50.0
    frame_id: str = "pelvis"
    home_x: float = 0.32
    home_y: float = -0.16
    home_z: float = 0.08
    min_x: float = 0.20
    max_x: float = 0.46
    min_y: float = -0.36
    max_y: float = 0.05
    min_z: float = 0.01
    max_z: float = 0.25


def _apply_deadzone(val: float, deadzone: float) -> float:
    if abs(val) < deadzone:
        return 0.0
    sign: float = 1.0 if val > 0.0 else -1.0
    return sign * (abs(val) - deadzone) / (1.0 - deadzone)


class GamepadTeleopNode(Node):
    def __init__(self, config: TeleopConfig) -> None:
        super().__init__("gamepad_teleop_node")
        self._config: TeleopConfig = config
        self._lock: threading.Lock = threading.Lock()

        self._target_x: float = config.home_x
        self._target_y: float = config.home_y
        self._target_z: float = config.home_z
        self._gripper_cmd: float = 0.0
        self._active: bool = False

        self._joy_sub = self.create_subscription(
            Joy, "/joy", self._joy_callback, 10
        )
        self._ee_target_pub = self.create_publisher(
            PointStamped, "/teleop/ee_target", 10
        )
        self._gripper_pub = self.create_publisher(
            Float32, "/teleop/gripper_cmd", 10
        )

        timer_period: float = 1.0 / self._config.publish_rate_hz
        self._timer = self.create_timer(timer_period, self._publish_callback)

    def _joy_callback(self, msg: Joy) -> None:
        with self._lock:
            self._active = True
            num_buttons: int = len(msg.buttons)
            if (
                self._config.button_reset < num_buttons
                and msg.buttons[self._config.button_reset] == 1
            ):
                self._target_x = self._config.home_x
                self._target_y = self._config.home_y
                self._target_z = self._config.home_z
                self._gripper_cmd = 0.0
                return

            num_axes: int = len(msg.axes)
            val_lx: float = 0.0
            val_ly: float = 0.0
            val_ry: float = 0.0
            val_trigger: float = 1.0

            if self._config.axis_left_x < num_axes:
                val_lx = _apply_deadzone(
                    float(msg.axes[self._config.axis_left_x]),
                    self._config.deadzone,
                )
            if self._config.axis_left_y < num_axes:
                val_ly = _apply_deadzone(
                    float(msg.axes[self._config.axis_left_y]),
                    self._config.deadzone,
                )
            if self._config.axis_right_y < num_axes:
                val_ry = _apply_deadzone(
                    float(msg.axes[self._config.axis_right_y]),
                    self._config.deadzone,
                )
            if self._config.axis_trigger < num_axes:
                val_trigger = float(msg.axes[self._config.axis_trigger])

            delta_x: float = val_ly * self._config.speed_scale_xy
            delta_y: float = val_lx * self._config.speed_scale_xy
            delta_z: float = val_ry * self._config.speed_scale_z

            self._target_x = float(
                np.clip(
                    self._target_x + delta_x,
                    self._config.min_x,
                    self._config.max_x,
                )
            )
            self._target_y = float(
                np.clip(
                    self._target_y + delta_y,
                    self._config.min_y,
                    self._config.max_y,
                )
            )
            self._target_z = float(
                np.clip(
                    self._target_z + delta_z,
                    self._config.min_z,
                    self._config.max_z,
                )
            )

            if val_trigger < 0.99:
                closure: float = float((1.0 - val_trigger) / 2.0)
                self._gripper_cmd = float(np.clip(closure, 0.0, 1.0))
            else:
                self._gripper_cmd = 0.0

    def _publish_callback(self) -> None:
        with self._lock:
            if not self._active:
                return
            tx: float = self._target_x
            ty: float = self._target_y
            tz: float = self._target_z
            grip: float = self._gripper_cmd

        now = self.get_clock().now().to_msg()

        pt_msg = PointStamped()
        pt_msg.header.stamp = now
        pt_msg.header.frame_id = self._config.frame_id
        pt_msg.point.x = tx
        pt_msg.point.y = ty
        pt_msg.point.z = tz
        self._ee_target_pub.publish(pt_msg)

        grip_msg = Float32()
        grip_msg.data = grip
        self._gripper_pub.publish(grip_msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    config = TeleopConfig()
    node = GamepadTeleopNode(config=config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (RuntimeError, RCLError):
                pass
