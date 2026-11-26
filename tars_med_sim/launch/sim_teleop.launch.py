from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    enable_viewer_arg = DeclareLaunchArgument(
        "enable_viewer",
        default_value="true",
        description="Launch MuJoCo passive viewer",
    )
    joy_dev_arg = DeclareLaunchArgument(
        "joy_dev",
        default_value="/dev/input/js0",
        description="Path to gamepad device",
    )
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        parameters=[{"device_name": LaunchConfiguration("joy_dev")}],
        output="screen",
    )
    teleop_node = Node(
        package="tars_med_sim",
        executable="gamepad_teleop_node",
        name="gamepad_teleop_node",
        output="screen",
    )
    sim_node = Node(
        package="tars_med_sim",
        executable="sim_node",
        name="sim_node",
        parameters=[{"enable_viewer": LaunchConfiguration("enable_viewer")}],
        output="screen",
    )
    return LaunchDescription(
        [
            enable_viewer_arg,
            joy_dev_arg,
            joy_node,
            teleop_node,
            sim_node,
        ]
    )
