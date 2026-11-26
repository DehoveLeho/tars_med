# T.A.R.S.-Med

**Tactical Autonomous Robotic Scrub Nurse for Medical Assistance**

T.A.R.S.-Med is a ROS 2 Humble robotics workspace implementing simulation, perception, teleoperation, and autonomous grasping/handover of surgical instruments using the Unitree G1 humanoid robot in MuJoCo.

> 📖 **Guide complet d'installation pas-à-pas (Ubuntu 22.04 natif, WSL2, Fedora Distrobox) :** voir [INSTALL.md](INSTALL.md).

---

## Repository Structure

```
tars_med/
├── .gitignore
├── README.md
├── INSTALL.md                       # Comprehensive team installation & setup guide
├── projet_interfaces/               # ROS 2 message & action interface package
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── action/
│   │   └── ExecuteGrasp.action      # Grasping & instrument handover action
│   └── msg/
│       ├── ErrorCode.msg            # Error taxonomy definitions
│       ├── ObjectPose.msg           # 3D perceived pose of instruments
│       └── OrchestratorState.msg    # State machine status (Approach, Grasp, Handover, etc.)
└── tars_med_sim/                    # MuJoCo simulation & teleoperation package (ament_python)
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── launch/
    │   └── sim_teleop.launch.py     # Launch joy_node, gamepad_teleop_node, sim_node
    ├── models/
    │   ├── tars_med_scene.xml       # Medical tray, instruments & G1 weld scene
    │   └── meshes -> ../../../unitree_mujoco/unitree_robots/g1/meshes
    └── tars_med_sim/
        ├── __init__.py
        ├── ik_solver.py             # Damped least-squares IK with nullspace projection
        ├── sim_node.py              # MuJoCo bridge, IK tracking, weld grasping & action server
        └── gamepad_teleop_node.py   # Gamepad / joystick teleop node
```

---

## Workspace Layout & Prerequisites

This repository is cloned inside a ROS 2 workspace at `~/tars_med_ws/src/tars_med`:

```
~/tars_med_ws/src/
├── unitree_mujoco/                  # Upstream Unitree MuJoCo models (Untracked)
│   └── COLCON_IGNORE                # Must exist to avoid building upstream C++ examples
└── tars_med/                        # Team Git repository root
    ├── projet_interfaces/
    └── tars_med_sim/
```

### Upstream `unitree_mujoco` Setup
The simulation depends on the Unitree G1 MJCF models from `unitree_mujoco`. Clone and pin to commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`:

```bash
cd ~/tars_med_ws/src
git clone https://github.com/unitreerobotics/unitree_mujoco.git
cd unitree_mujoco
git checkout 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
touch COLCON_IGNORE
```

> **IMPORTANT**: `COLCON_IGNORE` must remain in `unitree_mujoco/` so `colcon build` ignores upstream non-ROS C++ projects.

---

## Build

From `~/tars_med_ws`:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## Running the Simulation

Launch the complete simulation with the passive MuJoCo viewer and teleoperation:

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 ros2 launch tars_med_sim sim_teleop.launch.py
```

*(Setting `XDG_SESSION_TYPE=x11` and unsetting `WAYLAND_DISPLAY` prevents GLFW crashes in containerized/Wayland environments).*

### Interactive Controls (MuJoCo 3D Window)
Click on the MuJoCo viewer window to control the right arm directly:
- **D** : Full autonomous handover demo (approach, physical grasp, lift +20 cm, present forward, snatch release, return home)
- **Arrows (↑ / ↓ / ← / →)** : Translate right hand in X/Y plane
- **W / S** (or **PgUp / PgDn**) : Translate right hand vertically in Z
- **Space** : Toggle gripper open / close (automatically welds/unwelds proxy object within reach)
- **R** : Trigger instrument handover snatch & return home
- **H** : Return arm to default home pose

### Autonomous Grasping Action
While the simulation is running, trigger an autonomous pick-and-handover sequence via ROS 2 Action:

```bash
ros2 action send_goal /manipulation/execute_grasp projet_interfaces/action/ExecuteGrasp '{object_id: "bottle", max_force_newton: 10.0}'
```

Available proxy objects on the medical tray: `scalpel`, `tourniquet`, `bottle`.
