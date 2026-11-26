import os
from glob import glob
from typing import List, Tuple
from setuptools import find_packages, setup

package_name: str = "tars_med_sim"

data_files_list: List[Tuple[str, List[str]]] = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (
        os.path.join("share", package_name, "launch"),
        glob(os.path.join("launch", "*launch.[pxy][yma]*")),
    ),
    (
        os.path.join("share", package_name, "models"),
        glob(os.path.join("models", "*.xml")),
    ),
]

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=data_files_list,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="limace",
    maintainer_email="limace@todo.todo",
    description="T.A.R.S.-Med MuJoCo simulation package",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_node = tars_med_sim.sim_node:main",
            "gamepad_teleop_node = tars_med_sim.gamepad_teleop_node:main",
        ],
    },
)
