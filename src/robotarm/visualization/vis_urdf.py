#!/usr/bin/env python3
"""使用 Placo/Meshcat 可视化 URDF 模型。

用法：python3 -m robotarm.visualization.vis_urdf --urdf_path <model.urdf>
迁移自 vis_urdf.py。
"""

from __future__ import annotations

import argparse


def main() -> int:
    import webbrowser

    import placo
    from placo_utils.visualization import robot_viz

    parser = argparse.ArgumentParser(description="Visualize a URDF model using Placo and Meshcat")
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to the URDF file to visualize.",
    )
    args = parser.parse_args()
    robot = placo.RobotWrapper(args.urdf_path)
    print(f"robot state: {robot.state.q}")
    print(f"robot state shape: {robot.state.q.shape}")
    robot.update_kinematics()
    viz = robot_viz(robot)
    webbrowser.open(viz.viewer.url())

    while True:
        robot.update_kinematics()
        viz.display(robot.state.q)


if __name__ == "__main__":
    raise SystemExit(main())
