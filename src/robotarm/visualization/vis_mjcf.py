#!/usr/bin/env python3
"""使用 MuJoCo 查看器可视化 MJCF 场景。

用法：python3 -m robotarm.visualization.vis_mjcf --xml_path <scene.xml>
迁移自 vis_mjcf.py。
"""

from __future__ import annotations

import argparse


def main() -> int:
    import mujoco
    from mujoco import viewer as mj_viewer

    parser = argparse.ArgumentParser(description="Visualize an MJCF scene using MuJoCo")
    parser.add_argument(
        "--xml_path",
        type=str,
        required=True,
        help="Path to the xml file to visualize.",
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    mj_viewer.launch_passive(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
