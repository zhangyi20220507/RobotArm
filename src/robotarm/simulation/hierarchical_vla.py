"""Hierarchical VL+A controller for Piper pick-and-place.

This file demonstrates a layered Vision-Language + Action architecture:

  1. Language Grounding
     Natural language -> target object name + target bin name.

  2. Vision Grounding
     Camera RGB -> simple color masks / image centers for the grounded object
     and bin. This is intentionally modular so a real detector/segmenter can
     replace the color threshold code later.

  3. Action Execution
     3D target poses -> Cartesian waypoint pick-and-place controller.

Run one command:
  robotarm simulation hierarchical-vla -- \
    --display :1 --render \
    --command "put the red cube into the blue box"

Interactive language loop:
  robotarm simulation hierarchical-vla -- --display :1 --render --interactive

Multi-step command:
  robotarm simulation hierarchical-vla -- \
    --display :1 --render \
    --command "put red cube into blue box then move cyan cylinder to orange bin"

Notes:
  - This is a layered VL+A scaffold, not a pretrained foundation VLA.
  - Vision masks currently use color thresholding on the external D435 RGB.
  - 3D poses default to simulator actor poses via ``--pose-backend sim``.
  - The low-level action executor uses the same stable waypoint controller as
    step08, with ideal grasp by default. Pass ``--no-ideal-grasp`` to test pure
    physical contact.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import torch

from . import environment  # noqa: F401 - registers env/robot
from .viewer import render_view
from .scripted_pick_place import go_home, randomize_task, to_numpy
from .open_language_vla import (
    BIN_ALIASES,
    OBJECT_ALIASES,
    PickPlaceCommand,
    bin_center,
    execute_pick_place,
    object_actor,
    parse_commands,
)


# HSV ranges are intentionally broad because simulator lighting changes color a bit.
HSV_RANGES = {
    "red_target_cube": [((0, 70, 60), (12, 255, 255)), ((165, 70, 60), (180, 255, 255))],
    "green_tall_block": [((45, 50, 50), (85, 255, 255))],
    "yellow_flat_block": [((18, 60, 80), (38, 255, 255))],
    "purple_long_block": [((125, 35, 50), (165, 255, 255))],
    "cyan_round_block": [((80, 40, 50), (105, 255, 255))],
    "blue_square_bin": [((100, 40, 35), (130, 255, 255))],
    "orange_tall_bin": [((5, 60, 60), (25, 255, 255))],
    "green_wide_bin": [((45, 45, 35), (85, 255, 255))],
}


@dataclass
class VisionGrounding:
    name: str
    mask: np.ndarray
    center_xy: tuple[int, int] | None
    area: int


@dataclass
class GroundedTask:
    command: PickPlaceCommand
    object_vision: VisionGrounding
    bin_vision: VisionGrounding
    object_xyz: np.ndarray
    bin_xy: np.ndarray


def make_env(args):
    return gym.make(
        "PiperPickCubeToBox-v1",
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        sensor_configs={"width": args.image_width, "height": args.image_height},
        human_render_camera_configs={"width": 512, "height": 384},
    )


def sensor_rgb(sensor):
    sensor.capture()
    obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    img = obs.get("rgb", obs.get("Color"))
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def external_rgb(env):
    base = env.unwrapped
    base.scene.update_render(update_sensors=True, update_human_render_cameras=False)
    return sensor_rgb(base.scene.sensors["external_realsense_d435"])


def color_mask(rgb: np.ndarray, name: str) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in HSV_RANGES[name]:
        mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def mask_center(mask: np.ndarray):
    moments = cv2.moments(mask)
    if moments["m00"] <= 1e-6:
        return None, 0
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy), int(moments["m00"] / 255)


def vision_ground(rgb: np.ndarray, name: str) -> VisionGrounding:
    mask = color_mask(rgb, name)
    center, area = mask_center(mask)
    return VisionGrounding(name=name, mask=mask, center_xy=center, area=area)


def estimate_object_xyz(env, object_name: str, pose_backend: str):
    if pose_backend != "sim":
        raise NotImplementedError("Only --pose-backend sim is implemented in this scaffold.")
    actor, _ = object_actor(env, object_name)
    return to_numpy(actor.pose.p)[0].astype(np.float32)


def estimate_bin_xy(env, bin_name: str, pose_backend: str):
    if pose_backend != "sim":
        raise NotImplementedError("Only --pose-backend sim is implemented in this scaffold.")
    return bin_center(env, bin_name).astype(np.float32)


def ground_task(env, command: PickPlaceCommand, pose_backend: str) -> GroundedTask:
    rgb = external_rgb(env)
    object_vg = vision_ground(rgb, command.object_name)
    bin_vg = vision_ground(rgb, command.bin_name)
    object_xyz = estimate_object_xyz(env, command.object_name, pose_backend)
    bin_xy = estimate_bin_xy(env, command.bin_name, pose_backend)
    return GroundedTask(command, object_vg, bin_vg, object_xyz, bin_xy)


def overlay_grounding(rgb: np.ndarray, tasks: list[GroundedTask]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    colors = [(0, 0, 255), (255, 128, 0), (0, 255, 255), (255, 0, 255)]
    for i, task in enumerate(tasks):
        color = colors[i % len(colors)]
        for label, vg in [("obj", task.object_vision), ("bin", task.bin_vision)]:
            if vg.center_xy is not None:
                cv2.circle(bgr, vg.center_xy, 8, color, 2)
                cv2.putText(
                    bgr,
                    f"{label}:{vg.name} area={vg.area}",
                    (vg.center_xy[0] + 8, vg.center_xy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
    return bgr


def print_grounding(tasks: list[GroundedTask]):
    for task in tasks:
        print(
            f"VL ground: {task.command.object_name} -> {task.command.bin_name} | "
            f"obj_px={task.object_vision.center_xy} area={task.object_vision.area} | "
            f"bin_px={task.bin_vision.center_xy} area={task.bin_vision.area} | "
            f"obj_xyz={np.round(task.object_xyz, 3)} bin_xy={np.round(task.bin_xy, 3)}"
        )


def render_status(env, view, label, tasks=None):
    if view == "grounding":
        rgb = external_rgb(env)
        return overlay_grounding(rgb, tasks or [])
    frame = render_view(env, view)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        "VL+A | type command in terminal | 1/2/3/4/g view | q quit window",
        (12, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return frame


def execute_tasks(env, tasks: list[GroundedTask], args, render_cb=None):
    for task in tasks:
        execute_pick_place(
            env,
            task.command,
            render_cb=render_cb,
            ideal_grasp=not args.no_ideal_grasp,
        )


def process_language(env, text: str, args, render_cb=None):
    commands = parse_commands(text)
    tasks = [ground_task(env, command, args.pose_backend) for command in commands]
    print_grounding(tasks)
    if args.render and args.pause_after_grounding:
        frame = render_status(env, "grounding", "grounding", tasks)
        cv2.imshow("step09_hierarchical_vla", frame)
        cv2.waitKey(args.grounding_pause_ms)
    execute_tasks(env, tasks, args, render_cb=render_cb)


def print_help():
    print(__doc__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--display", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--view", default="split", choices=["render", "d405", "d435", "split", "grounding"])
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--pose-backend", default="sim", choices=["sim"])
    parser.add_argument("--no-ideal-grasp", action="store_true")
    parser.add_argument("--pause-after-grounding", action="store_true")
    parser.add_argument("--grounding-pause-ms", type=int, default=1200)
    args = parser.parse_args()

    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    rng = np.random.default_rng(args.seed)
    env = make_env(args)
    env.reset(seed=args.seed)
    randomize_task(env, rng)
    go_home(env)

    view = [args.view]
    delay_ms = max(1, int(1000 / args.fps))
    if args.render:
        cv2.namedWindow("step09_hierarchical_vla", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("step09_hierarchical_vla", args.width, args.height)

    def render_cb(label):
        if not args.render:
            return
        frame = render_status(env, view[0], label)
        cv2.imshow("step09_hierarchical_vla", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("1"):
            view[0] = "render"
        elif key == ord("2"):
            view[0] = "d405"
        elif key == ord("3"):
            view[0] = "d435"
        elif key == ord("4"):
            view[0] = "split"
        elif key == ord("g"):
            view[0] = "grounding"
        elif key in (27, ord("q")):
            raise KeyboardInterrupt
        time.sleep(0.001)

    try:
        if args.command:
            process_language(env, args.command, args, render_cb=render_cb)
            return
        if not args.interactive:
            print_help()
            return

        print("Hierarchical VL+A ready. Type 'help', 'reset', 'home', or a pick-place command.")
        while True:
            if args.render:
                frame = render_status(env, view[0], "waiting for language command")
                cv2.imshow("step09_hierarchical_vla", frame)
                cv2.waitKey(1)
            text = input("vl+a> ").strip()
            if not text:
                continue
            low = text.lower().strip()
            if low in {"quit", "exit", "q"}:
                break
            if low == "help":
                print_help()
                continue
            if low == "reset":
                env.reset(seed=int(rng.integers(0, 1_000_000)))
                randomize_task(env, rng)
                go_home(env, render_cb=render_cb)
                continue
            if low == "home":
                go_home(env, render_cb=render_cb)
                continue
            try:
                process_language(env, text, args, render_cb=render_cb)
            except Exception as exc:
                print(f"Could not execute command: {exc}")
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
