"""Open-language VLA-style controller for Piper pick-and-place.

This is a practical language-interactive controller for the current simulator.
It accepts open-ish natural language commands, grounds object/bin words to the
scene, and executes one or more pick-and-place tasks in sequence.

Examples:
  robotarm simulation open-language-vla -- --display :1 --render --interactive

  robotarm simulation open-language-vla -- \
    --command "put the red cube into the blue box"

  robotarm simulation open-language-vla -- \
    --display :1 --render \
    --command "put the red cube in the blue box then move the cyan cylinder to the orange bin"

Supported objects:
  red cube, green block, yellow block, purple block, cyan cylinder

Supported boxes:
  blue box, orange box, green box

Interactive commands:
  Any pick-place instruction, for example:
    put red cube into blue box
    move yellow block to green bin then put cyan cylinder into orange box
  reset   randomize the scene
  home    return robot to Home
  help    print help
  quit    exit

Notes:
  - This is a VLA-style language control scaffold, not a large pretrained VLA.
  - The high-level language grounding is rule based.
  - The low-level executor uses the same Cartesian waypoint controller as step03.
  - By default it uses ideal grasp for robust command execution. Pass
    ``--no-ideal-grasp`` to test pure physical contact.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import sapien

from . import environment  # noqa: F401 - registers env/robot
from .viewer import render_view
from .scripted_pick_place import (
    CARRIED_OBJ_OFFSET_FROM_TCP,
    CARRIED_OBJ_XY_OFFSET_FROM_TCP,
    CLOSED_GRIPPER,
    GRASP_TCP_XY_OFFSET,
    OPEN_GRIPPER,
    current_tcp,
    go_home,
    hold,
    make_action,
    randomize_task,
    set_actor_xyz,
    to_numpy,
)


OBJECT_ALIASES = {
    "red_target_cube": ["red target cube", "red cube", "red block", "red"],
    "green_tall_block": ["green tall block", "green block", "green cuboid"],
    "yellow_flat_block": ["yellow flat block", "yellow block", "yellow cube", "yellow"],
    "purple_long_block": ["purple long block", "purple block", "purple"],
    "cyan_round_block": ["cyan round block", "cyan cylinder", "blue cylinder", "cyan"],
}

BIN_ALIASES = {
    "blue_square_bin": ["blue square bin", "blue box", "blue bin", "blue"],
    "orange_tall_bin": ["orange tall bin", "orange box", "orange bin", "orange"],
    "green_wide_bin": ["green wide bin", "green box", "green bin"],
}


@dataclass
class PickPlaceCommand:
    object_name: str
    bin_name: str
    raw_text: str


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s;]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_alias(text: str, alias_map: dict[str, list[str]]):
    matches = []
    padded = f" {text} "
    for name, aliases in alias_map.items():
        for alias in aliases:
            pattern = f" {alias} "
            idx = padded.find(pattern)
            if idx >= 0:
                matches.append((idx, len(alias), name, alias))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], -item[1]))
    return matches[0][2]


def find_all_aliases(text: str, alias_map: dict[str, list[str]]):
    matches = []
    padded = f" {text} "
    for name, aliases in alias_map.items():
        for alias in aliases:
            start = 0
            pattern = f" {alias} "
            while True:
                idx = padded.find(pattern, start)
                if idx < 0:
                    break
                matches.append((idx, len(alias), name, alias))
                start = idx + 1
    matches.sort(key=lambda item: item[0])
    dedup = []
    seen_spans = set()
    for item in matches:
        span = (item[0], item[0] + item[1])
        if span not in seen_spans:
            dedup.append(item)
            seen_spans.add(span)
    return dedup


def parse_commands(text: str) -> list[PickPlaceCommand]:
    text = normalize(text)
    if not text:
        return []

    chunks = []
    for part in re.split(r"\s+then\s+|;", text):
        part = part.strip()
        if part:
            chunks.append(part)

    commands = []
    for chunk in chunks:
        object_matches = find_all_aliases(chunk, OBJECT_ALIASES)
        bin_matches = find_all_aliases(chunk, BIN_ALIASES)
        unique_objects = []
        unique_bins = []
        for _, _, name, _ in object_matches:
            if name not in unique_objects:
                unique_objects.append(name)
        for _, _, name, _ in bin_matches:
            if name not in unique_bins:
                unique_bins.append(name)

        if len(unique_objects) > 1 and len(unique_objects) == len(unique_bins):
            for obj, bin_name in zip(unique_objects, unique_bins):
                commands.append(PickPlaceCommand(obj, bin_name, chunk))
            continue

        obj = find_alias(chunk, OBJECT_ALIASES)
        bin_name = find_alias(chunk, BIN_ALIASES)
        if obj is None or bin_name is None:
            raise ValueError(
                f"Could not parse command: {chunk!r}. "
                "Mention one supported object and one supported box."
            )
        commands.append(PickPlaceCommand(obj, bin_name, chunk))
    return commands


def object_actor(env, object_name: str):
    e = env.unwrapped
    for actor, spec in zip(e.objects, e.object_specs):
        if spec["name"] == object_name:
            return actor, spec
    raise KeyError(object_name)


def bin_center(env, bin_name: str) -> np.ndarray:
    e = env.unwrapped
    for idx, spec in enumerate(e.box_specs):
        if spec["name"] != bin_name:
            continue
        if idx == 0 and hasattr(e, "target_box_centers"):
            return to_numpy(e.target_box_centers)[0].astype(np.float32)
        return np.asarray(spec["center"], dtype=np.float32)
    raise KeyError(bin_name)


def bin_height(env, bin_name: str) -> float:
    e = env.unwrapped
    for spec in e.box_specs:
        if spec["name"] == bin_name:
            return float(spec["height"])
    raise KeyError(bin_name)


def set_goal_to_bin(env, bin_name: str):
    e = env.unwrapped
    center = bin_center(env, bin_name)
    z = e.box_wall_thickness + e.cube_half_size
    set_actor_xyz(e.goal_site, [center[0], center[1], z], device=e.device)


def attach_actor_to_tcp(env, actor, offset=CARRIED_OBJ_OFFSET_FROM_TCP):
    e = env.unwrapped
    tcp = current_tcp(env)
    xyz = tcp + offset
    xyz[2] = max(float(xyz[2]), e.cube_half_size)
    set_actor_xyz(actor, xyz, device=e.device)


def step_to_actor(
    env,
    target_xyz,
    gripper,
    actor=None,
    attach=False,
    max_steps=80,
    tol=0.012,
    render_cb=None,
    label="",
):
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    for _ in range(max_steps):
        tcp = current_tcp(env)
        delta = target_xyz - tcp
        action = make_action(delta, gripper)
        env.step(action)
        if attach and actor is not None:
            attach_actor_to_tcp(env, actor)
        if render_cb is not None:
            render_cb(label)
        if np.linalg.norm(delta) < tol:
            break


def hold_actor(env, gripper, actor=None, attach=False, steps=15, render_cb=None, label=""):
    for _ in range(steps):
        env.step(make_action([0, 0, 0], gripper))
        if attach and actor is not None:
            attach_actor_to_tcp(env, actor)
        if render_cb is not None:
            render_cb(label)


def execute_pick_place(env, command: PickPlaceCommand, render_cb=None, ideal_grasp=True):
    actor, spec = object_actor(env, command.object_name)
    obj_pos = to_numpy(actor.pose.p)[0].astype(np.float32)
    target_bin = bin_center(env, command.bin_name)
    set_goal_to_bin(env, command.bin_name)

    grasp_xy = obj_pos[:2] + GRASP_TCP_XY_OFFSET
    place_tcp_xy = target_bin - CARRIED_OBJ_XY_OFFSET_FROM_TCP

    pre_grasp = np.array([grasp_xy[0], grasp_xy[1], 0.16], dtype=np.float32)
    grasp = np.array([grasp_xy[0], grasp_xy[1], 0.015], dtype=np.float32)
    lift = np.array([grasp_xy[0], grasp_xy[1], 0.20], dtype=np.float32)
    pre_place = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.22], dtype=np.float32)
    place = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.075], dtype=np.float32)
    retreat = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.22], dtype=np.float32)

    print(f"execute: {command.object_name} -> {command.bin_name}")
    step_to_actor(env, pre_grasp, OPEN_GRIPPER, max_steps=100, render_cb=render_cb, label="pre_grasp")
    step_to_actor(env, grasp, OPEN_GRIPPER, max_steps=70, render_cb=render_cb, label="grasp_down")
    hold_actor(env, CLOSED_GRIPPER, actor=actor, attach=ideal_grasp, steps=25, render_cb=render_cb, label="close")
    step_to_actor(env, lift, CLOSED_GRIPPER, actor=actor, attach=ideal_grasp, max_steps=90, render_cb=render_cb, label="lift")
    step_to_actor(env, pre_place, CLOSED_GRIPPER, actor=actor, attach=ideal_grasp, max_steps=120, render_cb=render_cb, label="pre_place")
    step_to_actor(env, place, CLOSED_GRIPPER, actor=actor, attach=ideal_grasp, max_steps=80, render_cb=render_cb, label="place_down")

    if ideal_grasp:
        drop_z = env.unwrapped.box_wall_thickness + env.unwrapped.cube_half_size
        set_actor_xyz(actor, [target_bin[0], target_bin[1], drop_z], device=env.unwrapped.device)
    hold_actor(env, OPEN_GRIPPER, steps=25, render_cb=render_cb, label="open")
    step_to_actor(env, retreat, OPEN_GRIPPER, max_steps=80, render_cb=render_cb, label="retreat")
    hold_actor(env, OPEN_GRIPPER, steps=20, render_cb=render_cb, label="settle")
    go_home(env, render_cb=render_cb)


def print_help():
    print(__doc__)


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
        human_render_camera_configs={"width": 512, "height": 384},
    )


def draw_frame(env, view, label):
    frame = render_view(env, view)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        "type language in terminal | 1/2/3/4 switch view | q quit window",
        (12, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return frame


def run_commands(env, commands, render_cb=None, ideal_grasp=True):
    for command in commands:
        execute_pick_place(env, command, render_cb=render_cb, ideal_grasp=ideal_grasp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default=None, help="Language command to execute once.")
    parser.add_argument("--interactive", action="store_true", help="Read language commands from stdin.")
    parser.add_argument("--display", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--view", default="split", choices=["render", "d405", "d435", "split"])
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--no-ideal-grasp", action="store_true")
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
        cv2.namedWindow("step08_open_language_vla", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("step08_open_language_vla", args.width, args.height)

    def render_cb(label):
        if not args.render:
            return
        frame = draw_frame(env, view[0], label)
        cv2.imshow("step08_open_language_vla", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("1"):
            view[0] = "render"
        elif key == ord("2"):
            view[0] = "d405"
        elif key == ord("3"):
            view[0] = "d435"
        elif key == ord("4"):
            view[0] = "split"
        elif key in (27, ord("q")):
            raise KeyboardInterrupt
        time.sleep(0.001)

    try:
        if args.command:
            commands = parse_commands(args.command)
            print("parsed:", commands)
            run_commands(env, commands, render_cb=render_cb, ideal_grasp=not args.no_ideal_grasp)
            return

        if not args.interactive:
            print_help()
            return

        print("Open-language VLA controller ready. Type 'help' for examples, 'quit' to exit.")
        while True:
            if args.render:
                frame = draw_frame(env, view[0], "waiting for language command")
                cv2.imshow("step08_open_language_vla", frame)
                cv2.waitKey(1)
            text = input("vla> ").strip()
            if not text:
                continue
            cmd = normalize(text)
            if cmd in {"quit", "exit", "q"}:
                break
            if cmd == "help":
                print_help()
                continue
            if cmd == "reset":
                env.reset(seed=int(rng.integers(0, 1_000_000)))
                randomize_task(env, rng)
                go_home(env, render_cb=render_cb)
                continue
            if cmd == "home":
                go_home(env, render_cb=render_cb)
                continue
            try:
                commands = parse_commands(text)
                print("parsed:", commands)
                run_commands(env, commands, render_cb=render_cb, ideal_grasp=not args.no_ideal_grasp)
            except Exception as exc:
                print(f"Could not execute command: {exc}")
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
