"""VNC-friendly OpenCV viewer for the Piper ManiSkill task.

This avoids ManiSkill/SAPIEN's official interactive viewer. It renders with
``render_mode="rgb_array"`` and displays frames in a regular OpenCV window.

Run random actions in VNC display :1:
  robotarm simulation viewer -- --display :1

Run with the three-panel camera layout:
  robotarm simulation viewer -- --display :1 --view split

Run without random motion, useful for inspecting the scene:
  robotarm simulation viewer -- --display :1 --zero-action --view split

Keys:
  q / ESC  quit
  r        reset environment
  SPACE    pause / continue
  1        third-person render camera
  2        hand Realsense D405 view
  3        external Realsense D435 view
  4        split view: large third-person left, D405/D435 stacked right
"""

from __future__ import annotations

import argparse
import os
import time

# OpenCV's bundled Qt plugin still probes a removed cv2/qt/fonts directory.
# Point it at the system font collection before importing cv2.
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import gymnasium as gym
import numpy as np
import torch

from . import environment  # noqa: F401 - registers PiperPickCubeToBox-v1


def frame_to_rgb(frame):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame




def rgb_to_bgr(frame):
    return cv2.cvtColor(frame_to_rgb(frame), cv2.COLOR_RGB2BGR)


def sensor_rgb_to_bgr(sensor_obs):
    if "rgb" in sensor_obs:
        frame = sensor_obs["rgb"]
    elif "Color" in sensor_obs:
        frame = sensor_obs["Color"]
    else:
        frame = next(iter(sensor_obs.values()))
    return rgb_to_bgr(frame)


def render_view(env, view_mode):
    base_env = env.unwrapped
    if view_mode == "render":
        return rgb_to_bgr(env.render())

    base_env.scene.update_render(update_sensors=True, update_human_render_cameras=False)

    def resolve_sensor(role):
        aliases = {
            "wrist": ("observation.images.wrist", "hand_realsense_d405"),
            "global": ("observation.images.global", "external_realsense_d435"),
        }
        for uid in aliases[role]:
            if uid in base_env.scene.sensors:
                return uid
        available = ", ".join(sorted(base_env.scene.sensors)) or "none"
        raise KeyError(
            f"No {role} camera found; tried {aliases[role]}. "
            f"Available sensors: {available}"
        )

    def capture_sensor(uid):
        sensor = base_env.scene.sensors[uid]
        sensor.capture()
        return sensor_rgb_to_bgr(sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False))

    if view_mode in ("d405", "d435"):
        uid = resolve_sensor("wrist" if view_mode == "d405" else "global")
        return capture_sensor(uid)

    third_person = rgb_to_bgr(env.render())
    d405 = capture_sensor(resolve_sensor("wrist"))
    d435 = capture_sensor(resolve_sensor("global"))

    canvas_h = third_person.shape[0]
    left_w = int(canvas_h * 4 / 3)
    right_w = int(left_w * 0.5)
    small_h = canvas_h // 2

    left = cv2.resize(third_person, (left_w, canvas_h), interpolation=cv2.INTER_AREA)
    d405_small = cv2.resize(d405, (right_w, small_h), interpolation=cv2.INTER_AREA)
    d435_small = cv2.resize(d435, (right_w, canvas_h - small_h), interpolation=cv2.INTER_AREA)
    right = cv2.vconcat([d405_small, d435_small])
    return cv2.hconcat([left, right])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PiperPickCupToContainer-v1")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--sim-backend", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--view", default="render", choices=["render", "d405", "d435", "split"])
    parser.add_argument(
        "--display",
        default=None,
        help="X display to use, for example ':1' for many VNC sessions.",
    )
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help="Hold still instead of sampling random actions.",
    )
    args = parser.parse_args()
    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        human_render_camera_configs={"width": args.render_width, "height": args.render_height},
    )
    env.reset(seed=args.seed)

    cv2.namedWindow(args.env_id, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.env_id, args.width, args.height)
    delay_ms = max(1, int(1000 / args.fps))

    print(
        f"OpenCV viewer ready. sim_backend={args.sim_backend}, "
        f"render_backend={args.render_backend}, "
        f"render={args.render_width}x{args.render_height}. "
        "Keys: q/ESC quit, r reset, SPACE pause, 1 render, 2 D405, 3 D435, 4 split."
    )
    paused = False
    view_mode = args.view
    try:
        while True:
            if not paused:
                if args.zero_action:
                    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                else:
                    action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
                if done:
                    env.reset()

            frame = render_view(env, view_mode)
            cv2.imshow(args.env_id, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                env.reset()
            if key == ord(" "):
                paused = not paused
            if key == ord("1"):
                view_mode = "render"
            if key == ord("2"):
                view_mode = "d405"
            if key == ord("3"):
                view_mode = "d435"
            if key == ord("4"):
                view_mode = "split"
            time.sleep(0.001)
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
