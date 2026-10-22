"""Train a small VLA model for the Piper pick-and-place task.

This file is a compact VLA pipeline for this project. It is not a large
foundation VLA; it is a small behavior-cloning model that consumes:
  - D405 hand-camera RGB
  - D435 external-camera RGB
  - robot/task state vector
  - a language instruction
and predicts the same 5-D action used by ``pd_ee_delta_pos``:
  ``dx, dy, dz, left_gripper_target, right_gripper_target``.

Step 1, collect scripted demonstrations:
  robotarm simulation small-vla -- collect \
    --episodes 100 \
    --dataset data/simulation/piper_vla_demos.h5

Step 2, train the small VLA model:
  robotarm simulation small-vla -- train \
    --dataset data/simulation/piper_vla_demos.h5 \
    --epochs 20 \
    --device cuda

Step 3, evaluate a trained VLA checkpoint visually:
  robotarm simulation small-vla -- eval \
    --checkpoint artifacts/simulation/piper_small_vla/best.pt \
    --display :1 \
    --view split \
    --device cuda

Notes:
  - Data collection uses the scripted controller from step03, with ideal grasp
    by default, to generate consistent successful trajectories.
  - Evaluation does not use scripted waypoints or ideal grasp; actions come from
    the VLA model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from . import environment  # noqa: F401 - registers env/robot
from .viewer import render_view
from .scripted_pick_place import (
    CARRIED_OBJ_OFFSET_FROM_TCP,
    CARRIED_OBJ_XY_OFFSET_FROM_TCP,
    CLOSED_GRIPPER,
    GRASP_TCP_XY_OFFSET,
    OPEN_GRIPPER,
    attach_cube_to_tcp,
    current_tcp,
    go_home,
    hold,
    make_action,
    move_blue_box,
    randomize_task,
    set_actor_xyz,
    to_numpy,
)
from ..paths import SIMULATION_ARTIFACTS_ROOT, SIMULATION_DATA_ROOT

DEFAULT_INSTRUCTION = "pick up the red cube and place it into the blue box"


class Vocab:
    def __init__(self, texts=None):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        if texts:
            for text in texts:
                self.add_text(text)

    def add_text(self, text: str):
        for token in self.tokenize(text):
            if token not in self.stoi:
                self.stoi[token] = len(self.stoi)

    @staticmethod
    def tokenize(text: str):
        return text.lower().replace(".", "").replace(",", "").split()

    def encode(self, text: str, max_len: int = 16):
        ids = [self.stoi.get(tok, self.stoi["<unk>"]) for tok in self.tokenize(text)]
        ids = ids[:max_len]
        ids += [self.stoi["<pad>"]] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)

    def to_json(self):
        return json.dumps(self.stoi)

    @classmethod
    def from_json(cls, value: str):
        vocab = cls()
        vocab.stoi = json.loads(value)
        return vocab


def make_env(args, sensor_width=None, sensor_height=None):
    sensor_width = sensor_width or args.image_width
    sensor_height = sensor_height or args.image_height
    return gym.make(
        "PiperPickCubeToBox-v1",
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        sensor_configs={"width": sensor_width, "height": sensor_height},
        human_render_camera_configs={"width": 512, "height": 384},
    )


def obs_to_numpy(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = np.asarray(obs)
    if obs.ndim == 2 and obs.shape[0] == 1:
        obs = obs[0]
    return obs.astype(np.float32, copy=False)


def sensor_rgb(sensor):
    sensor.capture()
    obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    if "rgb" in obs:
        img = obs["rgb"]
    else:
        img = obs["Color"]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def capture_images(env):
    base = env.unwrapped
    base.scene.update_render(update_sensors=True, update_human_render_cameras=False)
    d405 = sensor_rgb(base.scene.sensors["hand_realsense_d405"])
    d435 = sensor_rgb(base.scene.sensors["external_realsense_d435"])
    return d405, d435


def get_state_obs(env):
    obs = env.unwrapped.get_obs()
    return obs_to_numpy(obs)


def record_step(buffer, env, action, instruction):
    d405, d435 = capture_images(env)
    buffer["d405_rgb"].append(d405)
    buffer["d435_rgb"].append(d435)
    buffer["state"].append(get_state_obs(env))
    buffer["action"].append(np.asarray(action, dtype=np.float32))
    buffer["instruction"].append(instruction)


def step_to_record(
    env,
    target_xyz,
    gripper,
    buffer,
    instruction,
    max_steps=80,
    tol=0.012,
    attach_cube=False,
):
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    for _ in range(max_steps):
        tcp = current_tcp(env)
        delta = target_xyz - tcp
        action = make_action(delta, gripper)
        record_step(buffer, env, action, instruction)
        env.step(action)
        if attach_cube:
            attach_cube_to_tcp(env)
        if np.linalg.norm(delta) < tol:
            break


def hold_record(env, gripper, buffer, instruction, steps=15, attach_cube=False):
    for _ in range(steps):
        action = make_action([0, 0, 0], gripper)
        record_step(buffer, env, action, instruction)
        env.step(action)
        if attach_cube:
            attach_cube_to_tcp(env)


def collect_episode(env, rng, instruction, ideal_grasp=True):
    env.reset(seed=int(rng.integers(0, 1_000_000)))
    cube_xyz, box_xy = randomize_task(env, rng)
    e = env.unwrapped
    buffer = {"d405_rgb": [], "d435_rgb": [], "state": [], "action": [], "instruction": []}

    grasp_xy = cube_xyz[:2] + GRASP_TCP_XY_OFFSET
    place_tcp_xy = box_xy - CARRIED_OBJ_XY_OFFSET_FROM_TCP

    pre_grasp = np.array([grasp_xy[0], grasp_xy[1], 0.16], dtype=np.float32)
    grasp = np.array([grasp_xy[0], grasp_xy[1], 0.015], dtype=np.float32)
    lift = np.array([grasp_xy[0], grasp_xy[1], 0.20], dtype=np.float32)
    pre_place = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.22], dtype=np.float32)
    place = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.075], dtype=np.float32)
    retreat = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.22], dtype=np.float32)

    step_to_record(env, pre_grasp, OPEN_GRIPPER, buffer, instruction, max_steps=100)
    step_to_record(env, grasp, OPEN_GRIPPER, buffer, instruction, max_steps=70)
    hold_record(env, CLOSED_GRIPPER, buffer, instruction, steps=25, attach_cube=ideal_grasp)
    step_to_record(env, lift, CLOSED_GRIPPER, buffer, instruction, max_steps=90, attach_cube=ideal_grasp)
    step_to_record(env, pre_place, CLOSED_GRIPPER, buffer, instruction, max_steps=120, attach_cube=ideal_grasp)
    step_to_record(env, place, CLOSED_GRIPPER, buffer, instruction, max_steps=80, attach_cube=ideal_grasp)
    if ideal_grasp:
        set_actor_xyz(
            e.cube,
            [box_xy[0], box_xy[1], e.box_wall_thickness + e.cube_half_size],
            device=e.device,
        )
    hold_record(env, OPEN_GRIPPER, buffer, instruction, steps=25)
    step_to_record(env, retreat, OPEN_GRIPPER, buffer, instruction, max_steps=80)
    hold_record(env, OPEN_GRIPPER, buffer, instruction, steps=20)
    go_home(env)

    info = e.evaluate()
    success = bool(to_numpy(info["success"])[0])
    return buffer, success, cube_xyz, box_xy


def append_episode(group, episode_id, episode, success, cube_xyz, box_xy):
    ep = group.create_group(f"{episode_id:06d}")
    ep.create_dataset("d405_rgb", data=np.stack(episode["d405_rgb"]), compression="gzip", compression_opts=4)
    ep.create_dataset("d435_rgb", data=np.stack(episode["d435_rgb"]), compression="gzip", compression_opts=4)
    ep.create_dataset("state", data=np.stack(episode["state"]).astype(np.float32), compression="gzip", compression_opts=4)
    ep.create_dataset("action", data=np.stack(episode["action"]).astype(np.float32), compression="gzip", compression_opts=4)
    ep.attrs["instruction"] = episode["instruction"][0]
    ep.attrs["success"] = success
    ep.attrs["cube_xyz"] = cube_xyz.astype(np.float32)
    ep.attrs["box_xy"] = box_xy.astype(np.float32)


class H5VLADataset(Dataset):
    def __init__(self, dataset_path, image_size=96, max_lang_len=16):
        self.path = Path(dataset_path)
        self.image_size = image_size
        self.max_lang_len = max_lang_len
        self.h5 = None
        with h5py.File(self.path, "r") as f:
            self.episode_keys = sorted(f["episodes"].keys())
            texts = [f["episodes"][k].attrs["instruction"] for k in self.episode_keys]
            self.vocab = Vocab(texts)
            self.index = []
            for key in self.episode_keys:
                n = f["episodes"][key]["action"].shape[0]
                self.index.extend((key, i) for i in range(n))
            self.state_dim = f["episodes"][self.episode_keys[0]]["state"].shape[1]
            self.action_dim = f["episodes"][self.episode_keys[0]]["action"].shape[1]

    def __len__(self):
        return len(self.index)

    def _file(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")
        return self.h5

    def _prep_image(self, img):
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def __getitem__(self, idx):
        key, t = self.index[idx]
        ep = self._file()["episodes"][key]
        instruction = ep.attrs["instruction"]
        return {
            "d405": self._prep_image(ep["d405_rgb"][t]),
            "d435": self._prep_image(ep["d435_rgb"][t]),
            "state": torch.from_numpy(ep["state"][t]).float(),
            "lang": torch.from_numpy(self.vocab.encode(instruction, self.max_lang_len)),
            "action": torch.from_numpy(ep["action"][t]).float(),
        }


class SmallVLAModel(nn.Module):
    def __init__(self, state_dim, action_dim, vocab_size, lang_dim=64):
        super().__init__()
        self.lang = nn.Embedding(vocab_size, lang_dim, padding_idx=0)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.state = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + 128 + lang_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, d405, d435, state, lang_ids):
        z405 = self.cnn(d405)
        z435 = self.cnn(d435)
        zstate = self.state(state)
        mask = (lang_ids != 0).float().unsqueeze(-1)
        zlang = self.lang(lang_ids)
        zlang = (zlang * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([z405, z435, zstate, zlang], dim=1))


def collect(args):
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = make_env(args)
    mode = "a" if args.append and args.dataset.exists() else "w"
    with h5py.File(args.dataset, mode) as f:
        f.attrs["instruction"] = args.instruction
        f.attrs["image_width"] = args.image_width
        f.attrs["image_height"] = args.image_height
        group = f.require_group("episodes")
        start = len(group)
        for i in range(args.episodes):
            episode, success, cube_xyz, box_xy = collect_episode(
                env, rng, args.instruction, ideal_grasp=not args.no_ideal_grasp
            )
            append_episode(group, start + i, episode, success, cube_xyz, box_xy)
            print(f"episode {start + i:06d}: steps={len(episode['action'])} success={success}")
    env.close()


def train(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = H5VLADataset(args.dataset, image_size=args.image_size)
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SmallVLAModel(dataset.state_dim, dataset.action_dim, len(dataset.vocab.stoi)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")

    meta = {
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "vocab": dataset.vocab.stoi,
        "image_size": args.image_size,
        "instruction": DEFAULT_INSTRUCTION,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pred = model(
                batch["d405"].to(device), batch["d435"].to(device),
                batch["state"].to(device), batch["lang"].to(device),
            )
            loss = F.mse_loss(pred, batch["action"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * batch["action"].shape[0]
        train_loss /= max(len(train_set), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["d405"].to(device), batch["d435"].to(device),
                    batch["state"].to(device), batch["lang"].to(device),
                )
                loss = F.mse_loss(pred, batch["action"].to(device))
                val_loss += loss.item() * batch["action"].shape[0]
        val_loss /= max(len(val_set), 1)
        print(f"epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        ckpt = {"model": model.state_dict(), "meta": meta}
        torch.save(ckpt, args.output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"  saved best: {args.output_dir / 'best.pt'}")


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    meta = ckpt["meta"]
    model = SmallVLAModel(meta["state_dim"], meta["action_dim"], len(meta["vocab"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocab = Vocab()
    vocab.stoi = meta["vocab"]
    return model, vocab, meta


def eval_model(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, vocab, meta = load_model(args.checkpoint, device)
    env = make_env(args, sensor_width=args.image_width, sensor_height=args.image_height)
    rng = np.random.default_rng(args.seed)
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.render:
        cv2.namedWindow("step07_small_vla_eval", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("step07_small_vla_eval", args.window_width, args.window_height)

    lang = torch.from_numpy(vocab.encode(args.instruction)).unsqueeze(0).to(device)
    successes = []
    for ep in range(args.episodes):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        randomize_task(env, rng)
        go_home(env)
        total_reward = 0.0
        for t in range(args.max_steps):
            d405, d435 = capture_images(env)
            d405_t = torch.from_numpy(cv2.resize(d405, (meta["image_size"], meta["image_size"]))).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            d435_t = torch.from_numpy(cv2.resize(d435, (meta["image_size"], meta["image_size"]))).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            state_t = torch.from_numpy(get_state_obs(env)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(d405_t, d435_t, state_t, lang).cpu().numpy()[0]
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(to_numpy(reward).reshape(-1)[0])
            if args.render:
                frame = render_view(env, args.view)
                cv2.putText(frame, f"VLA ep={ep} step={t} r={total_reward:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("step07_small_vla_eval", frame)
                key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
                if key in (27, ord("q")):
                    env.close()
                    return
            if bool(to_numpy(terminated).reshape(-1)[0] or to_numpy(truncated).reshape(-1)[0]):
                break
        info = env.unwrapped.evaluate()
        success = bool(to_numpy(info["success"])[0])
        successes.append(success)
        print(f"episode {ep}: success={success} reward={total_reward:.3f}")
    print(f"success_rate={np.mean(successes):.3f} episodes={len(successes)}")
    env.close()
    if args.render:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env_args(p):
        p.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu", "auto"])
        p.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
        p.add_argument("--image-width", type=int, default=128)
        p.add_argument("--image-height", type=int, default=96)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    p_collect = sub.add_parser("collect")
    add_env_args(p_collect)
    p_collect.add_argument("--dataset", type=Path, default=SIMULATION_DATA_ROOT / "piper_vla_demos.h5")
    p_collect.add_argument("--episodes", type=int, default=100)
    p_collect.add_argument("--append", action="store_true")
    p_collect.add_argument("--no-ideal-grasp", action="store_true")

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", type=Path, default=SIMULATION_DATA_ROOT / "piper_vla_demos.h5")
    p_train.add_argument("--output-dir", type=Path, default=SIMULATION_ARTIFACTS_ROOT / "piper_small_vla")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--image-size", type=int, default=96)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--val-ratio", type=float, default=0.1)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--seed", type=int, default=0)

    p_eval = sub.add_parser("eval")
    add_env_args(p_eval)
    p_eval.add_argument("--checkpoint", type=Path, default=SIMULATION_ARTIFACTS_ROOT / "piper_small_vla/best.pt")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--max-steps", type=int, default=120)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--render", action="store_true")
    p_eval.add_argument("--display", default=None)
    p_eval.add_argument("--view", default="split", choices=["render", "d405", "d435", "split"])
    p_eval.add_argument("--fps", type=float, default=20)
    p_eval.add_argument("--window-width", type=int, default=960)
    p_eval.add_argument("--window-height", type=int, default=540)

    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "train":
        train(args)
    elif args.command == "eval":
        eval_model(args)


if __name__ == "__main__":
    main()
