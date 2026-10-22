"""Piper ManiSkill environments for tabletop pick-and-place.

What this file does:
  - Registers robot uid: ``piper``.
  - Registers env id: ``PiperPickCubeToBox-v1``.
  - Registers env id: ``PiperPickCupToContainer-v1`` (recommended).
  - Loads the self-contained Piper URDF assets distributed with ``robotarm``.
  - Builds 5 colored objects, 3 colored bins, and D405/D435 camera sensors.
  - Adds gripper contact-force feedback to state observations.

Typical usage from another script:
  import gymnasium as gym
  import robotarm.simulation.environment  # required: registers the env

  env = gym.make(
      "PiperPickCubeToBox-v1",
      obs_mode="state",
      control_mode="pd_ee_delta_pos",
      reward_mode="normalized_dense",
      render_mode="rgb_array",
      num_envs=1,
  )

Useful env ids / sensor names:
  - Realistic task: ``PiperPickCupToContainer-v1``
  - Env id: ``PiperPickCubeToBox-v1``
  - Hand camera: ``hand_realsense_d405``
  - External camera: ``external_realsense_d435``
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


from ..paths import PIPER_MANISKILL_URDF


@register_agent()
class Piper(BaseAgent):
    uid = "piper"
    urdf_path = str(PIPER_MANISKILL_URDF)
    disable_self_collisions = True

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            link7=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            link8=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    gripper_joint_names = ["joint7", "joint8"]
    ee_link_name = "piper_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    # The previous 100 N-equivalent drive limit let Cartesian IK target changes
    # produce unrealistically large 20 Hz joint jumps. A lower limit yields a
    # trajectory closer to the measured Piper motion used for real demos.
    arm_force_limit = 30

    gripper_stiffness = 1e4
    gripper_damping = 1e3
    # Joint-drive limit, not a direct fingertip-newton command. Kept well below
    # the old debug value (100) while retaining enough margin for a 0.23 kg cup.
    gripper_force_limit = 50

    keyframes = dict(
        rest=Keyframe(
            # Task-ready pose on the same continuous IK branch used by the
            # Cartesian expert. Starting from the old generic pose forced J4
            # and J6 through a large branch transition in the first frames.
            qpos=np.array(
                [-0.5802, 1.1430, -1.2574, 0.9077, 1.1434, -0.8913, 0.035, -0.035]
            ),
            pose=sapien.Pose(),
        ),
        zeros=Keyframe(qpos=np.zeros(8), pose=sapien.Pose()),
    )

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=[0.0, -0.035],
            upper=[0.035, 0.0],
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
        )
        return deepcopy(
            dict(
                pd_joint_delta_pos=dict(
                    arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
                ),
                pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
                pd_ee_delta_pos=dict(
                    arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos
                ),
                pd_ee_delta_pose=dict(
                    arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
                ),
            )
        )

    def _after_init(self):
        links = self.robot.get_links()
        self.finger1_link = sapien_utils.get_obj_by_name(links, "link7")
        self.finger2_link = sapien_utils.get_obj_by_name(links, "link8")
        self.tcp = sapien_utils.get_obj_by_name(links, self.ee_link_name)

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    def is_grasping(self, obj: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, obj)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, obj)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :6]
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold


@register_env("PiperPickCubeToBox-v1", max_episode_steps=100)
class PiperPickCubeToBoxEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["piper"]
    agent: Piper

    cube_half_size = 0.025
    cube_spawn_half_size = 0.06
    cube_spawn_center = (0.0, 0.0)
    box_center = (0.18, 0.0)
    box_inner_half_size = (0.075, 0.075)
    box_wall_thickness = 0.008
    box_wall_height = 0.06
    goal_thresh = 0.06
    d405_width = 848
    d405_height = 480
    d405_vfov = np.deg2rad(58.0)
    d435_width = 848
    d435_height = 480
    d435_vfov = np.deg2rad(58.0)

    def __init__(self, *args, robot_uids="piper", robot_init_qpos_noise=0.01, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        eye_in_hand_pose = sapien_utils.look_at(
            eye=[0.0, 0.0, 0.02], target=[0.0, 0.0, 0.25]
        )
        eye_to_hand_pose = sapien_utils.look_at(
            eye=[0.45, -0.55, 0.45], target=[0.08, 0.0, 0.04]
        )
        return [
            CameraConfig(
                "hand_realsense_d405",
                eye_in_hand_pose,
                self.d405_width,
                self.d405_height,
                self.d405_vfov,
                near=0.07,
                far=2.0,
                mount=self.agent.tcp,
            ),
            CameraConfig(
                "external_realsense_d435",
                eye_to_hand_pose,
                self.d435_width,
                self.d435_height,
                self.d435_vfov,
                near=0.28,
                far=4.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, -0.75, 0.55], target=[0.03, 0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.52, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self._build_objects()
        self.cube = self.objects[0]
        self._build_boxes()

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.015,
            color=[0.05, 0.8, 0.1, 0.6],
            name="box_goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _build_objects(self):
        self.object_specs = [
            dict(kind="cube", name="red_target_cube", size=0.025, color=[0.9, 0.05, 0.05, 1]),
            dict(kind="box", name="green_tall_block", half_sizes=[0.018, 0.026, 0.032], color=[0.05, 0.75, 0.18, 1]),
            dict(kind="box", name="yellow_flat_block", half_sizes=[0.032, 0.02, 0.018], color=[0.95, 0.75, 0.05, 1]),
            dict(kind="box", name="purple_long_block", half_sizes=[0.018, 0.038, 0.022], color=[0.55, 0.2, 0.85, 1]),
            dict(kind="cylinder", name="cyan_round_block", radius=0.022, half_length=0.026, color=[0.0, 0.65, 0.85, 1]),
        ]
        self.objects = []
        for spec in self.object_specs:
            if spec["kind"] == "cube":
                actor = actors.build_cube(
                    self.scene,
                    half_size=spec["size"],
                    color=spec["color"],
                    name=spec["name"],
                    initial_pose=sapien.Pose(p=[0, 0, spec["size"]]),
                )
            elif spec["kind"] == "box":
                actor = actors.build_box(
                    self.scene,
                    half_sizes=spec["half_sizes"],
                    color=spec["color"],
                    name=spec["name"],
                    initial_pose=sapien.Pose(p=[0, 0, spec["half_sizes"][2]]),
                )
            else:
                actor = actors.build_cylinder(
                    self.scene,
                    radius=spec["radius"],
                    half_length=spec["half_length"],
                    color=spec["color"],
                    name=spec["name"],
                    initial_pose=sapien.Pose(p=[0, 0, spec["half_length"]]),
                )
            self.objects.append(actor)

    def _build_boxes(self):
        self.box_specs = [
            dict(center=(0.18, 0.0), inner_half_size=(0.075, 0.075), height=0.06, color=[0.1, 0.35, 0.9, 0.45], name="blue_square_bin"),
            dict(center=(0.18, 0.18), inner_half_size=(0.055, 0.09), height=0.075, color=[0.9, 0.25, 0.12, 0.45], name="orange_tall_bin"),
            dict(center=(0.18, -0.18), inner_half_size=(0.095, 0.05), height=0.045, color=[0.1, 0.75, 0.28, 0.45], name="green_wide_bin"),
        ]
        self.box_parts = []
        for spec in self.box_specs:
            self.box_parts.extend(self._build_single_box(**spec))

    def _build_single_box(self, center, inner_half_size, height, color, name):
        cx, cy = center
        hx, hy = inner_half_size
        t = self.box_wall_thickness
        bottom_z = t / 2
        wall_z = t + height / 2
        return [
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, hy + t, t / 2],
                color=color,
                name=f"{name}_bottom",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx, cy, bottom_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, t / 2, height / 2],
                color=color,
                name=f"{name}_front_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx, cy + hy + t / 2, wall_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, t / 2, height / 2],
                color=color,
                name=f"{name}_back_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx, cy - hy - t / 2, wall_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[t / 2, hy, height / 2],
                color=color,
                name=f"{name}_left_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx - hx - t / 2, cy, wall_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[t / 2, hy, height / 2],
                color=color,
                name=f"{name}_right_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx + hx + t / 2, cy, wall_z]),
            ),
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            qpos = self.agent.keyframes["rest"].qpos
            if self._enhanced_determinism:
                qpos = (
                    self._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                ) + qpos
            qpos[:, -2:] = np.array([0.035, -0.035])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.52, 0, 0]))

            object_xy = torch.tensor(
                [
                    [-0.02, 0.0],
                    [0.045, 0.065],
                    [0.055, -0.065],
                    [-0.07, 0.06],
                    [-0.075, -0.055],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            jitter = (torch.rand((b, len(self.objects), 2), device=self.device) - 0.5) * 0.015
            for i, (actor, spec) in enumerate(zip(self.objects, self.object_specs)):
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = object_xy[i] + jitter[:, i]
                if spec["kind"] == "cube":
                    xyz[:, 2] = spec["size"]
                elif spec["kind"] == "box":
                    xyz[:, 2] = spec["half_sizes"][2]
                else:
                    xyz[:, 2] = spec["half_length"]
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                actor.set_pose(Pose.create_from_pq(xyz, qs))

            target_box = self.box_specs[0]
            box_xy = torch.zeros((b, 2), device=self.device)
            box_xy[:, 0] = torch.rand((b), device=self.device) * 0.12 + 0.13
            box_xy[:, 1] = torch.rand((b), device=self.device) * 0.16 - 0.08
            self.target_box_centers = box_xy

            hx, hy = target_box["inner_half_size"]
            t = self.box_wall_thickness
            h = target_box["height"]
            bottom_z = t / 2
            wall_z = t + h / 2
            part_offsets = torch.tensor(
                [
                    [0.0, 0.0, bottom_z],
                    [0.0, hy + t / 2, wall_z],
                    [0.0, -hy - t / 2, wall_z],
                    [-hx - t / 2, 0.0, wall_z],
                    [hx + t / 2, 0.0, wall_z],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            for part_i, actor in enumerate(self.box_parts[:5]):
                part_xyz = torch.zeros((b, 3), device=self.device)
                part_xyz[:, :2] = box_xy
                part_xyz += part_offsets[part_i]
                actor.set_pose(Pose.create_from_pq(part_xyz))

            goal_xyz = torch.zeros((b, 3), device=self.device)
            goal_xyz[:, :2] = box_xy
            goal_xyz[:, 2] = self.box_wall_thickness + self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        left_force = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.cube
        )
        right_force = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.cube
        )
        force_norms = torch.stack(
            [
                torch.linalg.norm(left_force, axis=1),
                torch.linalg.norm(right_force, axis=1),
            ],
            dim=1,
        )
        obs = dict(
            is_grasped=info["is_grasped"],
            is_cube_in_box=info["is_cube_in_box"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            box_goal_pos=self.goal_site.pose.p,
            left_finger_force=left_force,
            right_finger_force=right_force,
            finger_force_norms=force_norms,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                all_obj_poses=torch.cat([obj.pose.raw_pose for obj in self.objects], dim=1),
                box_centers=torch.cat(
                    [
                        self.target_box_centers,
                        torch.tensor(
                            [spec["center"] for spec in self.box_specs[1:]],
                            device=self.device,
                            dtype=torch.float32,
                        ).flatten().repeat(self.num_envs, 1),
                    ],
                    dim=1,
                ),
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        cube_p = self.cube.pose.p
        if hasattr(self, "target_box_centers"):
            box_centers = self.target_box_centers
        else:
            box_centers = torch.tensor(
                [self.box_center], device=self.device, dtype=torch.float32
            ).repeat(cube_p.shape[0], 1)
        hx, hy = self.box_inner_half_size
        in_xy = (torch.abs(cube_p[:, 0] - box_centers[:, 0]) <= hx) & (
            torch.abs(cube_p[:, 1] - box_centers[:, 1]) <= hy
        )
        in_z = cube_p[:, 2] <= self.box_wall_height + 2 * self.cube_half_size
        is_cube_in_box = in_xy & in_z
        is_obj_near_goal = (
            torch.linalg.norm(self.goal_site.pose.p - cube_p, axis=1) <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_cube_in_box & is_obj_near_goal & is_robot_static,
            "is_cube_in_box": is_cube_in_box,
            "is_obj_near_goal": is_obj_near_goal,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :6], axis=1)
        )
        reward += static_reward * info["is_cube_in_box"]
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


@register_env("PiperPickCupToContainer-v1", max_episode_steps=300)
class PiperPickCupToContainerEnv(BaseEnv):
    """More realistic cup-to-container task used for imitation/RL experiments.

    The task is self-contained: the mug, tray, placemat and distractors are made
    from SAPIEN primitives, so a fresh checkout does not silently download a
    multi-gigabyte asset pack.  The compound mug has a hollow rim and a handle,
    and all task geometry participates in physics.
    """

    SUPPORTED_ROBOTS = ["piper"]
    agent: Piper

    # 60 mm body diameter leaves 5 mm clearance per side in the 70 mm gripper.
    cup_radius = 0.030
    cup_height = 0.105
    cup_wall = 0.003
    cup_handle_reach = 0.026
    container_inner_half_size = (0.090, 0.080)
    container_wall_thickness = 0.006
    container_wall_height = 0.065
    object_spawn_x = (-0.070, -0.035)
    object_spawn_y = (-0.12, 0.12)
    # Conservative placement workspace for the fixed-orientation scripted
    # expert. Wider distributions can be enabled later for learned policies.
    container_spawn_x = (0.08, 0.105)
    container_spawn_y = (-0.055, 0.055)
    min_task_separation = 0.16
    lift_height = 0.07

    def __init__(
        self, *args, robot_uids="piper", robot_init_qpos_noise=0.015,
        domain_randomization=True, **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.domain_randomization = domain_randomization
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        wrist_pose = sapien_utils.look_at(
            eye=[0.0, 0.0, 0.025], target=[0.0, 0.0, 0.24]
        )
        # Similar viewpoint/resolution to the real global D435 acquisition stream.
        global_pose = sapien_utils.look_at(
            eye=[0.48, -0.58, 0.46], target=[0.08, 0.0, 0.04]
        )
        return [
            CameraConfig(
                "observation.images.wrist", wrist_pose, 640, 480,
                np.deg2rad(58.0), near=0.06, far=2.0, mount=self.agent.tcp,
            ),
            CameraConfig(
                "observation.images.global", global_pose, 640, 480,
                np.deg2rad(58.0), near=0.20, far=4.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.58, -0.72, 0.52], target=[0.06, 0, 0.05])
        return CameraConfig("render_camera", pose, 640, 480, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.52, 0, 0]))

    def _load_lighting(self, options: dict):
        """Studio-like three-point lighting with real-time contact shadows."""
        self.scene.set_ambient_light([0.24, 0.25, 0.28])
        self.scene.add_directional_light(
            direction=[0.7, 0.45, -1.0], color=[1.10, 1.02, 0.90],
            shadow=True, shadow_scale=2.0, shadow_near=-1.0, shadow_far=2.0,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light(
            direction=[-0.55, -0.2, -1.0], color=[0.34, 0.42, 0.58],
            shadow=False,
        )
        self.scene.add_point_light(
            position=[0.05, -0.25, 0.80], color=[0.62, 0.60, 0.52],
            shadow=False,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cup = self._build_cup()
        # Compatibility alias for utilities that expect the manipulated actor here.
        self.cube = self.cup
        self.container_parts = self._build_container()
        self._build_scene_dressing()
        self.goal_site = actors.build_sphere(
            self.scene, radius=0.012, color=[0.15, 0.9, 0.25, 0.35],
            name="cup_goal_site", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _build_cup(self):
        builder = self.scene.create_actor_builder()
        ceramic = sapien.render.RenderMaterial(
            base_color=[0.92, 0.92, 0.88, 1.0], roughness=0.28,
        )
        physics = sapien.physx.PhysxMaterial(
            static_friction=0.8, dynamic_friction=0.65, restitution=0.02
        )
        # Bottom plus twelve wall segments: visibly and physically hollow.
        bottom_thickness = 0.008
        cylinder_upright = sapien.Pose(q=[np.sqrt(0.5), 0, np.sqrt(0.5), 0])
        builder.add_cylinder_collision(
            pose=sapien.Pose(
                p=[0, 0, bottom_thickness / 2], q=cylinder_upright.q
            ),
            radius=self.cup_radius, half_length=bottom_thickness / 2,
            material=physics, density=2600, patch_radius=0.002,
            min_patch_radius=0.001,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[0, 0, bottom_thickness / 2], q=cylinder_upright.q
            ),
            radius=self.cup_radius, half_length=bottom_thickness / 2, material=ceramic,
        )
        segment_half_width = self.cup_radius * np.sin(np.pi / 12)
        for i in range(12):
            angle = 2 * np.pi * i / 12
            pose = sapien.Pose(
                p=[
                    (self.cup_radius - self.cup_wall / 2) * np.cos(angle),
                    (self.cup_radius - self.cup_wall / 2) * np.sin(angle),
                    self.cup_height / 2,
                ],
                q=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)],
            )
            half_size = [self.cup_wall / 2, segment_half_width, self.cup_height / 2]
            builder.add_box_collision(
                pose=pose, half_size=half_size, material=physics, density=2600,
                patch_radius=0.002, min_patch_radius=0.001,
            )
            builder.add_box_visual(pose=pose, half_size=half_size, material=ceramic)
        # Squared C-shaped handle; it gives the vision policy a realistic asymmetry.
        handle_x = self.cup_radius + self.cup_handle_reach / 2
        for z in (0.026, 0.064):
            pose = sapien.Pose(p=[handle_x, 0, z])
            half_size = [self.cup_handle_reach / 2, 0.005, 0.005]
            builder.add_box_collision(
                pose=pose, half_size=half_size, material=physics, density=420,
                patch_radius=0.002, min_patch_radius=0.001,
            )
            builder.add_box_visual(pose=pose, half_size=half_size, material=ceramic)
        pose = sapien.Pose(p=[self.cup_radius + self.cup_handle_reach, 0, 0.045])
        half_size = [0.005, 0.005, 0.024]
        builder.add_box_collision(
            pose=pose, half_size=half_size, material=physics, density=420,
            patch_radius=0.002, min_patch_radius=0.001,
        )
        builder.add_box_visual(pose=pose, half_size=half_size, material=ceramic)
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.001]))
        return builder.build(name="white_ceramic_cup")

    def _build_container(self):
        hx, hy = self.container_inner_half_size
        t, h = self.container_wall_thickness, self.container_wall_height
        visual = sapien.render.RenderMaterial(
            base_color=[0.08, 0.32, 0.62, 0.82], roughness=0.34, metallic=0.05,
        )
        physics = sapien.physx.PhysxMaterial(
            static_friction=0.55, dynamic_friction=0.42, restitution=0.03
        )
        specs = [
            ([hx + t, hy + t, t / 2], [0, 0, t / 2]),
            ([hx + t, t / 2, h / 2], [0, hy + t / 2, t + h / 2]),
            ([hx + t, t / 2, h / 2], [0, -hy - t / 2, t + h / 2]),
            ([t / 2, hy, h / 2], [-hx - t / 2, 0, t + h / 2]),
            ([t / 2, hy, h / 2], [hx + t / 2, 0, t + h / 2]),
        ]
        parts = []
        for i, (half_size, offset) in enumerate(specs):
            parts.append(self._build_material_box(
                half_size, visual, physics, f"blue_target_container_{i}",
                "kinematic", sapien.Pose(p=offset), density=920,
            ))
        return parts

    def _build_material_box(
        self, half_size, visual, physics, name, body_type, initial_pose, density=1000,
    ):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=half_size, material=physics, density=density,
            patch_radius=0.002, min_patch_radius=0.001,
        )
        builder.add_box_visual(half_size=half_size, material=visual)
        builder.set_initial_pose(initial_pose)
        if body_type == "kinematic":
            return builder.build_kinematic(name=name)
        if body_type == "static":
            return builder.build_static(name=name)
        return builder.build(name=name)

    def _build_material_cylinder(
        self, radius, half_length, visual, physics, name, initial_pose, density=1000,
    ):
        builder = self.scene.create_actor_builder()
        shape_pose = sapien.Pose(q=[np.sqrt(0.5), 0, np.sqrt(0.5), 0])
        builder.add_cylinder_collision(
            pose=shape_pose, radius=radius, half_length=half_length, material=physics,
            density=density, patch_radius=0.002, min_patch_radius=0.001,
        )
        builder.add_cylinder_visual(
            pose=shape_pose, radius=radius, half_length=half_length, material=visual,
        )
        builder.set_initial_pose(initial_pose)
        return builder.build(name=name)

    def _build_scene_dressing(self):
        # A dark work mat anchors the workspace visually without changing the task.
        rubber_visual = sapien.render.RenderMaterial(
            base_color=[0.026, 0.034, 0.038, 1], roughness=0.91, metallic=0.0,
        )
        rubber_physics = sapien.physx.PhysxMaterial(
            static_friction=1.15, dynamic_friction=0.92, restitution=0.01
        )
        self.work_mat = self._build_material_box(
            [0.30, 0.27, 0.0015], rubber_visual, rubber_physics, "work_mat",
            "kinematic", sapien.Pose(p=[0.08, 0, 0.0015]), density=1100,
        )
        can_visual = sapien.render.RenderMaterial(
            base_color=[0.72, 0.08, 0.045, 1], roughness=0.24, metallic=0.72,
        )
        sponge_visual = sapien.render.RenderMaterial(
            base_color=[0.92, 0.66, 0.08, 1], roughness=0.88, metallic=0.0,
        )
        clutter_physics = sapien.physx.PhysxMaterial(
            static_friction=0.62, dynamic_friction=0.48, restitution=0.08
        )
        self.distractors = [
            self._build_material_cylinder(
                0.024, 0.035, can_visual, clutter_physics, "red_can",
                sapien.Pose(p=[0.08, 0.17, 0.035]), density=280,
            ),
            self._build_material_box(
                [0.025, 0.035, 0.018], sponge_visual, clutter_physics,
                "yellow_sponge", "dynamic", sapien.Pose(p=[0.10, -0.18, 0.018]),
                density=90,
            ),
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self.domain_randomization:
                # Independent exposure/white-balance variation for each sub-scene.
                for scene_i in env_idx.detach().cpu().tolist():
                    brightness = float(self._episode_rng.uniform(0.78, 1.18))
                    warmth = float(self._episode_rng.uniform(-0.035, 0.035))
                    self.scene.sub_scenes[scene_i].render_system.ambient_light = [
                        brightness * (0.24 + warmth),
                        brightness * 0.25,
                        brightness * (0.28 - warmth),
                    ]
            qpos = self.agent.keyframes["rest"].qpos
            noise = self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
            qpos = qpos + noise
            qpos[:, -2:] = np.array([0.035, -0.035])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.52, 0, 0]))

            cup_xy = torch.as_tensor(
                self._episode_rng.uniform(
                    [self.object_spawn_x[0], self.object_spawn_y[0]],
                    [self.object_spawn_x[1], self.object_spawn_y[1]],
                    size=(b, 2),
                ), device=self.device, dtype=torch.float32,
            )
            container_xy = torch.as_tensor(
                self._episode_rng.uniform(
                    [self.container_spawn_x[0], self.container_spawn_y[0]],
                    [self.container_spawn_x[1], self.container_spawn_y[1]],
                    size=(b, 2),
                ), device=self.device, dtype=torch.float32,
            )
            # Re-sample close pairs; a meaningful transport phase is part of the task.
            for _ in range(8):
                too_close = torch.linalg.norm(container_xy - cup_xy, axis=1) < self.min_task_separation
                if not torch.any(too_close):
                    break
                count = int(torch.sum(too_close).item())
                replacement = torch.as_tensor(
                    self._episode_rng.uniform(
                        [self.container_spawn_x[0], self.container_spawn_y[0]],
                        [self.container_spawn_x[1], self.container_spawn_y[1]],
                        size=(count, 2),
                    ), device=self.device, dtype=torch.float32,
                )
                container_xy[too_close] = replacement
            too_close = torch.linalg.norm(container_xy - cup_xy, axis=1) < self.min_task_separation
            container_xy[too_close, 0] = self.container_spawn_x[1]
            container_xy[too_close, 1] = torch.where(
                cup_xy[too_close, 1] >= 0,
                torch.tensor(self.container_spawn_y[0], device=self.device),
                torch.tensor(self.container_spawn_y[1], device=self.device),
            )
            self.target_container_centers = container_xy

            cup_xyz = torch.zeros((b, 3), device=self.device)
            cup_xyz[:, :2] = cup_xy
            cup_xyz[:, 2] = 0.003
            # Keep the handle roughly lateral to the fixed-orientation gripper;
            # unrestricted yaw can put the handle between the fingers and make
            # the scripted baseline geometrically unsolvable.
            # The handle points away from the approaching wrist so it cannot
            # hook onto the palm during release; retain a small yaw variation.
            yaw = torch.as_tensor(
                self._episode_rng.uniform(np.pi - 0.10, np.pi + 0.10, size=b),
                device=self.device, dtype=torch.float32,
            )
            cup_q = torch.zeros((b, 4), device=self.device)
            cup_q[:, 0], cup_q[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            self.cup.set_pose(Pose.create_from_pq(cup_xyz, cup_q))

            t, h = self.container_wall_thickness, self.container_wall_height
            hx, hy = self.container_inner_half_size
            offsets = torch.tensor([
                [0, 0, t / 2], [0, hy + t / 2, t + h / 2],
                [0, -hy - t / 2, t + h / 2], [-hx - t / 2, 0, t + h / 2],
                [hx + t / 2, 0, t + h / 2],
            ], device=self.device, dtype=torch.float32)
            for i, part in enumerate(self.container_parts):
                p = torch.zeros((b, 3), device=self.device)
                p[:, :2] = container_xy
                p += offsets[i]
                part.set_pose(Pose.create_from_pq(p))

            goal = torch.zeros((b, 3), device=self.device)
            goal[:, :2] = container_xy
            goal[:, 2] = t
            self.goal_site.set_pose(Pose.create_from_pq(goal))

            # Small clutter randomization, kept outside the nominal pick/place corridor.
            distractor_xy = torch.tensor([[0.08, 0.17], [0.10, -0.18]], device=self.device)
            distractor_z = [0.035, 0.018]
            for i, actor in enumerate(self.distractors):
                p = torch.zeros((b, 3), device=self.device)
                jitter = torch.as_tensor(
                    self._episode_rng.uniform(-0.0125, 0.0125, size=(b, 2)),
                    device=self.device, dtype=torch.float32,
                )
                p[:, :2] = distractor_xy[i] + jitter
                p[:, 2] = distractor_z[i]
                actor.set_pose(Pose.create_from_pq(p))

    def evaluate(self):
        p = self.cup.pose.p
        center = self.target_container_centers
        hx, hy = self.container_inner_half_size
        margin = self.cup_radius + 0.004
        inside_xy = (
            (torch.abs(p[:, 0] - center[:, 0]) <= hx - margin)
            & (torch.abs(p[:, 1] - center[:, 1]) <= hy - margin)
        )
        inside_z = (p[:, 2] >= 0) & (p[:, 2] <= self.container_wall_height + 0.025)
        is_in_container = inside_xy & inside_z
        is_grasped = self.agent.is_grasping(self.cup)
        linear_speed = torch.linalg.norm(self.cup.linear_velocity, axis=1)
        angular_speed = torch.linalg.norm(self.cup.angular_velocity, axis=1)
        is_object_static = (linear_speed < 0.06) & (angular_speed < 0.35)
        cup_up_axis = self.cup.pose.to_transformation_matrix()[:, :3, 2]
        is_upright = cup_up_axis[:, 2] > 0.70
        is_released = ~is_grasped
        is_robot_static = self.agent.is_static(0.2)
        is_lifted = p[:, 2] > self.lift_height
        success = is_in_container & is_upright & is_released & is_object_static & is_robot_static
        return dict(
            success=success, is_grasped=is_grasped, is_lifted=is_lifted,
            is_in_container=is_in_container, is_released=is_released,
            is_upright=is_upright, is_object_static=is_object_static,
            is_robot_static=is_robot_static,
        )

    def _get_obs_extra(self, info: dict):
        left_force = self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.cup)
        right_force = self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.cup)
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            cup_pose=self.cup.pose.raw_pose,
            container_pos=self.target_container_centers,
            tcp_to_cup_pos=self.cup.pose.p - self.agent.tcp_pose.p,
            cup_to_container_pos=self.goal_site.pose.p - self.cup.pose.p,
            left_finger_force=left_force, right_finger_force=right_force,
            finger_force_norms=torch.stack([
                torch.linalg.norm(left_force, axis=1),
                torch.linalg.norm(right_force, axis=1),
            ], dim=1),
            **{key: info[key] for key in (
                "is_grasped", "is_lifted", "is_in_container",
                "is_released", "is_upright", "is_object_static",
            )},
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_dist = torch.linalg.norm(self.cup.pose.p - self.agent.tcp_pose.p, axis=1)
        reward = 1.5 * (1 - torch.tanh(6 * tcp_dist))
        reward += 1.5 * info["is_grasped"]
        lift_progress = torch.clamp(self.cup.pose.p[:, 2] / self.lift_height, 0, 1)
        reward += lift_progress * info["is_grasped"]
        xy_dist = torch.linalg.norm(
            self.cup.pose.p[:, :2] - self.target_container_centers, axis=1
        )
        reward += 2.0 * (1 - torch.tanh(7 * xy_dist)) * info["is_lifted"]
        reward += 2.0 * info["is_in_container"]
        reward += 1.0 * (info["is_in_container"] & info["is_released"])
        reward -= 0.005 * torch.sum(torch.square(action), dim=1)
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 10.0
