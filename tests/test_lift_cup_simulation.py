"""CPU physics checks; run with the project's simulation dependencies installed."""

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("mani_skill"), "requires simulation extra")
class LiftCupSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import robotarm.simulation.environment  # noqa: F401

        cls.env = gym.make(
            "PiperLiftCup-v1", cup_pose="mixed", sim_backend="cpu", render_backend="none",
            obs_mode="state", control_mode="pd_ee_delta_pose", robot_init_qpos_noise=0,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_seed_repeats_pose_and_starts_above_table_without_success(self):
        e = self.env.unwrapped
        _, info = self.env.reset(seed=42)
        pose = e.cup.pose.raw_pose.clone()
        self.env.reset(seed=43)
        _, repeated = self.env.reset(seed=42)
        np.testing.assert_allclose(pose.cpu(), e.cup.pose.raw_pose.cpu(), atol=1e-6)
        self.assertAlmostEqual(float(info["cup_clearance"][0]), 0.001, places=5)
        self.assertFalse(bool(repeated["success"][0]))
        self.assertEqual(int(repeated["hold_count"][0]), 0)
        self.assertEqual(self.env.action_space.shape, (8,))

    def test_airborne_cup_without_finger_contact_is_not_success(self):
        import torch
        from mani_skill.utils.structs.pose import Pose

        self.env.reset(seed=0)
        e = self.env.unwrapped
        position = e.cup.pose.p.clone()
        position[:, 2] = 0.3
        e.cup.set_pose(Pose.create_from_pq(position, e.cup.pose.q))
        e.cup.set_linear_velocity(torch.zeros((1, 3)))
        e.cup.set_angular_velocity(torch.zeros((1, 3)))
        for _ in range(25):
            info = e.evaluate()
            self.assertTrue(bool(info["is_lifted"][0]))
            self.assertFalse(bool(info["is_grasped"][0]))
            self.assertFalse(bool(info["success"][0]))
            self.assertEqual(int(info["hold_count"][0]), 0)

    def test_idle_physics_produces_finite_observations_and_reward(self):
        self.env.reset(seed=2)
        action = np.array([0, 0, 0, 0, 0, 0, 0.035, -0.035], dtype=np.float32)
        for _ in range(20):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertTrue(np.isfinite(obs.cpu().numpy()).all())
            self.assertTrue(np.isfinite(reward.cpu().numpy()).all())
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated))
        self.assertLess(abs(float(info["cup_clearance"][0])), 0.015)

    def test_success_requires_consecutive_physics_steps_not_evaluation_calls(self):
        import torch
        from unittest.mock import patch
        from mani_skill.utils.structs.pose import Pose

        self.env.reset(seed=0)
        e = self.env.unwrapped
        position = e.cup.pose.p.clone()
        position[:, 2] = 0.3
        e.cup.set_pose(Pose.create_from_pq(position, e.cup.pose.q))
        e.cup.set_linear_velocity(torch.zeros((1, 3)))
        e.cup.set_angular_velocity(torch.zeros((1, 3)))

        def forces(link, cup):
            return torch.tensor([[1.0 if link is e.agent.finger1_link else -1.0, 0, 0]])

        with patch.object(e.scene, "get_pairwise_contact_forces", side_effect=forces):
            e._elapsed_steps += 1
            for _ in range(20):
                self.assertFalse(bool(e.evaluate()["success"][0]))
            self.assertEqual(int(e.evaluate()["hold_count"][0]), 1)
            for _ in range(e.hold_steps - 1):
                e._elapsed_steps += 1
                info = e.evaluate()
            self.assertTrue(bool(info["success"][0]))
        e._elapsed_steps += 1
        self.assertFalse(bool(e.evaluate()["success"][0]))
        self.assertEqual(int(e.evaluate()["hold_count"][0]), 0)

    def test_invalid_pose_is_rejected(self):
        from robotarm.simulation.environment import PiperLiftCupEnv

        with self.assertRaises(ValueError):
            PiperLiftCupEnv(cup_pose="invalid")

    def test_waypoint_stagnation_guard_stops_only_when_progress_stalls(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from robotarm.simulation.scripted_pick_place import step_to, OPEN_GRIPPER

        for guard, increment, expected in ((8, 0.0, 9), (None, 0.0, 40), (8, 0.001, 40)):
            with self.subTest(guard=guard, increment=increment):
                position = np.zeros(3, dtype=np.float32)
                calls = []

                def step(action):
                    calls.append(action)
                    position[0] += increment
                    return None, 0, False, False, {}

                env = SimpleNamespace(step=step)
                with patch("robotarm.simulation.scripted_pick_place.current_tcp",
                           side_effect=lambda env: position.copy()):
                    step_to(env, [0.1, 0, 0], OPEN_GRIPPER, max_steps=40,
                            stagnation_steps=guard)
                self.assertEqual(len(calls), expected)


if __name__ == "__main__":
    unittest.main()
