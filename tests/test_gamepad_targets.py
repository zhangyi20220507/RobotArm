"""Regression tests for Cartesian gamepad target locking."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from robotarm.teleop.gamepad.targets import cartesian_pose, solve_cartesian_target
from robotarm.core.kinematics.dls_ik import IKError, solve_ik_dls


class _Robot:
    def __init__(self, pose: list[float]) -> None:
        self.pose = pose

    def fk(self, _joints: list[float]) -> list[float]:
        return self.pose[:]


class CartesianTargetTests(unittest.TestCase):
    def test_retry_shrinks_only_commanded_axis(self):
        current = [.3, 0., .25, 0., 0., 0.]
        held = [.3, .001, .251, 0., 0., 0.]
        seen = []
        def solver(_fk, target, seed, bounds, **kwargs):
            seen.append(target[:])
            if len(seen) == 1:
                raise IKError("stalled")
            return seed[:]
        with patch("robotarm.teleop.gamepad.targets.solve_ik_dls", side_effect=solver):
            _, accepted = solve_cartesian_target(
                _Robot(current), [0.]*6, held, [1.,0.,0.,0.,0.,0.],
                [(-math.pi,math.pi)]*6, [.005]*3, .01, .02, .1, .2, [0.]*6)
        self.assertAlmostEqual(accepted[0], .3025)
        self.assertEqual(accepted[1:], held[1:])

    def test_time_budget_rejects_without_returning_solution(self):
        with self.assertRaisesRegex(IKError, "time budget"):
            solve_ik_dls(lambda q: q[:], [0.]*6, [0.]*6,
                         [(-1.,1.)]*6, max_time_s=0.)

    def test_translation_does_not_compute_unused_rotation_error(self):
        with patch("robotarm.core.kinematics.dls_ik.pose_error", side_effect=AssertionError):
            result = solve_ik_dls(lambda q: q[:], [.01,0.,0.,0.,0.,0.],
                                  [0.]*6, [(-1.,1.)]*6, active_axes=(0,1,2))
        self.assertLess(abs(result[0] - .01), .0008)

    def test_x_jog_keeps_other_components_in_target_buffer(self):
        current = [0.30, -0.02, 0.25, 0.10, 0.20, -0.30]
        held = [0.31, -0.01, 0.26, 0.40, 0.50, -0.60]
        captured: dict[str, object] = {}

        def fake_solve(_fk, target, seed, _bounds, **kwargs):
            captured["target"] = list(target)
            captured["active_axes"] = kwargs["active_axes"]
            return list(seed)

        with patch(
            "robotarm.teleop.gamepad.targets.solve_ik_dls",
            side_effect=fake_solve,
        ):
            _, requested = solve_cartesian_target(
                _Robot(current),
                [0.0] * 6,
                held,
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [(-math.pi, math.pi)] * 6,
                [0.005, 0.005, 0.002],
                math.radians(1.0),
                0.020,
                math.radians(5.0),
                math.radians(10.0),
                [0.0] * 6,
            )

        self.assertAlmostEqual(requested[0], 0.315)
        self.assertEqual(requested[1:], held[1:])
        self.assertEqual(captured["target"], requested)
        self.assertEqual(captured["active_axes"], (0, 1, 2))

    def test_yaw_jog_holds_xyz_roll_and_pitch(self):
        current = [0.30, -0.02, 0.25, 0.10, 0.20, -0.30]
        held = [0.31, -0.01, 0.26, 0.40, 0.50, -0.60]

        with patch(
            "robotarm.teleop.gamepad.targets.solve_ik_dls",
            return_value=[0.0] * 6,
        ):
            _, requested = solve_cartesian_target(
                _Robot(current),
                [0.0] * 6,
                held,
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [(-math.pi, math.pi)] * 6,
                [0.005, 0.005, 0.002],
                math.radians(1.0),
                0.020,
                math.radians(5.0),
                math.radians(10.0),
                [0.0] * 6,
            )

        self.assertEqual(requested[:5], held[:5])
        # The existing lead guard re-anchors an active axis when its queued
        # target is behind feedback in the newly requested direction.
        self.assertAlmostEqual(requested[5], current[5] + math.radians(1.0))

    def test_tcp_offset_follows_flange_local_z(self):
        robot = _Robot([0.30, -0.02, 0.25, 0.0, 0.0, 0.0])
        pose = cartesian_pose(robot, [0.0] * 6, 0.1425)
        self.assertAlmostEqual(pose[0], 0.30)
        self.assertAlmostEqual(pose[1], -0.02)
        self.assertAlmostEqual(pose[2], 0.3925)

        robot.pose[4] = math.pi / 2
        pose = cartesian_pose(robot, [0.0] * 6, 0.1425)
        self.assertAlmostEqual(pose[0], 0.4425)
        self.assertAlmostEqual(pose[1], -0.02)
        self.assertAlmostEqual(pose[2], 0.25)


if __name__ == "__main__":
    unittest.main()
