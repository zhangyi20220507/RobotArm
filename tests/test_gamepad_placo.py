"""Offline integration tests: real Placo and SDK FK, no CAN/camera access."""
import math
import unittest

try:
    import numpy as np
    from pyAgxArm.utiles.mdh_kinematics import get_mdh, fk_from_mdh
    from robotarm.teleop.gamepad.placo_ik import GamepadPlacoIK
    from robotarm.teleop.xr.piper_ik import PiperIK, PIPER_JOINT_MIN, PIPER_JOINT_MAX
except ImportError:
    GamepadPlacoIK = None

from robotarm.core.kinematics.dls_ik import IKError
from robotarm.teleop.gamepad.targets import cartesian_pose, solve_cartesian_target


class Robot:
    def fk(self, joints):
        return fk_from_mdh(get_mdh("piper"), joints)


@unittest.skipIf(GamepadPlacoIK is None, "optional Placo/SDK dependencies unavailable")
class PlacoTests(unittest.TestCase):
    def setUp(self):
        self.robot = Robot()
        self.solver = GamepadPlacoIK(1 / 30, 0.1425)
        self.joints = np.radians([0, 70, -100, 0, 40, 0]).tolist()
        self.bounds = list(zip(PIPER_JOINT_MIN, PIPER_JOINT_MAX))

    def test_six_axes_both_directions(self):
        for axis in range(6):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    inputs = [0.] * 6
                    inputs[axis] = sign
                    joints, target = solve_cartesian_target(
                        self.robot, self.joints, cartesian_pose(self.robot, self.joints, .1425),
                        inputs, self.bounds, [.0033, .0033, .0013], math.radians(.66),
                        .02, math.radians(5), math.radians(10), self.joints, .1425, self.solver,
                    )
                    achieved = cartesian_pose(self.robot, joints, .1425)
                    self.assertLessEqual(np.linalg.norm(np.array(achieved[:3]) - target[:3]), .0008)
                    self.assertTrue(all(lo - 1e-6 <= q <= hi + 1e-6 for q, (lo, hi) in zip(joints, self.bounds)))

    def test_tcp_frame_does_not_change_xr_model(self):
        original = PiperIK(1 / 30, False)
        flange = original.flange_pose(np.array(self.joints))
        self.solver.ik.set_state(np.array(self.joints))
        tcp = self.solver.ik.robot.get_T_world_frame("link6")[:3, 3]
        expected = cartesian_pose(self.robot, self.joints, .1425)
        self.assertLess(np.linalg.norm(tcp - expected[:3]), .0002)
        self.assertAlmostEqual(np.linalg.norm(tcp - flange[:3]), .1425)

    def test_unreachable_target_rejected(self):
        with self.assertRaises(IKError):
            self.solver(lambda q: cartesian_pose(self.robot, q, .1425),
                        [10., 10., 10., 0., 0., 0.], self.joints, self.bounds,
                        active_axes=(0, 1, 2))

    def test_posture_anchor_persists_and_can_reset(self):
        pose = cartesian_pose(self.robot, self.joints, .1425)
        fk = lambda q: cartesian_pose(self.robot, q, .1425)
        self.solver(fk, pose, self.joints, self.bounds, active_axes=(0, 1, 2))
        anchor = self.solver.posture_anchor.copy()
        changed = self.joints[:]
        changed[0] += .01
        self.solver(fk, fk(changed), changed, self.bounds, active_axes=(0, 1, 2))
        np.testing.assert_array_equal(self.solver.posture_anchor, anchor)
        self.solver.reset_anchor()
        self.solver(fk, fk(changed), changed, self.bounds, active_axes=(0, 1, 2))
        np.testing.assert_array_equal(self.solver.posture_anchor, changed)
