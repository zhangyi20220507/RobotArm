"""ACT/RL action composition preserves the base policy's safety contract."""

import math
import unittest

import numpy as np

from robotarm.modeling.act_rl_residual import blend_act_rl


class ResidualActionTests(unittest.TestCase):
    def test_correction_is_bounded_per_axis(self):
        base = np.zeros(7)
        proposed = np.asarray([1, -1, 0.1, 0, 0, 0, 0.02])
        result = blend_act_rl(
            base, proposed, gain=1.0,
            max_joint_delta_deg=0.5, max_gripper_delta_mm=1.0,
        )
        self.assertAlmostEqual(result[0], math.radians(0.5))
        self.assertAlmostEqual(result[1], -math.radians(0.5))
        self.assertAlmostEqual(result[6], 0.001)

    def test_zero_gain_and_bad_actions(self):
        base = np.arange(7, dtype=float) / 100
        self.assertTrue(np.array_equal(
            blend_act_rl(base, base + 1, gain=0,
                         max_joint_delta_deg=0.5, max_gripper_delta_mm=1),
            base,
        ))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            blend_act_rl(base, [math.nan] * 7, gain=1,
                         max_joint_delta_deg=0.5, max_gripper_delta_mm=1)
