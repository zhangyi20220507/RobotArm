from __future__ import annotations

import math
import unittest

from robotarm.modeling.hg_dagger import ClutchedLeaderMapping


class HGDAggerTests(unittest.TestCase):
    def test_clutch_has_no_takeover_jump_and_preserves_relative_motion(self):
        mapping = ClutchedLeaderMapping([1] * 6, [0.0] * 6, 0.1)
        leader = [math.radians(20.0)] * 6 + [0.04]
        follower = [math.radians(-10.0)] * 6 + [0.02]
        bounds = [(-math.pi, math.pi)] * 6

        mapping.engage(leader, follower)
        engaged = mapping.target(leader, bounds)
        for actual, expected in zip(engaged, follower):
            self.assertAlmostEqual(actual, expected)

        moved = [value + math.radians(3.0) for value in leader[:6]] + [0.05]
        target = mapping.target(moved, bounds)
        for actual, start in zip(target[:6], follower[:6]):
            self.assertAlmostEqual(actual - start, math.radians(3.0))
        self.assertAlmostEqual(target[6], 0.03)

    def test_target_is_clipped_to_follower_bounds(self):
        mapping = ClutchedLeaderMapping([1] * 6, [0.0] * 6, 0.07)
        mapping.engage([0.0] * 7, [0.0] * 7)
        target = mapping.target([1.0] * 6 + [0.2], [(-0.5, 0.5)] * 6)
        self.assertEqual(target[:6], [0.5] * 6)
        self.assertEqual(target[6], 0.07)


if __name__ == "__main__":
    unittest.main()
