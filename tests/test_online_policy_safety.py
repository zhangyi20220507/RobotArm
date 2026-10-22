"""Pure safety helpers used by the online policy runner."""

from __future__ import annotations

import math
import json
import tempfile
import unittest
from argparse import Namespace
from collections import deque
from pathlib import Path
from unittest.mock import patch

from robotarm.modeling.online_policy import (
    act_action_envelope_failures,
    act_action_within_envelope,
    act_tracking_action,
    clamp_step,
    action_queue_depth,
    action_was_clamped,
    approval_required,
    validate_approval,
    validate_execution_approval,
    validate_motion_limits,
)


class _ActPolicy:
    def __init__(self):
        self._action_queue = deque([[0.0] * 7, [0.0] * 7])


class OnlinePolicySafetyTests(unittest.TestCase):
    def setUp(self):
        self.bounds = [(-math.pi, math.pi)] * 6
        self.state = [0.0] * 7

    def test_act_queue_depth_uses_act_private_queue(self):
        self.assertEqual(action_queue_depth(_ActPolicy()), 2)

    def test_clamp_detection_has_numeric_tolerance(self):
        self.assertFalse(action_was_clamped([0.0] * 7, [1e-7] * 7))
        self.assertTrue(action_was_clamped([0.0] * 7, [1e-4] + [0.0] * 6))

    def limited_target(self, tracking, state, previous=None):
        return clamp_step(tracking, state, math.radians(4), .008, .1,
                          self.bounds, previous, math.radians(6), .016)

    def test_tolerated_gripper_bound_correction_does_not_hold_act_queue(self):
        for width, endpoint in ((-.00015337113291025162, 0.), (.1005, .1)):
            with self.subTest(width=width):
                raw = [0.] * 6 + [width]
                state = [0.] * 6 + [endpoint]
                tracking = act_tracking_action(raw, .1, .001)
                target = self.limited_target(tracking, state)
                self.assertEqual(tracking[6], endpoint)
                self.assertEqual(raw[6], width)  # Raw prediction retained.
                self.assertTrue(action_was_clamped(raw, target))
                self.assertFalse(action_was_clamped(tracking, target))

    def test_boundary_correction_still_holds_real_joint_rate_limit(self):
        raw = [math.radians(5)] + [0.] * 5 + [-.000153]
        tracking = act_tracking_action(raw, .1, .001)
        target = self.limited_target(tracking, self.state)
        self.assertTrue(action_was_clamped(tracking, target))
        target = self.limited_target(tracking, target, target)
        self.assertFalse(action_was_clamped(tracking, target))

    def test_boundary_correction_still_holds_real_gripper_rate_limit(self):
        raw = [0.] * 6 + [-.000153]
        state = [0.] * 6 + [.017]
        tracking = act_tracking_action(raw, .1, .001)
        self.assertTrue(action_was_clamped(tracking, self.limited_target(tracking, state)))

    def test_correction_does_not_hide_out_of_tolerance_or_large_raw_gap(self):
        for raw, state in (([0.] * 6 + [-.0011], self.state),
                           ([0.] * 6 + [.1011], [0.] * 6 + [.1]),
                           ([0.] * 6 + [-.000153], [0.] * 6 + [.018])):
            with self.subTest(raw=raw):
                self.assertFalse(act_action_within_envelope(
                    raw, state, self.bounds, .1, math.radians(9), .018, .001))
                if abs(raw[6]) > .001:
                    self.assertEqual(act_tracking_action(raw, .1, .001)[6], raw[6])

    def test_act_envelope_accepts_dataset_scale_target(self):
        raw = [math.radians(8.5), 0.0, 0.0, 0.0, 0.0, 0.0, 0.017]
        self.assertTrue(
            act_action_within_envelope(
                raw, self.state, self.bounds, 0.1,
                math.radians(9.0), 0.018,
            )
        )

    def test_act_envelope_allows_small_gripper_bound_overshoot(self):
        raw_below_zero = [0.0] * 6 + [-0.00029]
        raw_above_max = [0.0] * 6 + [0.1005]
        self.assertTrue(
            act_action_within_envelope(
                raw_below_zero, self.state, self.bounds, 0.1,
                math.radians(9.0), 0.018, 0.001,
            )
        )
        self.assertTrue(
            act_action_within_envelope(
                raw_above_max, [0.0] * 6 + [0.1], self.bounds, 0.1,
                math.radians(9.0), 0.018, 0.001,
            )
        )

        too_far = [0.0] * 6 + [-0.0011]
        self.assertFalse(
            act_action_within_envelope(
                too_far, self.state, self.bounds, 0.1,
                math.radians(9.0), 0.018, 0.001,
            )
        )

    def test_act_envelope_rejects_large_or_out_of_bounds_target(self):
        too_far = [math.radians(9.1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertFalse(
            act_action_within_envelope(
                too_far, self.state, self.bounds, 0.1,
                math.radians(9.0), 0.018,
            )
        )
        out_of_bounds = [math.pi + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertFalse(
            act_action_within_envelope(
                out_of_bounds, self.state, self.bounds, 0.1,
                math.radians(180.0), 0.018,
            )
        )

    def test_act_envelope_explains_joint_bound_and_gap_failures(self):
        raw = [math.pi + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        reasons = act_action_envelope_failures(
            raw, self.state, self.bounds, 0.1,
            math.radians(9.0), 0.018,
        )
        self.assertTrue(any("raw joint bound" in reason for reason in reasons))
        self.assertTrue(any("raw joint gap J1" in reason for reason in reasons))

    def test_limits_are_revalidated_after_policy_override(self):
        args = Namespace(
            max_joint_step_deg=3.0,
            max_joint_lead_deg=1.0,
            max_gripper_step_mm=1.0,
            max_gripper_lead_mm=2.0,
            max_act_clamp_hold_steps=10,
            max_act_raw_joint_gap_deg=9.0,
            max_act_raw_gripper_gap_mm=18.0,
            max_act_raw_gripper_bound_tolerance_mm=1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "joint-lead"):
            validate_motion_limits(args, "act")
        args.max_joint_lead_deg = 3.0
        args.max_act_raw_gripper_bound_tolerance_mm = -0.1
        with self.assertRaisesRegex(RuntimeError, "tolerance-mm"):
            validate_motion_limits(args, "act")

    def test_legacy_act_approval_without_raw_watchdog_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.json"
            path.write_text(json.dumps({
                "approved": True,
                "checkpoint_sha256": "0" * 64,
                "policy_type": "act",
                "runtime": {},
            }))
            with self.assertRaisesRegex(
                RuntimeError, "predates raw-action shadow validation"
            ):
                validate_approval(path, "0" * 64, "act", {})

    def test_approval_required_must_be_a_boolean(self):
        with patch(
            "robotarm.modeling.online_policy.defaults",
            return_value={"approval-required": False},
        ):
            self.assertFalse(approval_required())

        with patch(
            "robotarm.modeling.online_policy.defaults",
            return_value={"approval-required": True},
        ):
            self.assertTrue(approval_required())

        with patch(
            "robotarm.modeling.online_policy.defaults",
            return_value={"approval-required": "false"},
        ):
            with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                approval_required()

    def test_disabled_approval_skips_missing_approval_file(self):
        with (
            patch(
                "robotarm.modeling.online_policy.defaults",
                return_value={"approval-required": False},
            ),
            patch("robotarm.modeling.online_policy.validate_approval") as validate,
        ):
            checked = validate_execution_approval(
                Path("/missing/approval.json"), "0" * 64, "act", {}
            )
        self.assertFalse(checked)
        validate.assert_not_called()

    def test_required_approval_rejects_missing_file(self):
        with patch(
            "robotarm.modeling.online_policy.defaults",
            return_value={"approval-required": True},
        ):
            with self.assertRaisesRegex(RuntimeError, "no approval"):
                validate_execution_approval(
                    Path("/missing/approval.json"), "0" * 64, "act", {}
                )


if __name__ == "__main__":
    unittest.main()
