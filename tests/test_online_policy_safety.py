"""Pure safety helpers used by the online policy runner."""

from __future__ import annotations

import math
import json
import tempfile
import unittest
from itertools import count
from argparse import Namespace
from collections import deque
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from robotarm.modeling.online_policy import (
    act_action_envelope_failures,
    act_action_within_envelope,
    act_backpressure_required,
    act_joint_gap_failure,
    act_subgoal_errors,
    act_tracking_action,
    bounded_act_subgoal,
    clamp_step,
    engage_expert_with_follower_hold,
    action_queue_depth,
    action_was_clamped,
    approval_required,
    isolate_leader_home_failure,
    home_arm_to_zero,
    inference_timing_messages,
    KeyboardIntentTracker,
    latch_act_subgoal,
    limit_joint_target_acceleration,
    move_gripper_and_wait,
    prepare_js_control,
    reset_act_to_home,
    update_act_backpressure_progress,
    update_act_backpressure_target,
    update_act_raw_gap_streak,
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

    def test_takeover_holds_follower_before_blocking_leader_switch(self):
        events = []
        robot = Mock()
        gripper = Mock()
        expert = Mock()
        state = [0.1] * 6 + [0.04]
        robot.move_js.side_effect = lambda q: events.append(("hold_joints", q))
        gripper.move_gripper_m.side_effect = lambda width, force: events.append(("hold_gripper", width))
        expert.engage.side_effect = lambda q, bounds: events.append(("leader_role_switch", q)) or "sample"
        result = engage_expert_with_follower_hold(robot, gripper, expert, state, self.bounds, 1.0, 0.1)
        self.assertEqual(result, "sample")
        self.assertEqual(events, [("hold_joints", state[:6]), ("hold_gripper", 0.04),
                                  ("leader_role_switch", state)])

    def test_takeover_rejects_invalid_state_before_any_motion_command(self):
        robot, gripper, expert = Mock(), Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "finite 7D"):
            engage_expert_with_follower_hold(robot, gripper, expert, [math.nan] * 7, self.bounds, 1, 0.1)
        robot.move_js.assert_not_called()
        expert.engage.assert_not_called()

    def test_inference_timing_uses_soft_warning_before_hard_stop(self):
        warnings, failures = inference_timing_messages(229.3, 60.0, 500.0)
        self.assertEqual(
            warnings,
            ["inference 229.3ms > warning threshold 60.0ms"],
        )
        self.assertEqual(failures, [])

        warnings, failures = inference_timing_messages(501.0, 60.0, 500.0)
        self.assertEqual(warnings, [])
        self.assertEqual(failures, ["inference 501.0ms > 500.0ms"])

    def test_act_raw_joint_gap_streak_is_available_for_warning_telemetry(self):
        warning = act_joint_gap_failure(
            [math.radians(12.69)] + [0.0] * 6,
            self.state,
            math.radians(12.0),
        )
        self.assertIn("12.69deg > 12.00deg", warning)

        hard = act_joint_gap_failure(
            [math.radians(15.5)] + [0.0] * 6,
            self.state,
            math.radians(15.0),
        )
        steps, failure = update_act_raw_gap_streak(hard, 0, 3)
        self.assertEqual((steps, failure), (1, None))
        steps, failure = update_act_raw_gap_streak(hard, steps, 3)
        self.assertEqual((steps, failure), (2, None))
        steps, failure = update_act_raw_gap_streak(hard, steps, 3)
        self.assertEqual(steps, 3)
        self.assertIn("persisted for 3 consecutive ticks", failure)
        self.assertEqual(update_act_raw_gap_streak(None, steps, 3), (0, None))

    def test_act_queue_depth_uses_act_private_queue(self):
        self.assertEqual(action_queue_depth(_ActPolicy()), 2)

    def test_r_resets_once_and_next_x_needs_a_fresh_press(self):
        keys = KeyboardIntentTracker(0.3)
        self.assertEqual(keys.update([("x", 1.0)], 1.0), (True, False, False, False, False))
        self.assertEqual(keys.update([("r", 1.1)], 1.1), (False, False, True, False, False))
        keys.after_home(2.0)
        self.assertEqual(keys.update([("r", 2.1)], 2.1), (False, False, False, False, False))
        self.assertEqual(keys.update([("x", 2.2)], 2.2), (True, False, False, False, False))
        self.assertEqual(keys.update([("r", 3.0)], 3.0), (False, False, True, False, False))

    def test_v_toggles_recording_with_key_repeat_debounce(self):
        keys = KeyboardIntentTracker(0.3)
        self.assertEqual(
            keys.update([("v", 1.0)], 1.0),
            (False, False, False, False, True),
        )
        self.assertEqual(
            keys.update([("v", 1.1)], 1.1),
            (False, False, False, False, False),
        )
        self.assertEqual(
            keys.update([("v", 2.0)], 2.0),
            (False, False, False, False, True),
        )

    def test_i_toggles_hg_dagger_with_key_repeat_debounce(self):
        keys = KeyboardIntentTracker(0.3)
        self.assertTrue(keys.intervention_toggle([("i", 1.0)]))
        self.assertFalse(keys.intervention_toggle([("i", 1.1)]))
        self.assertTrue(keys.intervention_toggle([("i", 2.0)]))

    def test_i_allows_quick_return_without_waiting_for_x_debounce(self):
        keys = KeyboardIntentTracker(0.3)
        self.assertTrue(keys.intervention_toggle([("i", 1.0)]))
        self.assertFalse(keys.intervention_toggle([("i", 1.1)]))
        self.assertTrue(keys.intervention_toggle([("i", 1.5)]))

    def test_d_discards_hg_trial_with_key_repeat_debounce(self):
        keys = KeyboardIntentTracker(0.3)
        self.assertTrue(keys.trial_discard([("d", 1.0)]))
        self.assertFalse(keys.trial_discard([("d", 1.1)]))
        self.assertTrue(keys.trial_discard([("d", 2.0)]))

    def test_r_reset_releases_before_home_and_closes_before_rearming(self):
        events = []
        robot = SimpleNamespace(move_js=lambda joints: events.append(("hold", tuple(joints))))
        gripper = SimpleNamespace(move_gripper_m=lambda width, force: events.append(("grip", width)))
        args = SimpleNamespace(
            feedback_timeout=0.5, gripper_max_mm=100.0, gripper_force=1.0,
            home_speed_percent=10, home_timeout=60.0,
        )
        with patch(
            "robotarm.modeling.online_policy.read_valid_joints",
            return_value=[0.1] * 6,
        ), patch(
            "robotarm.modeling.online_policy.base.wait_gripper_width",
            side_effect=[0.02, 0.079, 0.002],
        ), patch(
            "robotarm.modeling.online_policy.home_arm_to_zero",
            side_effect=lambda *args: events.append(("home", args[3])),
        ), patch(
            "robotarm.modeling.online_policy.prepare_js_control",
            side_effect=lambda *args: events.append(("rearm", args[3])),
        ), patch("robotarm.modeling.online_policy.time.sleep"):
            width = reset_act_to_home(
                robot, {}, gripper, args, 2, lambda: False,
            )
        self.assertEqual(width, 0.002)
        self.assertEqual(events, [
            ("hold", (0.1,) * 6),
            ("grip", 0.08),
            ("home", 0.079),
            ("rearm", 0.002),
        ])

    def test_gripper_reset_checks_emergency_before_command(self):
        gripper = SimpleNamespace(move_gripper_m=lambda *args: self.fail("gripper moved"))
        with self.assertRaisesRegex(InterruptedError, "emergency stop"):
            move_gripper_and_wait(gripper, 0.08, 1.0, 0.005, 8.0, 0.5, lambda: True)

    def test_r_reset_does_not_home_without_confirmed_release(self):
        robot = SimpleNamespace(move_js=lambda joints: None)
        args = SimpleNamespace(
            feedback_timeout=0.5, gripper_max_mm=100.0, gripper_force=1.0,
            home_speed_percent=10, home_timeout=60.0,
        )
        with patch(
            "robotarm.modeling.online_policy.read_valid_joints",
            return_value=[0.0] * 6,
        ), patch(
            "robotarm.modeling.online_policy.move_gripper_and_wait",
            side_effect=TimeoutError("gripper did not open"),
        ), patch("robotarm.modeling.online_policy.home_arm_to_zero") as home:
            with self.assertRaisesRegex(TimeoutError, "did not open"):
                reset_act_to_home(
                    robot, {}, object(), args, 2, lambda: False,
                )
        home.assert_not_called()

    def test_home_enables_before_sending_zero_target(self):
        events = []

        class Robot:
            OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(J=1))

            def move_j(self, joints):
                events.append(("move_j", tuple(joints)))

            def set_motion_mode(self, mode):
                events.append(("mode", mode))

            def set_speed_percent(self, speed):
                events.append(("speed", speed))

            def get_arm_status(self):
                return SimpleNamespace(msg=SimpleNamespace(
                    arm_status=0, motion_status=0,
                    err_status=SimpleNamespace(),
                ))

        class Gripper:
            def move_gripper_m(self, width, force):
                events.append(("gripper", width))

        config = {"joint_limits": {
            f"joint{index}": (-math.pi, math.pi)
            for index in range(1, 7)
        }}
        feedback = [[math.radians(5)] * 6] * 2 + [[0.0] * 6]
        clock = count(step=0.2)
        with patch("robotarm.modeling.online_policy.read_valid_joints", side_effect=feedback), patch(
            "robotarm.modeling.online_policy.base.enable_arm_and_gripper",
            side_effect=lambda *args: events.append(("enable",)),
        ), patch(
            "robotarm.modeling.online_policy.base.wait_gripper_width",
            return_value=0.001,
        ), patch("robotarm.modeling.online_policy.time.sleep"), patch(
            "robotarm.modeling.online_policy.time.monotonic",
            side_effect=lambda: next(clock),
        ):
            result = home_arm_to_zero(
                Robot(), config, Gripper(), 0.02, 1.0, 1.0, 5, 5.0,
            )
        self.assertEqual(result, [0.0] * 6)
        zero_target_index = events.index(("move_j", (0.0,) * 6))
        self.assertLess(events.index(("enable",)), zero_target_index)
        self.assertEqual(events[zero_target_index - 1], ("mode", 1))
        self.assertIn(("gripper", 0.02), events)
        self.assertLess(zero_target_index, events.index(("gripper", 0.0)))

    def test_home_starts_and_finishes_companion_arm_in_same_cycle(self):
        class Robot:
            OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(J=1))

            def move_j(self, joints):
                pass

            def set_motion_mode(self, mode):
                pass

            def set_speed_percent(self, speed):
                pass

            def get_arm_status(self):
                return SimpleNamespace(msg=SimpleNamespace(
                    arm_status=0, motion_status=0,
                    err_status=SimpleNamespace(),
                ))

        gripper = Mock()
        companion = Mock()
        companion.homing_active = False
        companion.homing_ready.return_value = True
        def progress(*args):
            self.assertTrue(companion.homing_active)
        config = {"joint_limits": {
            f"joint{index}": (-math.pi, math.pi)
            for index in range(1, 7)
        }}
        clock = count(step=0.2)
        with patch(
            "robotarm.modeling.online_policy.read_valid_joints",
            return_value=[0.0] * 6,
        ), patch(
            "robotarm.modeling.online_policy.base.enable_arm_and_gripper",
        ), patch(
            "robotarm.modeling.online_policy.base.wait_gripper_width",
            return_value=0.001,
        ), patch("robotarm.modeling.online_policy.time.sleep"), patch(
            "robotarm.modeling.online_policy.time.monotonic",
            side_effect=lambda: next(clock),
        ):
            home_arm_to_zero(
                Robot(), config, gripper, 0.02, 1.0, 1.0, 5, 5.0,
                companion=companion,
                progress_callback=progress,
            )

        companion.start_homing.assert_called_once_with(
            [0.0] * 6 + [0.02],
            5,
            timeout=5.0,
            stop_requested=ANY,
        )
        self.assertTrue(callable(companion.start_homing.call_args.kwargs["stop_requested"]))
        companion.homing_ready.assert_called()
        companion.finish_homing.assert_called_once_with(
            [0.0] * 6 + [0.001]
        )

    def test_paired_home_sends_follower_target_while_leader_home_runs(self):
        leader_started = Event()
        follower_target_sent = Event()
        follower_gripper_closed = Event()

        class Robot:
            OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(J=1))

            def move_j(self, joints):
                if list(joints) == [0.0] * 6:
                    self_test.assertTrue(leader_started.wait(1.0))
                    follower_target_sent.set()

            def set_motion_mode(self, mode):
                pass

            def set_speed_percent(self, speed):
                pass

            def get_arm_status(self):
                return SimpleNamespace(msg=SimpleNamespace(
                    arm_status=0, motion_status=0, err_status=SimpleNamespace(),
                ))

        def leader_home(*args, **kwargs):
            leader_started.set()
            self.assertTrue(follower_target_sent.wait(1.0))
            self.assertTrue(follower_gripper_closed.wait(1.5))

        def close_gripper(*args, **kwargs):
            follower_gripper_closed.set()
            return 0.001

        self_test = self
        companion = Mock()
        companion.start_homing.side_effect = leader_home
        companion.homing_ready.return_value = True
        config = {"joint_limits": {
            f"joint{index}": (-math.pi, math.pi) for index in range(1, 7)
        }}
        with patch(
            "robotarm.modeling.online_policy.read_valid_joints",
            side_effect=lambda *args: [0.0] * 6 if follower_target_sent.is_set() else [0.1] * 6,
        ), patch(
            "robotarm.modeling.online_policy.base.enable_arm_and_gripper",
        ), patch(
            "robotarm.modeling.online_policy.move_gripper_and_wait",
            side_effect=close_gripper,
        ), patch("robotarm.modeling.online_policy.time.sleep"):
            home_arm_to_zero(
                Robot(), config, Mock(), 0.02, 1.0, 1.0, 5, 5.0,
                companion=companion,
            )
        self.assertTrue(follower_target_sent.is_set())
        self.assertTrue(follower_gripper_closed.is_set())
        companion.finish_homing.assert_called_once()

    def test_leader_home_failure_stops_only_leader_when_follower_can_hold(self):
        robot = Mock()
        robot.OPTIONS.MOTION_MODE.J = "j"
        companion = Mock()
        measured = [0.1] * 6
        with patch(
            "robotarm.modeling.online_policy.read_valid_joints",
            return_value=measured,
        ):
            follower_error, leader_error = isolate_leader_home_failure(
                robot, {}, 1.0, companion
            )
        self.assertIsNone(follower_error)
        self.assertIsNone(leader_error)
        robot.set_motion_mode.assert_called_once_with("j")
        robot.move_j.assert_called_once_with(measured)
        robot.electronic_emergency_stop.assert_not_called()
        companion.emergency_stop.assert_called_once_with()

    def test_home_emergency_input_prevents_enable_and_motion(self):
        with patch("robotarm.modeling.online_policy.read_valid_joints") as read, patch(
            "robotarm.modeling.online_policy.base.enable_arm_and_gripper"
        ) as enable:
            with self.assertRaisesRegex(InterruptedError, "emergency stop"):
                home_arm_to_zero(
                    None, None, None, 0.0, 1.0, 1.0, 5, 5.0,
                    lambda: True,
                )
        read.assert_not_called()
        enable.assert_not_called()

    def test_act_stream_speed_is_restored_after_enable(self):
        events = []

        class Robot:
            OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(J=1, JS=2))

            def move_j(self, joints):
                events.append("move_j")

            def move_js(self, joints):
                events.append("move_js")

            def set_motion_mode(self, mode):
                events.append(("mode", mode))

            def set_speed_percent(self, speed):
                events.append(("speed", speed))

        with patch(
            "robotarm.modeling.online_policy.read_valid_joints",
            return_value=[0.0] * 6,
        ), patch(
            "robotarm.modeling.online_policy.base.enable_arm_and_gripper",
            side_effect=lambda *args: events.append("enable"),
        ), patch("robotarm.modeling.online_policy.wait_mode_feedback"), patch(
            "robotarm.modeling.online_policy.time.sleep"
        ):
            prepare_js_control(
                Robot(), {}, None, 0.02, 1.0, 1.0, 1, 100,
            )
        self.assertLess(events.index("enable"), events.index(("speed", 100)))
        self.assertLess(events.index(("speed", 100)), events.index("move_js"))

    def test_clamp_detection_has_numeric_tolerance(self):
        self.assertFalse(action_was_clamped([0.0] * 7, [1e-7] * 7))
        self.assertTrue(action_was_clamped([0.0] * 7, [1e-4] + [0.0] * 6))

    def test_joint_acceleration_limiter_ramps_velocity(self):
        import numpy as np

        desired = np.asarray([math.radians(20)] + [0.0] * 6)
        previous = np.zeros(7)
        velocity = np.zeros(6)
        positions = []
        for _ in range(3):
            shaped, velocity = limit_joint_target_acceleration(
                desired, previous, velocity, math.radians(200), 0.05,
                math.radians(50),
            )
            positions.append(math.degrees(shaped[0]))
            previous = shaped
        self.assertEqual([round(value, 6) for value in positions], [0.5, 1.5, 3.0])

    def test_joint_acceleration_limiter_decelerates_on_reversal(self):
        import numpy as np

        desired = np.asarray([math.radians(-20)] + [0.0] * 6)
        previous = np.asarray([math.radians(3)] + [0.0] * 6)
        velocity = np.asarray([math.radians(30)] + [0.0] * 5)
        shaped, next_velocity = limit_joint_target_acceleration(
            desired, previous, velocity, math.radians(200), 0.05,
            math.radians(50),
        )
        self.assertAlmostEqual(math.degrees(next_velocity[0]), 20.0)
        self.assertAlmostEqual(math.degrees(shaped[0]), 4.0)

    def test_act_backpressure_progress_resets_no_progress_counter(self):
        target = [math.radians(6)] + [0.0] * 6
        previous_state = [0.0] * 7
        previous_error, _, previous_gripper_error = act_subgoal_errors(
            target, previous_state
        )
        status = update_act_backpressure_progress(
            target,
            [math.radians(1)] + [0.0] * 6,
            previous_error,
            previous_gripper_error,
            4,
            math.radians(0.05),
            0.0001,
        )
        self.assertTrue(status["progressed"])
        self.assertEqual(status["no_progress_steps"], 0)
        self.assertAlmostEqual(math.degrees(status["joint_progress"]), 1.0)

    def test_act_subgoal_latch_does_not_chase_a_newer_farther_target(self):
        first_target = [math.radians(6)] + [0.0] * 6
        latched, created = latch_act_subgoal(None, first_target, True)
        self.assertTrue(created)
        newer_target = [math.radians(9)] + [0.0] * 6
        retained, created = latch_act_subgoal(latched, newer_target, True)
        self.assertFalse(created)
        self.assertEqual(retained.tolist(), first_target)

    def test_act_subgoal_includes_fixed_braking_distance(self):
        subgoal = bounded_act_subgoal(
            [math.radians(30)] + [0.0] * 5 + [0.05],
            [0.0] * 7,
            self.bounds,
            [math.radians(50)] + [0.0] * 5,
            math.radians(6),
            0.016,
            0.1,
            math.radians(200),
        )
        # 6 degree feedback lead plus v^2/(2a) = 6.25 degree braking room.
        self.assertAlmostEqual(math.degrees(subgoal[0]), 12.25)
        self.assertAlmostEqual(subgoal[6], 0.016)

    def test_act_backpressure_stall_counts_no_feedback_progress(self):
        target = [math.radians(6)] + [0.0] * 6
        joint_error, _, gripper_error = act_subgoal_errors(
            target, [0.0] * 7
        )
        status = update_act_backpressure_progress(
            target,
            [0.0] * 7,
            joint_error,
            gripper_error,
            2,
            math.radians(0.05),
            0.0001,
        )
        self.assertFalse(status["progressed"])
        self.assertEqual(status["no_progress_steps"], 3)

    def test_grasped_gripper_does_not_freeze_following_act_lift(self):
        import numpy as np

        # Reported stop: measured 78.3 mm, requested 62.1 mm, safe 62.3 mm.
        # Keep width fixed as though gripping an object while joints follow
        # successive policy lift targets for longer than the old 10-tick trip.
        state = np.asarray([0.0] * 6 + [0.0783])
        previous = state.copy()
        latched = [0.0] * 6 + [0.0621]  # Also release a pre-existing grip latch.
        for step in range(30):
            desired = np.asarray([0.001 * (step + 1)] + [0.0] * 5 + [0.0621])
            safe = self.limited_target(desired, state, previous)
            self.assertTrue(action_was_clamped(desired, safe))
            pressure = act_backpressure_required(desired, safe, state)
            self.assertFalse(pressure)
            latched, created, _ = update_act_backpressure_target(
                latched, desired, pressure, True,
            )
            self.assertIsNone(latched)
            self.assertFalse(created)
            self.assertLessEqual(abs(safe[6] - previous[6]), 0.008 + 1e-9)
            self.assertLessEqual(abs(safe[6] - state[6]), 0.016 + 1e-9)
            state[:6] = safe[:6]
            previous = safe
        self.assertAlmostEqual(state[0], 0.03)
        self.assertAlmostEqual(state[6], 0.0783)

    def test_grasp_exception_preserves_joint_and_opening_stall_protection(self):
        import numpy as np

        state = np.asarray([0.0] * 6 + [0.0783])
        for desired in (
            [math.radians(20)] + [0.0] * 5 + [0.0621],  # Joint blocked, closing.
            [0.0] * 6 + [0.1],  # Gripper blocked while opening.
        ):
            safe = self.limited_target(desired, state, state)
            self.assertTrue(act_backpressure_required(desired, safe, state))
            latch, created, _ = update_act_backpressure_target(
                None, desired, True, True,
            )
            self.assertTrue(created)
            joint_error, _, grip_error = act_subgoal_errors(latch, state)
            count = 0
            for _ in range(10):
                status = update_act_backpressure_progress(
                    latch, state, joint_error, grip_error, count,
                    math.radians(0.05), 0.0001,
                )
                count = status["no_progress_steps"]
            self.assertEqual(count, 10)

    def test_act_backpressure_rejects_invalid_state(self):
        for current in ([0.0] * 6, [math.nan] * 7):
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                act_backpressure_required([0.0] * 7, [0.0] * 7, current)

    def test_act_backpressure_small_changes_accumulate_against_baseline(self):
        target = [math.radians(6)] + [0.0] * 6
        baseline, _, gripper_error = act_subgoal_errors(target, [0.0] * 7)
        first = update_act_backpressure_progress(
            target, [math.radians(0.04)] + [0.0] * 6,
            baseline, gripper_error, 0,
            math.radians(0.05), 0.0001,
        )
        self.assertFalse(first["progressed"])
        second = update_act_backpressure_progress(
            target, [math.radians(0.08)] + [0.0] * 6,
            baseline, gripper_error, first["no_progress_steps"],
            math.radians(0.05), 0.0001,
        )
        self.assertTrue(second["progressed"])
        self.assertEqual(second["no_progress_steps"], 0)

    def test_act_backpressure_releases_on_unclamped_piper_residual(self):
        # The real run retained a 2.13 degree J5 residual, but the final safe
        # target already equalled the control target.  Backpressure must release
        # on that limiter state instead of waiting for exact JS convergence.
        target_deg = [13.613, 98.548, -69.429, -1.034, 19.128, 0.086]
        target = [math.radians(value) for value in target_deg] + [0.0384]
        next_target, created, released = update_act_backpressure_target(
            target,
            [math.radians(30)] + [0.0] * 6,
            tracking_clamped=False,
            raw_action_ok=True,
        )
        self.assertIsNone(next_target)
        self.assertFalse(created)
        self.assertTrue(released)

    def test_act_backpressure_retains_target_only_while_hard_clamped(self):
        target = [math.radians(6)] + [0.0] * 6
        retained, created, released = update_act_backpressure_target(
            target, [math.radians(12)] + [0.0] * 6, True, True,
        )
        self.assertEqual(retained.tolist(), target)
        self.assertFalse(created)
        self.assertFalse(released)

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

    def test_act_envelope_allows_small_raw_joint_bound_overshoot(self):
        raw = [math.pi + math.radians(0.7), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tolerated = act_action_envelope_failures(
            raw, self.state, self.bounds, 0.1,
            math.inf, math.inf, 0.0, math.radians(1.0),
        )
        rejected = act_action_envelope_failures(
            raw, self.state, self.bounds, 0.1,
            math.inf, math.inf, 0.0, math.radians(0.5),
        )
        self.assertEqual(tolerated, [])
        self.assertTrue(any("raw joint bound" in reason for reason in rejected))

    def test_limits_are_revalidated_after_policy_override(self):
        args = Namespace(
            max_joint_step_deg=3.0,
            max_joint_acceleration_deg_s2=200.0,
            max_joint_lead_deg=1.0,
            max_gripper_step_mm=1.0,
            max_gripper_lead_mm=2.0,
            max_act_clamp_hold_steps=10,
            warn_act_raw_joint_gap_deg=8.0,
            max_act_raw_joint_gap_deg=9.0,
            max_act_raw_gap_consecutive_steps=3,
            max_act_raw_gripper_gap_mm=18.0,
            max_act_raw_joint_bound_tolerance_deg=1.0,
            max_act_raw_gripper_bound_tolerance_mm=1.0,
            act_subgoal_min_progress_deg=0.05,
            act_subgoal_min_gripper_progress_mm=0.1,
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

    def test_pi0_execution_never_requires_shadow_approval(self):
        with patch(
            "robotarm.modeling.online_policy.defaults",
            return_value={"approval-required": True},
        ), patch(
            "robotarm.modeling.online_policy.validate_approval"
        ) as validate:
            checked = validate_execution_approval(
                Path("/missing/approval.json"), "0" * 64, "pi0", {}
            )
        self.assertFalse(checked)
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
