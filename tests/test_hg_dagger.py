from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robotarm.modeling.hg_dagger import (
    ClutchedLeaderMapping,
    FollowingLeaderMapping,
    LeaderExpert,
    LeaderHomingFailure,
    LeaderHomingNoMotion,
    TakeoverMotionGate,
)


class HGDAggerTests(unittest.TestCase):
    def test_gripper_clutch_recovers_full_closure_after_repeated_takeover(self):
        mapping = ClutchedLeaderMapping([1] * 6, [0.0] * 6, 0.1)
        bounds = [(-math.pi, math.pi)] * 6
        # Observed second takeover: leader closed, follower holding at 79 mm.
        for _ in range(3):
            mapping.engage([0.0] * 7, [0.0] * 6 + [0.079])
            self.assertAlmostEqual(mapping.target([0.0] * 7, bounds)[6], 0.079)
            for width in (0.02, 0.05, 0.102):
                requested = mapping.target([0.0] * 6 + [width], bounds)
                self.assertTrue(0.0 <= requested[6] <= 0.1)
            self.assertAlmostEqual(requested[6], 0.1)
            for width in (0.08, 0.02, -0.002):
                requested = mapping.target([0.0] * 6 + [width], bounds)
            self.assertAlmostEqual(requested[6], 0.0)
            for _ in range(5):
                self.assertAlmostEqual(mapping.target([0.0] * 7, bounds)[6], 0.0)

    def test_gripper_clutch_recovers_full_opening_after_lower_saturation(self):
        mapping = ClutchedLeaderMapping([1] * 6, [0.0] * 6, 0.1)
        bounds = [(-math.pi, math.pi)] * 6
        mapping.engage([0.0] * 6 + [0.1], [0.0] * 6 + [0.02])
        self.assertAlmostEqual(mapping.target([0.0] * 6 + [0.1], bounds)[6], 0.02)
        self.assertAlmostEqual(mapping.target([0.0] * 7, bounds)[6], 0.0)
        self.assertAlmostEqual(mapping.target([0.0] * 6 + [0.1], bounds)[6], 0.1)

    def test_policy_leader_gripper_recovers_after_saturating(self):
        mapping = FollowingLeaderMapping([1] * 6, [0.0] * 6, 0.1)
        bounds = [(-math.pi, math.pi)] * 6
        mapping.engage([0.0] * 7, [0.0] * 6 + [0.079])
        self.assertAlmostEqual(mapping.target([0.0] * 6 + [0.079], bounds)[6], 0.0)
        self.assertAlmostEqual(mapping.target([0.0] * 7, bounds)[6], 0.0)
        self.assertAlmostEqual(mapping.target([0.0] * 6 + [0.1], bounds)[6], 0.1)

    def test_home_preflight_waits_out_transient_bad_feedback(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.config = {}
        expert.feedback_timeout = 0.2
        clock = [0.0]
        with patch("robotarm.modeling.hg_dagger.read_joint_message", side_effect=[
            RuntimeError("J6 feedback -171.45 outside bounds"),
            ([0.0] * 6, 0.0), ([0.0] * 6, 0.0),
        ]), patch("robotarm.modeling.hg_dagger.time.monotonic", side_effect=lambda: clock[0]), patch(
            "robotarm.modeling.hg_dagger.time.sleep", side_effect=lambda delta: clock.__setitem__(0, clock[0] + delta)
        ):
            self.assertEqual(expert._wait_home_joints(0.2), [0.0] * 6)
        expert.robot.move_js.assert_not_called()
        expert.robot.move_leader_to_home.assert_not_called()

    def test_home_preflight_refuses_persistent_out_of_range_feedback(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.config = {}
        expert.feedback_timeout = 0.2
        clock = [0.0]
        with patch("robotarm.modeling.hg_dagger.read_joint_message", side_effect=RuntimeError("J6 -171.45")), patch(
            "robotarm.modeling.hg_dagger.time.monotonic", side_effect=lambda: clock[0]
        ), patch("robotarm.modeling.hg_dagger.time.sleep", side_effect=lambda delta: clock.__setitem__(0, clock[0] + delta)):
            with self.assertRaisesRegex(LeaderHomingFailure, "no leader homing/model target sent.*J6 -171.45"):
                expert._wait_home_joints(0.1)
        expert.robot.move_js.assert_not_called()
        expert.robot.move_leader_to_home.assert_not_called()

    def test_motion_gate_ignores_noise_and_requires_consecutive_motion(self):
        gate = TakeoverMotionGate(math.radians(0.75), 0.0015, 2)
        reference = [0.0] * 6 + [0.02]
        gate.arm(reference)

        self.assertFalse(gate.update([math.radians(0.5)] + [0.0] * 5 + [0.02]))
        self.assertEqual(gate.streak, 0)
        moved = [math.radians(1.0)] + [0.0] * 5 + [0.02]
        self.assertFalse(gate.update(moved))
        self.assertTrue(gate.update(moved))
        self.assertTrue(gate.triggered)

    def test_motion_gate_uses_relative_gripper_motion(self):
        gate = TakeoverMotionGate(math.radians(1.0), 0.0015, 1)
        gate.arm([math.radians(40.0)] * 6 + [0.04])
        self.assertTrue(
            gate.update([math.radians(40.0)] * 6 + [0.042])
        )

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

    def test_following_leader_starts_without_jump_and_tracks_follower_delta(self):
        mapping = FollowingLeaderMapping([1] * 6, [0.0] * 6, 0.1)
        leader = [math.radians(25.0)] * 6 + [0.04]
        follower = [math.radians(-5.0)] * 6 + [0.02]
        bounds = [(-math.pi, math.pi)] * 6

        mapping.engage(leader, follower)
        engaged = mapping.target(follower, bounds)
        for actual, expected in zip(engaged, leader):
            self.assertAlmostEqual(actual, expected)

        moved = [value + math.radians(2.0) for value in follower[:6]] + [0.025]
        target = mapping.target(moved, bounds)
        for actual, start in zip(target[:6], leader[:6]):
            self.assertAlmostEqual(actual - start, math.radians(2.0))
        self.assertAlmostEqual(target[6], 0.045)

    def test_following_mapping_inverts_signs_and_offsets(self):
        mapping = FollowingLeaderMapping(
            [-1, 1, -1, 1, -1, 1], [10.0] * 6, 0.1
        )
        leader = [math.radians(3.0)] * 6 + [0.01]
        follower = [math.radians(20.0)] * 6 + [0.02]
        bounds = [(-math.pi, math.pi)] * 6
        mapping.engage(leader, follower)
        first = mapping.target(follower, bounds)
        for actual, expected in zip(first, leader):
            self.assertAlmostEqual(actual, expected)

    def test_absolute_following_target_matches_follower_without_clutch_offset(self):
        mapping = FollowingLeaderMapping(
            [-1, 1, -1, 1, -1, 1], [10.0] * 6, 0.07
        )
        follower = [math.radians(20.0)] * 6 + [0.04]
        target = mapping.absolute_target(
            follower, [(-math.pi, math.pi)] * 6
        )
        expected = [
            math.radians(-10.0),
            math.radians(10.0),
            math.radians(-10.0),
            math.radians(10.0),
            math.radians(-10.0),
            math.radians(10.0),
        ]
        for actual, value in zip(target[:6], expected):
            self.assertAlmostEqual(actual, value)
        self.assertEqual(target[6], 0.04)

    def test_wait_controlled_retries_stale_cached_feedback(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.feedback_timeout = 0.2
        fresh = [0.1] * 6 + [0.02]
        with patch.object(
            expert,
            "_read_controlled",
            side_effect=[
                RuntimeError("controlled leader feedback is stale (21000 ms)"),
                (fresh, 0.01),
            ],
        ) as read, patch("robotarm.modeling.hg_dagger.time.sleep"):
            state, age = expert._wait_controlled(timeout=1.0)
        self.assertEqual(state, fresh)
        self.assertEqual(age, 0.01)
        self.assertEqual(read.call_count, 2)

    def test_initial_alignment_commands_absolute_follower_pose(self):
        robot = Mock()
        robot.OPTIONS.MOTION_MODE.J = "j"
        robot.OPTIONS.MOTION_MODE.JS = "js"
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.feedback_timeout = 0.2
        expert.gripper_force = 1.0
        expert.robot = robot
        expert.gripper = Mock()
        expert.config = {}
        expert.joints = [0.2] * 6
        expert.gripper_width = 0.04
        expert.follow_mapping = FollowingLeaderMapping(
            [1] * 6, [0.0] * 6, 0.07
        )
        follower = [0.0] * 6 + [0.01]
        initial = [0.2] * 6 + [0.04]
        aligned = list(follower)

        with patch.object(expert, "hold_controlled") as hold, patch.object(
            expert,
            "_wait_controlled",
            side_effect=[(initial, 0.0), (aligned, 0.0), (aligned, 0.0)],
        ), patch.object(expert, "follow") as follow, patch(
            "robotarm.modeling.hg_dagger.command_joint_bounds",
            return_value=[(-math.pi, math.pi)] * 6,
        ), patch(
            "robotarm.modeling.hg_dagger.time.monotonic",
            side_effect=[0.0, 0.1, 0.1, 0.5, 0.5],
        ), patch("robotarm.modeling.hg_dagger.time.sleep"):
            expert.align_to_follower(
                follower, speed_percent=5.0, timeout=30.0
            )

        hold.assert_called_once_with()
        robot.set_speed_percent.assert_called_once_with(5.0)
        self.assertIn(unittest.mock.call(follower[:6]), robot.move_j.call_args_list)
        expert.gripper.move_gripper_m.assert_called_once_with(0.01, 1.0)
        robot.set_motion_mode.assert_any_call("js")
        follow.assert_called_once_with(follower)

    def test_paired_leader_home_uses_verified_firmware_command(self):
        robot = Mock()
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.feedback_timeout = 0.2
        expert.channel = "can_leader1"
        expert.gripper_force = 1.0
        expert.robot = robot
        expert.gripper = Mock()
        expert.config = {}
        expert.joints = [0.2] * 6
        expert.gripper_width = 0.03
        expert.follow_mapping = FollowingLeaderMapping(
            [1] * 6, [0.0] * 6, 0.07
        )
        expert.homing_active = False
        expert.drag_active = False
        expert._homing_target = None
        expert.homing_error_degrees = None
        expert._homing_started_at = None
        expert._homing_initial_error = None
        follower_target = [0.0] * 6 + [0.01]

        stop_requested = Mock(return_value=False)
        with patch(
            "robotarm.modeling.hg_dagger.read_joint_message",
            return_value=([0.2] * 6, 0.0),
        ), patch.object(expert, "hold_controlled") as hold_before_home, patch.object(
            expert,
            "_wait_controlled",
            return_value=([0.0] * 6 + [0.01], 0.0),
        ), patch(
            "robotarm.modeling.hg_dagger.command_joint_bounds",
            return_value=[(-math.pi, math.pi)] * 6,
        ), patch(
            "robotarm.modeling.hg_dagger.home_leader_and_restore",
            return_value=[0.0] * 6,
        ) as native_home, patch(
            "robotarm.modeling.hg_dagger.exit_leader_drag_mode",
        ) as restore_control, patch(
            "robotarm.modeling.hg_dagger.time.monotonic",
            return_value=1.0,
        ):
            native_home.side_effect = lambda *args, **kwargs: (
                self.assertTrue(expert.drag_active),
                robot.set_leader_mode.assert_called_once_with(),
                [0.0] * 6,
            )[-1]
            expert.start_homing(
                follower_target,
                5.0,
                timeout=45.0,
                stop_requested=stop_requested,
            )

        hold_before_home.assert_not_called()
        native_home.assert_called_once_with(
            robot,
            {},
            "can_leader1",
            [0.2] * 6,
            45.0,
            stop_requested,
            move_j_speed_percent=5.0,
        )
        restore_control.assert_called_once_with(
            robot,
            [0.0] * 6,
            expert.gripper,
            0.01,
            1.0,
            0.2,
            {},
        )

    def test_paired_leader_home_reports_typed_no_motion_after_eight_seconds(self):
        robot = Mock()
        robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(
            ctrl_mode="CAN_CTRL", arm_status="NORMAL", mode_feedback="MOVE_J"
        ))
        robot.get_joints_enable_status_list.return_value = [True] * 6
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = robot
        expert.homing_active = True
        expert._homing_target = [0.0] * 7
        expert._homing_started_at = 0.0
        expert._homing_initial_error = 29.0
        expert.homing_error_degrees = 29.0
        expert.joints = [math.radians(29.0)] + [0.0] * 5
        expert.gripper_width = 0.02
        with patch.object(
            expert,
            "_read_controlled",
            return_value=([math.radians(29.0)] + [0.0] * 5 + [0.02], 0.0),
        ), patch("robotarm.modeling.hg_dagger.time.monotonic", return_value=8.1):
            with self.assertRaisesRegex(LeaderHomingNoMotion, "within 8s"):
                expert.homing_ready()

    def test_policy_action_is_refused_while_homing_owns_leader(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.homing_active = True
        with patch.object(expert, "_read_controlled") as read:
            with self.assertRaisesRegex(RuntimeError, "homing owns control"):
                expert.command_policy_action([0.0] * 7)
        read.assert_not_called()
        expert.robot.move_js.assert_not_called()

    def test_policy_action_is_commanded_directly_to_second_arm(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.feedback_timeout = 0.2
        expert.gripper_force = 1.0
        expert.robot = Mock()
        expert.gripper = Mock()
        expert.config = {}
        expert.drag_active = False
        expert.follow_mapping = FollowingLeaderMapping(
            [1] * 6, [0.0] * 6, 0.07
        )
        action = [0.1] * 6 + [0.03]
        with patch.object(
            expert, "_read_controlled", return_value=([0.0] * 7, 0.01)
        ), patch(
            "robotarm.modeling.hg_dagger.command_joint_bounds",
            return_value=[(-math.pi, math.pi)] * 6,
        ):
            sample = expert.command_policy_action(action)

        expert.robot.move_js.assert_called_once_with(action[:6])
        expert.gripper.move_gripper_m.assert_called_once_with(0.03, 1.0)
        self.assertEqual(sample.requested_action, action)

    def test_policy_resume_preserves_arm_and_gripper_offsets(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.gripper = Mock()
        expert.config = {}
        expert.gripper_force = 1.0
        expert.feedback_timeout = 0.2
        expert.robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(
            ctrl_mode=1, mode_feedback=1, arm_status=0,
        ))
        expert.robot.get_joints_enable_status_list.return_value = [True] * 6
        expert.drag_active = False
        expert.follow_mapping = FollowingLeaderMapping(
            [-1, 1, -1, 1, -1, 1], [10.0] * 6, 0.07
        )
        leader = [0.4] * 6 + [0.05]
        follower = [-0.2] * 6 + [0.02]
        with patch.object(expert, "hold_controlled"), patch.object(
            expert, "_wait_controlled", return_value=(leader, 0.0)
        ), patch.object(
            expert, "_read_controlled", return_value=(leader, 0.0)
        ), patch(
            "robotarm.modeling.hg_dagger.command_joint_bounds",
            return_value=[(-math.pi, math.pi)] * 6,
        ):
            expert.resume_policy_control(follower)
            expert.robot.set_speed_percent.assert_called_once_with(100)
            expert.robot.set_motion_mode.assert_called_once_with(expert.robot.OPTIONS.MOTION_MODE.JS)
            first = expert.robot.move_js.call_args.args[0]
            for actual, expected in zip(first, leader[:6]):
                self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(expert.gripper.move_gripper_m.call_args.args[0], 0.05)
            action = [v + 0.01 for v in follower[:6]] + [0.021]
            expert.command_policy_action(action)
            following = expert.robot.move_js.call_args.args[0]
            for actual, sign in zip(following, [-1, 1, -1, 1, -1, 1]):
                self.assertAlmostEqual(actual, 0.4 + sign * 0.01)
            self.assertAlmostEqual(expert.gripper.move_gripper_m.call_args.args[0], 0.051)

    def test_policy_control_waits_for_role_and_enable_acknowledgement(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.feedback_timeout = 0.2
        expert.robot.get_arm_status.side_effect = [
            SimpleNamespace(msg=SimpleNamespace(ctrl_mode=2, mode_feedback=1, arm_status=0)),
            SimpleNamespace(msg=SimpleNamespace(ctrl_mode=1, mode_feedback=1, arm_status=0)),
        ]
        expert.robot.get_joints_enable_status_list.side_effect = [[False] * 6, [True] * 6]
        with patch("robotarm.modeling.hg_dagger.time.sleep"):
            expert._wait_policy_control()
        self.assertEqual(expert.robot.get_arm_status.call_count, 2)

    def test_policy_control_reports_unaccepted_role_instead_of_silent_resume(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.feedback_timeout = 0.2
        expert.robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(
            ctrl_mode=2, mode_feedback=1, arm_status=0,
        ))
        expert.robot.get_joints_enable_status_list.return_value = [True] * 6
        with patch("robotarm.modeling.hg_dagger.time.monotonic", side_effect=[0.0, 0.1, 3.0]), patch(
            "robotarm.modeling.hg_dagger.time.sleep"
        ):
            with self.assertRaisesRegex(RuntimeError, "ctrl_mode=2"):
                expert._wait_policy_control()

    def test_valid_feedback_does_not_enter_zero_force_during_initialization(self):
        robot = Mock()
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0)
        )
        robot.OPTIONS.EFFECTOR.AGX_GRIPPER = "gripper"
        gripper = Mock()
        robot.init_effector.return_value = gripper

        with patch(
            "robotarm.modeling.hg_dagger.base.connect_robot",
            return_value=(robot, {}),
        ), patch(
            "robotarm.modeling.hg_dagger.wait_initial_leader_joints",
            return_value=([0.0] * 6, "feedback"),
        ), patch(
            "robotarm.modeling.hg_dagger.read_gripper_message",
            return_value=(0.04, 0.0),
        ), patch(
            "robotarm.modeling.hg_dagger.ensure_receive_running",
        ):
            expert = LeaderExpert(
                interface="socketcan",
                channel="can_leader1",
                firmware="v1",
                feedback_timeout=0.2,
                init_timeout=30.0,
                signs=[1] * 6,
                offsets_deg=[0.0] * 6,
                gripper_max=0.07,
                gripper_force=1.0,
            )

        robot.set_leader_mode.assert_not_called()
        robot.set_joint_limits_enabled.assert_called_once_with(False)
        robot.set_auto_set_motion_mode_enabled.assert_called_once_with(False)
        self.assertFalse(expert.drag_active)
        self.assertTrue(expert.controlled_ready)
        # Avoid hardware cleanup in this unit test.
        expert.robot = None

    def test_initialization_waits_for_delayed_leader_status_frame(self):
        robot = Mock()
        robot.OPTIONS.EFFECTOR.AGX_GRIPPER = "gripper"
        robot.get_arm_status.side_effect = [
            None,
            SimpleNamespace(msg=SimpleNamespace(arm_status=0)),
        ]
        gripper = Mock()
        robot.init_effector.return_value = gripper
        with patch(
            "robotarm.modeling.hg_dagger.base.connect_robot",
            return_value=(robot, {}),
        ), patch(
            "robotarm.modeling.hg_dagger.wait_initial_leader_joints",
            return_value=([0.0] * 6, "feedback"),
        ), patch(
            "robotarm.modeling.hg_dagger.read_gripper_message",
            return_value=(0.02, 0.0),
        ), patch(
            "robotarm.modeling.hg_dagger.ensure_receive_running",
        ), patch("robotarm.modeling.hg_dagger.time.sleep"):
            expert = LeaderExpert(
                interface="socketcan",
                channel="can_leader1",
                firmware="v189",
                feedback_timeout=0.2,
                init_timeout=30.0,
                signs=[1] * 6,
                offsets_deg=[0.0] * 6,
                gripper_max=0.07,
                gripper_force=1.0,
            )
        self.assertEqual(robot.get_arm_status.call_count, 2)
        expert.robot = None

    def test_resume_follows_then_engage_switches_to_zero_force(self):
        robot = Mock()
        gripper = Mock()
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.feedback_timeout = 0.2
        expert.gripper_force = 1.0
        expert.robot = robot
        expert.gripper = gripper
        expert.config = {}
        expert.drag_active = False
        expert.controlled_ready = False
        expert.joints = [0.0] * 6
        expert.gripper_width = 0.02
        expert.gripper_not_before = 0.0
        expert.mapping = ClutchedLeaderMapping([1] * 6, [0.0] * 6, 0.07)
        expert.follow_mapping = FollowingLeaderMapping(
            [1] * 6, [0.0] * 6, 0.07
        )
        follower = [0.1] * 6 + [0.03]
        leader = [0.2] * 6 + [0.04]

        with patch(
            "robotarm.modeling.hg_dagger.exit_leader_drag_mode"
        ) as exit_drag, patch.object(
            expert, "_read_controlled", return_value=(leader, 0.0)
        ), patch(
            "robotarm.modeling.hg_dagger.command_joint_bounds",
            return_value=[(-math.pi, math.pi)] * 6,
        ):
            expert.resume_following(follower)

        exit_drag.assert_called_once()
        robot.set_leader_mode.assert_not_called()
        robot.move_js.assert_called_once_with(leader[:6])
        gripper.move_gripper_m.assert_called_once_with(leader[6], 1.0)

        robot.reset_mock()
        gripper.reset_mock()
        # Mirror the state update performed by the real _read_controlled().
        expert.joints = leader[:6]
        expert.gripper_width = leader[6]
        with patch.object(
            expert, "_read_controlled", return_value=(leader, 0.0)
        ), patch.object(
            expert, "_read", return_value=(leader, 0.0)
        ), patch(
            "robotarm.modeling.hg_dagger.base.set_gripper_enabled"
        ) as set_gripper_enabled, patch("robotarm.modeling.hg_dagger.time.sleep"):
            robot.set_leader_mode.side_effect = lambda: set_gripper_enabled.assert_called_once_with(
                gripper, False, leader[6], 1.0, 2.0
            )
            sample = expert.engage(follower, [(-math.pi, math.pi)] * 6)

        robot.set_leader_mode.assert_called_once_with()
        self.assertTrue(expert.drag_active)
        self.assertEqual(sample.requested_action, follower)
        gripper.move_gripper_m.assert_not_called()

    def test_hold_controlled_reasserts_target_control_with_fresh_feedback(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.gripper = Mock()
        expert.config = {}
        expert.feedback_timeout = 0.2
        expert.gripper_force = 1.0
        expert.drag_active = False
        expert.controlled_ready = True
        expert.joints = [0.1] * 6
        expert.gripper_width = 0.02
        expert.follow_mapping = FollowingLeaderMapping(
            [1] * 6, [0.0] * 6, 0.07
        )
        with patch(
            "robotarm.modeling.hg_dagger.exit_leader_drag_mode"
        ) as enter_target_control:
            expert.hold_controlled()
        enter_target_control.assert_called_once_with(
            expert.robot,
            [0.1] * 6,
            expert.gripper,
            0.02,
            1.0,
            2.0,
            {},
        )
        self.assertTrue(expert.controlled_ready)
        self.assertFalse(expert.drag_active)

    def test_takeover_does_not_switch_role_if_gripper_release_fails(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        expert.robot = Mock()
        expert.gripper = Mock()
        expert.drag_active = False
        expert.feedback_timeout = 0.2
        expert.gripper_force = 1.0
        with patch.object(
            expert, "_read_controlled", return_value=([0.1] * 6 + [0.04], 0.0)
        ), patch(
            "robotarm.modeling.hg_dagger.base.set_gripper_enabled",
            side_effect=TimeoutError("Gripper disable timed out"),
        ):
            with self.assertRaisesRegex(TimeoutError, "disable timed out"):
                expert.engage([0.0] * 7, [(-math.pi, math.pi)] * 6)
        expert.robot.set_leader_mode.assert_not_called()
        expert.gripper.move_gripper_m.assert_not_called()

    def test_closing_controlled_leader_only_disconnects_and_keeps_enabled(self):
        expert = LeaderExpert.__new__(LeaderExpert)
        robot = Mock()
        expert.robot = robot
        expert.gripper = Mock()
        expert.drag_active = False
        expert.close()
        robot.disconnect.assert_called_once_with()
        robot.disable.assert_not_called()
        robot.electronic_emergency_stop.assert_not_called()
        expert.gripper.disable_gripper.assert_not_called()

    def test_missing_width_feedback_waits_for_new_gripper_input_after_drag_mode(self):
        events: list[str] = []
        robot = Mock()
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0)
        )
        robot.OPTIONS.EFFECTOR.AGX_GRIPPER = "gripper"
        gripper = Mock()
        robot.init_effector.return_value = gripper
        robot.set_leader_mode.side_effect = lambda: events.append("leader_mode")

        def wait_for_gripper(*args, **kwargs):
            events.append("gripper_input")
            self.assertEqual(kwargs["not_before"], 123.0)
            return 0.04, 0.0

        with patch(
            "robotarm.modeling.hg_dagger.base.connect_robot",
            return_value=(robot, {}),
        ), patch(
            "robotarm.modeling.hg_dagger.wait_initial_leader_joints",
            return_value=([0.0] * 6, "feedback"),
        ), patch(
            "robotarm.modeling.hg_dagger.read_gripper_message",
            side_effect=RuntimeError("no width feedback"),
        ), patch(
            "robotarm.modeling.hg_dagger.wait_leader_gripper_command",
            side_effect=wait_for_gripper,
        ) as wait_gripper, patch(
            "robotarm.modeling.hg_dagger.update_leader_joint_message",
            return_value=([0.0] * 6, 0.0),
        ), patch(
            "robotarm.modeling.hg_dagger.ensure_receive_running",
        ), patch(
            "robotarm.modeling.hg_dagger.time.sleep",
        ), patch(
            "robotarm.modeling.hg_dagger.time.time",
            return_value=123.0,
        ):
            expert = LeaderExpert(
                interface="socketcan",
                channel="can_leader1",
                firmware="v1",
                feedback_timeout=0.2,
                init_timeout=30.0,
                signs=[1] * 6,
                offsets_deg=[0.0] * 6,
                gripper_max=0.07,
                gripper_force=1.0,
            )

        self.assertEqual(events, ["leader_mode", "gripper_input"])
        self.assertEqual(expert.gripper_width, 0.04)
        wait_gripper.assert_called_once_with(
            gripper, 30.0, not_before=123.0
        )
        # Avoid hardware cleanup in this unit test.
        expert.drag_active = False

    def test_missing_leader_gripper_feedback_uses_follower_width_fallback(self):
        robot = Mock()
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0)
        )
        robot.OPTIONS.EFFECTOR.AGX_GRIPPER = "gripper"
        gripper = Mock()
        robot.init_effector.return_value = gripper

        with patch(
            "robotarm.modeling.hg_dagger.base.connect_robot",
            return_value=(robot, {}),
        ), patch(
            "robotarm.modeling.hg_dagger.wait_initial_leader_joints",
            return_value=([0.0] * 6, "feedback"),
        ), patch(
            "robotarm.modeling.hg_dagger.read_gripper_message",
            side_effect=RuntimeError("no leader width feedback"),
        ), patch(
            "robotarm.modeling.hg_dagger.wait_leader_gripper_command",
        ) as wait_gripper, patch(
            "robotarm.modeling.hg_dagger.ensure_receive_running",
        ):
            expert = LeaderExpert(
                interface="socketcan",
                channel="can_leader1",
                firmware="v1",
                feedback_timeout=0.2,
                init_timeout=30.0,
                signs=[1] * 6,
                offsets_deg=[0.0] * 6,
                gripper_max=0.07,
                gripper_force=1.0,
                initial_gripper_width=0.03,
            )

        self.assertEqual(expert.gripper_width, 0.03)
        robot.set_leader_mode.assert_not_called()
        wait_gripper.assert_not_called()
        expert.robot = None


if __name__ == "__main__":
    unittest.main()
