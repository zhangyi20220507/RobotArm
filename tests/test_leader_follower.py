"""Offline tests for dual-Piper leader-follower safety logic."""

import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robotarm.teleop.leader_follower.runtime import (
    alignment_complete,
    command_joint_bounds,
    clip_command_to_bounds,
    clip_feedback_zero_offsets,
    enter_follower_j,
    ensure_receive_running,
    exit_leader_drag_mode,
    feedback_age,
    joint_feedback_age,
    limit_joint_target,
    make_parser,
    MotionDiagnostics,
    normalize_hold_roundoff,
    prepare_leader_target,
    main,
    map_leader_joints,
    read_leader_gripper_command,
    run_check,
    update_leader_joint_message,
    validate_args,
    validate_target_bounds,
)
from robotarm.core.robot.safety import safe_joint_bounds


class LeaderFollowerTests(unittest.TestCase):
    def test_defaults_are_dry_run_and_use_distinct_can_channels(self):
        args = make_parser().parse_args([])
        validate_args(args)
        self.assertFalse(args.execute)
        self.assertNotEqual(args.leader_channel, args.follower_channel)
        self.assertFalse(args.check_only)
        self.assertEqual(args.leader_init_timeout, 30.0)
        self.assertEqual(args.joint_limit_margin_deg, 0.5)

    def test_check_only_reports_deltas_and_ages_without_controlling_either_arm(self):
        leader, follower = Mock(), Mock()
        with ExitStack() as stack:
            connect = stack.enter_context(patch(
                "robotarm.teleop.leader_follower.runtime.base.connect_robot",
                Mock(side_effect=[(leader, {}), (follower, {})]),
            ))
            stack.enter_context(patch(
                "robotarm.teleop.leader_follower.runtime.wait_initial_leader_joints",
                Mock(return_value=([0.0] * 6, "feedback")),
            ))
            stack.enter_context(patch(
                "robotarm.teleop.leader_follower.runtime.wait_joint_message",
                Mock(return_value=([0.1] * 6, 0.0)),
            ))
            stack.enter_context(patch(
                "robotarm.teleop.leader_follower.runtime.joint_feedback_age",
                Mock(return_value=0.05),
            ))
            stack.enter_context(patch(
                "robotarm.teleop.leader_follower.runtime.base.disconnect_quietly"
            ))
            output = stack.enter_context(patch("builtins.print"))
            self.assertEqual(run_check(make_parser().parse_args(["--check-only"])), 0)
        # piper-setup gives the two buses stable role-based names.
        self.assertEqual(
            [call.args[0].channel for call in connect.call_args_list],
            ["can_leader1", "can_follower1"],
        )
        self.assertTrue(any(
            "delta=  -5.730 deg" in str(call) for call in output.call_args_list
        ))
        self.assertTrue(any(
            "feedback age: leader=50.0 ms, follower=0.0 ms" in str(call)
            for call in output.call_args_list
        ))
        for arm in (leader, follower):
            arm.move_j.assert_not_called()
            arm.move_js.assert_not_called()
            arm.enable.assert_not_called()
            arm.set_motion_mode.assert_not_called()
            arm.set_leader_mode.assert_not_called()
            arm.set_follower_mode.assert_not_called()

    def test_recording_defaults_to_two_distinct_cameras(self):
        args = make_parser().parse_args([])
        self.assertEqual(args.global_camera_serial, "347522071736")
        self.assertEqual(args.wrist_camera_serial, "260322272554")
        validate_args(args)
        with patch("robotarm.teleop.leader_follower.runtime.base.connect_robot") as connect, patch("builtins.print") as output:
            self.assertEqual(main([]), 0)
        connect.assert_not_called()
        self.assertTrue(any(
            "cameras={'global': '347522071736', 'wrist': '260322272554'}" in str(call)
            for call in output.call_args_list
        ))

    def test_recording_rejects_missing_or_duplicate_camera_serials(self):
        for serial in (None, "", "347522071736"):
            args = make_parser().parse_args([])
            args.wrist_camera_serial = serial
            with self.assertRaisesRegex(ValueError, "two cameras|must differ"):
                validate_args(args)
        args = make_parser().parse_args(["--control-only"])
        args.wrist_camera_serial = None
        validate_args(args)

    def test_control_only_execute_does_not_require_recording_metadata(self):
        args = make_parser().parse_args(["--execute", "--control-only"])
        validate_args(args)
        args.control_only = False
        with self.assertRaisesRegex(ValueError, "setup-id"):
            validate_args(args)

    def test_control_only_dry_run_does_not_initialize_hardware_or_recorder(self):
        with patch("robotarm.teleop.leader_follower.runtime.base.connect_robot") as connect, patch(
            "robotarm.teleop.leader_follower.runtime.RawEpisodeRecorder"
        ) as recorder, patch("builtins.print") as output:
            self.assertEqual(main(["--control-only"]), 0)
        connect.assert_not_called()
        recorder.assert_not_called()
        self.assertIn("cameras={}", output.call_args_list[0].args[0])

    def test_alignment_waits_for_command_and_actual_position(self):
        target = [0.1] * 6
        self.assertFalse(alignment_complete(target, [0.0] * 6, target))
        self.assertFalse(alignment_complete(target, target, [0.0] * 6))
        self.assertTrue(alignment_complete(target, target, target))

    def run_alignment_runtime(self, keys, stale_during_alignment=False, residual_deg=0.0,
                              snapshot=0.01, follower_stuck=False, recording=False):
        actual = [0.0] * 6
        leader, follower, gripper = Mock(), Mock(), Mock()
        leader._parser = None
        leader._ctx = None
        follower._ctx = None
        follower.init_effector.return_value = gripper
        follower.fk.return_value = [0.0] * 6
        follower.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(arm_status=0))
        follower.get_joint_angles.side_effect = lambda: SimpleNamespace(msg=list(actual))
        def track(target):
            if not follower_stuck:
                actual[:] = [max(0.0, value - math.radians(residual_deg)) for value in target]
        follower.move_js.side_effect = track
        follower.move_j.side_effect = track
        reader = Mock()
        reader.__enter__ = Mock(return_value=reader)
        reader.__exit__ = Mock(return_value=False)
        reader.poll.side_effect = keys
        updates = 0

        def leader_input(*args):
            nonlocal updates
            updates += 1
            # The main loop sees the leader move after a; alignment must retain its snapshot.
            return ([snapshot] * 6 if updates <= 2 else [snapshot + 0.03] * 6), 0.0

        def follower_input(*args):
            age = 1.0 if stale_during_alignment and follower.move_j.called else 0.0
            return list(actual), age

        prefix = "robotarm.teleop.leader_follower.runtime."
        replacements = {
            "sys.stdin.isatty": Mock(return_value=True),
            "base.connect_robot": Mock(side_effect=[(leader, {}), (follower, {})]),
            "wait_initial_leader_joints": Mock(return_value=([0.0] * 6, "feedback")),
            "wait_joint_message": Mock(return_value=([0.0] * 6, 0.0)),
            "validate_joint_feedback": Mock(),
            "ensure_startup_status": Mock(),
            "base.wait_gripper_width": Mock(return_value=0.02),
            "base.get_all_joints_enabled_state": Mock(return_value=True),
            "safe_joint_bounds": Mock(return_value=[(-1.0, 1.0)] * 6),
            "KeyReader": Mock(return_value=reader),
            "update_leader_joint_message": Mock(side_effect=leader_input),
            "read_joint_message": Mock(side_effect=follower_input),
            "read_gripper_message": Mock(return_value=(0.02, 0.0)),
            "read_leader_gripper_command": Mock(return_value=(0.02, 0.0)),
            "enter_follower_js": Mock(),
            "enter_follower_j": Mock(),
            "time.monotonic": Mock(side_effect=(i * 0.005 for i in range(10000))),
            "exit_leader_drag_mode": Mock(),
            "base.disconnect_quietly": Mock(),
            "RawEpisodeRecorder": Mock(),
            "time.sleep": Mock(),
        }
        recorder = replacements["RawEpisodeRecorder"].return_value
        recorder.active = False
        recorder.start_episode.side_effect = lambda: setattr(recorder, "active", True)
        recorder.save_episode.side_effect = lambda: setattr(recorder, "active", False)
        follower.attach_mock(replacements["enter_follower_js"], "enter_js")
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch(prefix + name, replacement))
            output = stack.enter_context(patch("builtins.print"))
            mode_args = ["--setup-id", "offline-test", "--task", "test"] if recording else ["--control-only"]
            result = main(["--execute", "--allow-enabled-start", *mode_args])
        if recording:
            recorder.start_episode.assert_called_once()
            self.assertTrue(recorder.capture.called)
            for call in recorder.capture.call_args_list:
                self.assertTrue(all(-1.0 <= value <= 1.0 for value in call.args[2][:6]))
            self.assertTrue(any(max(call.args[5]["leader_state"][:6]) > 1.0
                                for call in recorder.capture.call_args_list))
            recorder.discard_episode.assert_not_called()
            recorder.mark_incomplete.assert_not_called()
        else:
            replacements["RawEpisodeRecorder"].assert_not_called()
        leader.enable.assert_not_called()
        leader.move_j.assert_not_called()
        leader.move_js.assert_not_called()
        leader.set_follower_mode.assert_not_called()
        return result, follower, output

    def test_diagnostics_show_both_channels_without_starting_motion(self):
        result, follower, output = self.run_alignment_runtime(["l", "s", "q"])
        self.assertEqual(result, 0)
        self.assertTrue(any(
            "leader can_leader1" in str(call) for call in output.call_args_list
        ))
        self.assertTrue(any(
            "follower can_follower1" in str(call) for call in output.call_args_list
        ))
        self.assertTrue(any("following=False" in str(call) for call in output.call_args_list))
        follower.set_motion_mode.assert_not_called()
        follower.enable.assert_not_called()

    def test_live_alignment_freezes_target_and_waits_for_f(self):
        result, follower, output = self.run_alignment_runtime(["l", "a"] + [None] * 120 + ["q"])
        self.assertEqual(result, 0)
        commands = [call.args[0] for call in follower.move_j.call_args_list]
        self.assertTrue(commands)
        self.assertTrue(all(max(command) <= 0.01 + 1e-9 for command in commands))
        self.assertAlmostEqual(commands[-1][0], 0.01)
        for previous, current in zip(commands, commands[1:]):
            self.assertLessEqual(max(abs(a-b) for a, b in zip(previous, current)), math.radians(5)/50 + 1e-9)
        self.assertTrue(any("alignment complete" in str(call) for call in output.call_args_list))
        self.assertTrue(all(call.args == (10,) for call in follower.set_speed_percent.call_args_list))
        follower.init_effector.return_value.move_gripper_m.assert_not_called()

    def test_alignment_accepts_stable_small_physical_residual(self):
        result, follower, output = self.run_alignment_runtime(
            ["l", "a"] + [None] * 300 + ["q"], residual_deg=1.86, snapshot=0.1,
        )
        self.assertEqual(result, 0)
        self.assertTrue(any("alignment complete" in str(call) for call in output.call_args_list))
        follower.move_js.assert_not_called()

    def test_alignment_rejects_residual_above_tolerance(self):
        result, _, output = self.run_alignment_runtime(
            ["l", "a"] + [None] * 300 + ["q"], residual_deg=2.1, snapshot=0.1,
        )
        self.assertEqual(result, 0)
        self.assertFalse(any("alignment complete" in str(call) for call in output.call_args_list))

    def test_following_limiter_does_not_trip_on_numerical_boundary(self):
        limit = math.radians(10)
        result = limit_joint_target(
            [1.0] * 6, [limit + 1e-16] * 6, [0.0] * 6,
            max_step=0.01, max_following_error=limit,
        )
        self.assertLessEqual(max(result), limit + 1e-9)

    def test_alignment_mode_uses_planned_j_without_js(self):
        robot, gripper = Mock(), Mock()
        robot.OPTIONS.MOTION_MODE.J = "j"
        with patch("robotarm.teleop.leader_follower.runtime.base.set_gripper_enabled"), patch(
            "robotarm.teleop.leader_follower.runtime.time.sleep"
        ):
            enter_follower_j(robot, gripper, [0.0] * 6, 0.02, 1.0, True)
        robot.move_js.assert_not_called()
        robot.set_motion_mode.assert_called_once_with("j")
        robot.enable.assert_not_called()

    def test_f_starts_following_after_alignment(self):
        result, follower, output = self.run_alignment_runtime(
            ["l", "a"] + [None] * 120 + ["f", "q"]
        )
        self.assertEqual(result, 0)
        self.assertTrue(any("following active" in str(call) for call in output.call_args_list))
        self.assertEqual(follower.set_speed_percent.call_args_list[-1].args, (100,))

    def test_following_speed_is_set_before_entering_js_mode(self):
        result, follower, _ = self.run_alignment_runtime(
            ["l", "a"] + [None] * 120 + ["f", "q"]
        )
        self.assertEqual(result, 0)
        calls = follower.mock_calls
        speed_index = next(i for i, call in enumerate(calls)
                           if call[0] == "set_speed_percent" and call.args == (100,))
        js_index = next(i for i, call in enumerate(calls) if call[0] == "enter_js")
        self.assertLess(speed_index, js_index)
        self.assertFalse(any(call[0] == "set_speed_percent" for call in calls[js_index+1:]))

    def test_f_rejects_large_initial_joint_difference(self):
        result, follower, output = self.run_alignment_runtime(["l", "f", "q"], snapshot=0.1)
        self.assertEqual(result, 0)
        self.assertTrue(any("startup alignment rejected" in str(call) for call in output.call_args_list))
        self.assertFalse(any("following active" in str(call) for call in output.call_args_list))
        follower.move_js.assert_not_called()

    def test_f_cancels_alignment_without_starting_following(self):
        result, follower, output = self.run_alignment_runtime(["l", "a", "f", "q"])
        self.assertEqual(result, 0)
        self.assertTrue(any("alignment cancelled" in str(call) for call in output.call_args_list))
        self.assertFalse(any("following active" in str(call) for call in output.call_args_list))

    def test_alignment_reports_stalled_follower_instead_of_waiting_until_timeout(self):
        with self.assertRaisesRegex(RuntimeError, 'alignment did not move for 5 s'):
            self.run_alignment_runtime(
                ["l", "a"] + [None] * 700 + ["q"],
                snapshot=0.8, follower_stuck=True,
            )

    def test_alignment_stops_on_stale_follower_feedback(self):
        with self.assertRaisesRegex(RuntimeError, "stale follower CAN feedback"):
            self.run_alignment_runtime(["l", "a", None, "q"], stale_during_alignment=True)

    def test_dead_receiver_is_rejected_even_with_cached_angles(self):
        robot = SimpleNamespace(_ctx=SimpleNamespace(
            comm=SimpleNamespace(last_error=None), _read_th=Mock(is_alive=Mock(return_value=False)),
        ))
        with self.assertRaisesRegex(RuntimeError, "leader CAN receiver.*stopped"):
            ensure_receive_running(robot, "leader")

    def test_parser_exception_is_reported_from_communication_error(self):
        robot = SimpleNamespace(_ctx=SimpleNamespace(
            comm=SimpleNamespace(last_error=ValueError("invalid leader frame")),
            _read_th=Mock(is_alive=Mock(return_value=False)),
        ))
        with self.assertRaisesRegex(RuntimeError, "ValueError: invalid leader frame"):
            ensure_receive_running(robot, "leader")

    def test_healthy_receiver_allows_event_driven_idle(self):
        robot = SimpleNamespace(_ctx=SimpleNamespace(
            comm=SimpleNamespace(last_error=None), _read_th=Mock(is_alive=Mock(return_value=True)),
        ))
        ensure_receive_running(robot, "leader")

    def test_motion_log_reports_real_sent_target_and_snapshot_delta(self):
        log = MotionDiagnostics()
        args = make_parser().parse_args([])
        leader, follower = Mock(), Mock()
        leader._parser = None
        actual = [0.0] * 6
        with patch("builtins.print") as output:
            log.emit(10.0, args, leader, follower, [0.1]*6, [0.1]*6,
                     [math.radians(2)]*6, actual, 17.0, 0.01, True, False)
            actual[0] = math.radians(1)
            log.emit(10.5, args, leader, follower, [0.1]*6, [0.1]*6,
                     [math.radians(2)]*6, actual, 17.5, 0.01, True, False)
        lines = [call.args[0] for call in output.call_args_list]
        self.assertTrue(any("leader_age=17000ms" in line for line in lines))
        self.assertTrue(any("sent_target=[+2.00" in line for line in lines))
        self.assertTrue(any("target_minus_actual=[+1.00, +2.00" in line for line in lines))
        self.assertTrue(any("follower_delta over 0.500s=[+1.00, +0.00" in line for line in lines))
        self.assertTrue(any("first snapshot" in line for line in lines))

    def test_motion_log_throttles_automatic_output_but_s_can_force_it(self):
        log = MotionDiagnostics()
        args = make_parser().parse_args([])
        leader, follower = Mock(), Mock()
        leader._parser = None
        values = [0.0]*6
        with patch("builtins.print") as output:
            log.emit(10.0, args, leader, follower, values, values, None, values,
                     0.0, 0.0, False, False, force=True)
            count = output.call_count
            log.emit(10.1, args, leader, follower, values, values, None, values,
                     0.0, 0.0, True, False)
            self.assertEqual(output.call_count, count)
            log.emit(10.1, args, leader, follower, values, values, None, values,
                     0.0, 0.0, True, False, force=True)
            self.assertGreater(output.call_count, count)
        self.assertTrue(any("sent_target=none" in str(call) for call in output.call_args_list))

    def test_following_shows_compact_status_without_automatic_motion_dump(self):
        result, _, output = self.run_alignment_runtime(
            ["l", "a"] + [None]*120 + ["f"] + [None]*120 + ["q"]
        )
        self.assertEqual(result, 0)
        printed = " ".join(str(call) for call in output.call_args_list)
        self.assertIn("[状态] 跟随中", printed)
        self.assertNotIn("[motion]", printed)
        self.assertNotIn("[diagnostic]", printed)

    def test_s_prints_detailed_snapshot_while_following(self):
        result, _, output = self.run_alignment_runtime(
            ["l", "a"] + [None]*120 + ["f", "s", "q"]
        )
        self.assertEqual(result, 0)
        printed = " ".join(str(call) for call in output.call_args_list)
        self.assertIn("[motion] mode=following", printed)
        self.assertIn("[diagnostic] leader", printed)
        self.assertIn("[diagnostic] follower", printed)

    def test_status_line_describes_recording_and_saved_episode(self):
        from robotarm.teleop.leader_follower.runtime import status_line
        self.assertIn("录制中", status_line(True, False, True, True, False))
        self.assertIn("已保存，可按 h 回零", status_line(True, False, True, False, True))
        self.assertIn("剩余 1.25°", status_line(False, True, True, False, False, 1.25))

    def test_recording_and_control_clamp_j5_without_widening_safe_bounds(self):
        bounds = [(-2.0, 2.0)] * 6
        bounds[4] = (-math.radians(69.5), math.radians(69.5))
        for sign in (-1, 1):
            target = [0.0]*6
            target[4] = sign * math.radians(70.0)
            applied = prepare_leader_target(target, bounds, True)
            self.assertAlmostEqual(math.degrees(applied[4]), sign * 69.5)
            self.assertAlmostEqual(math.degrees(target[4]), sign * 70.0)
            self.assertEqual(prepare_leader_target(target, bounds, False), applied)
        target[4] = math.radians(68.0)
        self.assertEqual(prepare_leader_target(target, bounds, True), target)
        self.assertEqual(prepare_leader_target(target, bounds, False), target)

    def test_feedback_tolerance_preserves_measured_values_but_rejects_large_excursions(self):
        from robotarm.teleop.leader_follower.runtime import validate_joint_feedback
        config = {"joint_limits": {f"joint{i}": (-1.0, 1.0) for i in range(1, 7)}}
        for sign in (-1, 1):
            joints = [sign * (1.0 + math.radians(4.9))] * 6
            original = joints[:]
            validate_joint_feedback(joints, config)
            self.assertEqual(joints, original)
            with self.assertRaisesRegex(RuntimeError, '反馈容差'):
                validate_joint_feedback([sign * (1.0 + math.radians(5.1))] * 6, config)
        for invalid in (math.nan, math.inf, -math.inf):
            with self.assertRaises(RuntimeError):
                validate_joint_feedback([invalid] * 6, config)
            with self.assertRaises(RuntimeError):
                prepare_leader_target([invalid] * 6, [(-1.0, 1.0)] * 6, False)

    def test_nonfinite_target_is_rejected_even_in_control_only(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(RuntimeError, "finite"):
                prepare_leader_target([invalid]*6, [(-1.0,1.0)]*6, True)

    def test_hold_normalizes_only_tiny_hard_limit_roundoff(self):
        config = {"joint_limits": {f"joint{i}": (-1.22173, 1.22173) for i in range(1,7)}}
        result = normalize_hold_roundoff([math.radians(70), -math.radians(70), 0.1, 0.2, 1.3, 0.0], config)
        self.assertEqual(result[0], 1.22173)
        self.assertEqual(result[1], -1.22173)
        self.assertEqual(result[4], 1.3)

    def test_control_only_following_continues_at_safe_limit(self):
        result, follower, output = self.run_alignment_runtime(
            ["l", "a"] + [None]*700 + ["f"] + [None]*120 + ["s", "q"], snapshot=0.98,
        )
        self.assertEqual(result, 0)
        self.assertTrue(any("following active" in str(call) for call in output.call_args_list))
        self.assertTrue(any("joint_limit_active" in str(call) for call in output.call_args_list))
        for call in follower.move_js.call_args_list:
            self.assertLessEqual(max(call.args[0]), 1.0)
        self.assertAlmostEqual(follower.move_js.call_args_list[-1].args[0][0], 1.0)

    def test_recording_continues_at_limit_with_bounded_actions_and_raw_leader_state(self):
        result, follower, output = self.run_alignment_runtime(
            ["l", "a"] + [None]*700 + ["f", "r"] + [None]*120 + ["r", "q"],
            snapshot=0.98, recording=True,
        )
        self.assertEqual(result, 0)
        self.assertTrue(follower.move_js.called)
        self.assertTrue(any("目标已自动限幅" in str(call) for call in output.call_args_list))

    def test_same_can_endpoint_is_rejected(self):
        args = make_parser().parse_args(
            ["--leader-channel", "can0", "--follower-channel", "can0"]
        )
        with self.assertRaisesRegex(ValueError, "must differ"):
            validate_args(args)

    def test_joint_mapping_applies_sign_and_offset(self):
        result = map_leader_joints(
            [1, 2, 3, 4, 5, 6],
            [1, -1, 1, -1, 1, -1],
            [0, 90, 0, 0, -90, 0],
        )
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(result[1], -2 + math.pi / 2)
        self.assertAlmostEqual(result[4], 5 - math.pi / 2)

    def test_joint_step_limit_preserves_ratios(self):
        previous = [0.0] * 6
        requested = [0.2, -0.1, 0.05, 0, 0, 0]
        result = limit_joint_target(
            requested, previous, previous, max_step=0.02,
            max_following_error=0.5,
        )
        self.assertAlmostEqual(result[0], 0.02)
        self.assertAlmostEqual(result[1], -0.01)
        self.assertAlmostEqual(result[2], 0.005)

    def test_existing_following_error_stops_control(self):
        with self.assertRaisesRegex(RuntimeError, "following error"):
            limit_joint_target(
                [0.0] * 6, [0.2] * 6, [0.0] * 6,
                max_step=0.01, max_following_error=0.1,
            )

    def test_target_outside_safe_bounds_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "outside"):
            validate_target_bounds([0, 0, 0, 0, 0, 2], [(-1, 1)] * 6)

    def test_configurable_joint_margin_keeps_target_inside_hardware_limit(self):
        config = {
            "joint_limits": {
                f"joint{index}": (-math.radians(70), math.radians(70))
                for index in range(1, 7)
            }
        }
        bounds = safe_joint_bounds(config, margin_degrees=0.5)
        self.assertAlmostEqual(math.degrees(bounds[4][0]), -69.5)
        self.assertAlmostEqual(math.degrees(bounds[4][1]), 69.5)
        validate_target_bounds(
            [0, 0, 0, 0, math.radians(69.11), 0], bounds
        )
        with self.assertRaisesRegex(RuntimeError, "safe range"):
            validate_target_bounds(
                [0, 0, 0, 0, math.radians(69.6), 0], bounds
            )

    def test_alignment_commands_respect_sdk_limits_despite_zero_offset_feedback(self):
        config = {'joint_limits': {
            'joint1': (-2.6, 2.6), 'joint2': (0.0, math.pi),
            'joint3': (-2.96706, 0.0), 'joint4': (-1.7, 1.7),
            'joint5': (-1.22173, 1.22173), 'joint6': (-2.0, 2.0),
        }}
        bounds = command_joint_bounds(config, margin_degrees=0.5)
        self.assertEqual(bounds[1][0], 0.0)
        self.assertEqual(bounds[2][1], 0.0)
        self.assertAlmostEqual(math.degrees(bounds[4][1]), math.degrees(1.22173) - 0.5)
        # The measured pose and the mapped leader can slightly cross a zero
        # boundary, but every target actually sent must remain commandable.
        feedback = [0.0, -0.02976, 0.02321, 0.0, 0.0, 0.0]
        mapped = [0.0, -0.0158, 0.0117, 0.0, 0.1, 0.0]
        command = clip_command_to_bounds(feedback, bounds)
        target = prepare_leader_target(
            clip_feedback_zero_offsets(mapped, config), bounds, False
        )
        self.assertEqual(command[1:3], [0.0, 0.0])
        self.assertEqual(target[1:3], [0.0, 0.0])
        for _ in range(15):
            command = limit_joint_target(
                target, command, feedback, math.radians(0.1), math.radians(10)
            )
            validate_target_bounds(command, bounds)
            self.assertGreaterEqual(command[1], 0.0)
            self.assertLessEqual(command[2], 0.0)
        # Recording also clips mapped targets, without widening hardware bounds.
        mapped[4] = math.radians(70.0)
        clipped = prepare_leader_target(
            clip_feedback_zero_offsets(mapped, config), bounds, False
        )
        self.assertEqual(clipped[4], bounds[4][1])

    def test_feedback_age_uses_can_wall_clock_timestamp(self):
        message = SimpleNamespace(timestamp=99.8)
        self.assertAlmostEqual(feedback_age(message, now_wall=100.0), 0.2)
        self.assertTrue(math.isinf(feedback_age(SimpleNamespace(timestamp=0), 100)))

    def test_joint_feedback_age_requires_all_three_can_frames(self):
        robot = SimpleNamespace(
            _parser=SimpleNamespace(
                joint_12=SimpleNamespace(timestamp=99.9),
                joint_34=SimpleNamespace(timestamp=99.8),
                joint_56=SimpleNamespace(timestamp=99.95),
            )
        )
        self.assertAlmostEqual(
            joint_feedback_age(robot, SimpleNamespace(timestamp=99.95), 100.0),
            0.2,
        )
        robot._parser.joint_34 = None
        self.assertTrue(
            math.isinf(
                joint_feedback_age(
                    robot, SimpleNamespace(timestamp=99.95), 100.0
                )
            )
        )

    def test_drag_input_merges_only_received_joint_pairs(self):
        config = {
            "joint_limits": {
                f"joint{index}": (-10.0, 10.0) for index in range(1, 7)
            }
        }
        robot = SimpleNamespace(
            _parser=SimpleNamespace(
                leader_joint_12=None,
                leader_joint_34=None,
                leader_joint_56=SimpleNamespace(
                    timestamp=99.9,
                    msg=SimpleNamespace(joint_5=0.5, joint_6=-0.6),
                ),
            )
        )
        result, age = update_leader_joint_message(
            robot, [0.1, 0.2, 0.3, 0.4, 0.0, 0.0], config
        )
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4, 0.5, -0.6])
        self.assertTrue(math.isfinite(age))

    def test_fresh_normal_feedback_overrides_old_drag_input(self):
        robot = SimpleNamespace(_parser=SimpleNamespace(
            leader_joint_12=SimpleNamespace(timestamp=84.0, msg=SimpleNamespace(joint_1=0.1, joint_2=0.2)),
            joint_12=SimpleNamespace(timestamp=99.99, msg=SimpleNamespace(joint_1=0.3, joint_2=0.4)),
        ))
        config = {"joint_limits": {f"joint{i}": (-1.0, 1.0) for i in range(1, 7)}}
        with patch("robotarm.teleop.leader_follower.runtime.time.time", return_value=100.0):
            result, age = update_leader_joint_message(robot, [0.0] * 6, config)
        self.assertEqual(result, [0.3, 0.4, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(age, 0.01)

    def test_sources_are_selected_per_pair_without_rolling_back_new_drag_input(self):
        robot = SimpleNamespace(_parser=SimpleNamespace(
            leader_joint_12=SimpleNamespace(timestamp=99.99, msg=SimpleNamespace(joint_1=0.3, joint_2=0.4)),
            joint_12=SimpleNamespace(timestamp=90.0, msg=SimpleNamespace(joint_1=0.1, joint_2=0.2)),
            leader_joint_34=SimpleNamespace(timestamp=84.0, msg=SimpleNamespace(joint_3=0.0, joint_4=0.0)),
            joint_34=SimpleNamespace(timestamp=99.98, msg=SimpleNamespace(joint_3=0.5, joint_4=0.6)),
        ))
        config = {"joint_limits": {f"joint{i}": (-1.0, 1.0) for i in range(1, 7)}}
        with patch("robotarm.teleop.leader_follower.runtime.time.time", return_value=100.0):
            result, _ = update_leader_joint_message(robot, [0.0] * 6, config)
        self.assertEqual(result, [0.3, 0.4, 0.5, 0.6, 0.0, 0.0])

    def test_ordinary_feedback_updates_next_target_while_drag_stream_is_stale(self):
        normal = SimpleNamespace(timestamp=99.0, msg=SimpleNamespace(joint_1=0.0, joint_2=0.0))
        robot = SimpleNamespace(_parser=SimpleNamespace(
            leader_joint_12=SimpleNamespace(timestamp=84.0, msg=SimpleNamespace(joint_1=0.0, joint_2=0.0)),
            joint_12=normal,
        ))
        config = {"joint_limits": {f"joint{i}": (-1.0, 1.0) for i in range(1, 7)}}
        with patch("robotarm.teleop.leader_follower.runtime.time.time", return_value=100.0):
            previous, _ = update_leader_joint_message(robot, [0.0] * 6, config)
            normal.timestamp = 99.99
            normal.msg.joint_1 = math.radians(1.0)
            target, age = update_leader_joint_message(robot, previous, config)
        command = limit_joint_target(target, previous, [0.0] * 6, math.radians(3), math.radians(10))
        self.assertAlmostEqual(command[0], math.radians(1.0))
        self.assertLess(age, 0.02)

    def test_drag_gripper_uses_159_command_and_retains_previous(self):
        gripper = SimpleNamespace(get_gripper_ctrl_states=lambda: None)
        value, age = read_leader_gripper_command(gripper, 0.03)
        self.assertEqual(value, 0.03)
        self.assertTrue(math.isinf(age))
        message = SimpleNamespace(
            timestamp=99.9,
            msg=SimpleNamespace(value=0.04, status_code=1),
        )
        gripper.get_gripper_ctrl_states = lambda: message
        value, age = read_leader_gripper_command(gripper, 0.03)
        self.assertEqual(value, 0.04)
        self.assertTrue(math.isfinite(age))

    def test_unsafe_limits_are_rejected(self):
        args = make_parser().parse_args(["--max-step-joint-deg", "4"])
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_args(args)
        args = make_parser().parse_args(["--joint-limit-margin-deg", "0.1"])
        with self.assertRaisesRegex(ValueError, "joint-limit-margin-deg"):
            validate_args(args)

    def test_startup_alignment_can_exceed_closed_loop_error_limit(self):
        args = make_parser().parse_args(
            [
                "--startup-alignment-deg", "15",
                "--max-following-error-deg", "8",
            ]
        )
        validate_args(args)

    def test_exit_drag_mode_enters_enabled_js_hold(self):
        robot = Mock()
        robot.OPTIONS.MOTION_MODE.J = "j"
        robot.OPTIONS.MOTION_MODE.JS = "js"
        robot.enable.return_value = True
        robot.get_joints_enable_status_list.return_value = [True] * 6
        gripper = Mock()
        gripper.get_gripper_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(
                foc_status=SimpleNamespace(driver_enable_status=True)
            )
        )
        joints = [0.1, 0.2, -0.3, 0.0, 0.2, 0.0]
        with patch("robotarm.teleop.leader_follower.runtime.time.sleep"):
            exit_leader_drag_mode(robot, joints, gripper, 0.04, 1.0)
        robot.set_follower_mode.assert_called_once_with()
        robot.enable.assert_called()
        robot.set_motion_mode.assert_any_call("j")
        robot.set_motion_mode.assert_any_call("js")
        robot.move_js.assert_called_once_with(joints)

    def test_exit_drag_mode_rejects_invalid_hold_feedback(self):
        with self.assertRaisesRegex(RuntimeError, "six valid joints"):
            exit_leader_drag_mode(Mock(), [0.0] * 5, None, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
