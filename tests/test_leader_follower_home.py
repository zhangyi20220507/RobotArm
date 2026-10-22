"""Offline episode-to-home integration checks; all arms and recorders are mocks."""
import math
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robotarm.teleop.leader_follower import runtime as lf


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def leader_pair_frames(joints, stamp):
    return SimpleNamespace(
        joint_12=SimpleNamespace(
            timestamp=stamp,
            msg=SimpleNamespace(joint_1=joints[0], joint_2=joints[1]),
        ),
        joint_34=SimpleNamespace(
            timestamp=stamp,
            msg=SimpleNamespace(joint_3=joints[2], joint_4=joints[3]),
        ),
        joint_56=SimpleNamespace(
            timestamp=stamp,
            msg=SimpleNamespace(joint_5=joints[4], joint_6=joints[5]),
        ),
    )


class LeaderFollowerHomeTests(unittest.TestCase):
    def test_native_leader_homes_and_holds_before_follower(self):
        follower, leader = Mock(), Mock()
        fg, lg = Mock(), Mock()
        ordered = Mock()
        with patch.object(lf, 'home_leader_native', return_value=[0.0] * 6) as native, patch.object(
            lf, 'exit_leader_drag_mode'
        ) as hold, patch.object(lf, 'close_gripper_slowly') as close, patch.object(
            lf, 'home_one_arm'
        ) as follower_home:
            ordered.attach_mock(native, 'native')
            ordered.attach_mock(hold, 'hold')
            ordered.attach_mock(close, 'close')
            ordered.attach_mock(follower_home, 'follower')
            result = lf.home_both_sequentially(
                follower, {}, 'can0', leader, {}, 'can1',
                fg, lg, 1.0, 10.0, 60, lambda:False,
                [0.2] * 6, 0.02,
            )
        self.assertEqual(result, [0.0] * 6)
        self.assertEqual([call[0] for call in ordered.mock_calls],
                         ['native', 'hold', 'close', 'follower'])
        leader.electronic_emergency_stop.assert_not_called()
        follower.electronic_emergency_stop.assert_not_called()

    def test_native_leader_failure_estops_leader_before_follower_starts(self):
        follower, leader = Mock(), Mock()
        def stall_after_send(*_args, **kwargs):
            kwargs['on_command_start']()
            raise RuntimeError('stalled')
        with patch.object(lf, 'home_leader_native', side_effect=stall_after_send), patch.object(
            lf, 'home_one_arm'
        ) as follower_home, patch('builtins.print') as output:
            with self.assertRaisesRegex(RuntimeError, 'stalled'):
                lf.home_both_sequentially(
                    follower, {}, 'can0', leader, {}, 'can1',
                    Mock(), Mock(), 1.0, 10.0, 60, lambda:False,
                    [0.2] * 6, 0.02,
                )
        leader.electronic_emergency_stop.assert_called_once()
        follower.electronic_emergency_stop.assert_not_called()
        follower_home.assert_not_called()
        self.assertTrue(any('electronic emergency stop sent to can1' in str(call)
                            for call in output.call_args_list))

    def test_leader_gripper_closes_at_bounded_speed_after_native_home(self):
        clock = FakeClock()
        width = [0.02]
        gripper = Mock()
        gripper.get_gripper_status.side_effect = lambda: SimpleNamespace(
            msg=SimpleNamespace(mode='width', value=width[0]),
            timestamp=clock.now,
        )
        gripper.move_gripper_m.side_effect = lambda target, _force: width.__setitem__(0, target)
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', clock.sleep), patch('builtins.print'):
            lf.close_gripper_slowly(gripper, 1.0, 10.0, 10.0, lambda:False)
        commands = [call.args[0] for call in gripper.move_gripper_m.call_args_list]
        self.assertEqual(commands[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(commands, commands[1:])))
        self.assertLessEqual(clock.now - 1000.0, 4.0)

    def test_leader_gripper_stall_raises_before_timeout(self):
        clock = FakeClock()
        gripper = Mock()
        gripper.get_gripper_status.side_effect = lambda: SimpleNamespace(
            msg=SimpleNamespace(mode='width', value=0.02),
            timestamp=clock.now,
        )
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', clock.sleep):
            with self.assertRaisesRegex(RuntimeError, 'no closing progress'):
                lf.close_gripper_slowly(gripper, 1.0, 10.0, 20.0, lambda:False)
        self.assertLess(clock.now - 1000.0, 10.0)

    def test_native_home_skips_command_when_leader_is_already_zero(self):
        robot = Mock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        with patch.object(lf.base, 'ensure_arm_status_normal'), patch('builtins.print'):
            result = lf.home_leader_native(
                robot, config, 'can1', [0.0]*6, 20.0, lambda:False
            )
        self.assertEqual(result, [0.0]*6)
        robot.move_leader_to_home.assert_not_called()

    def test_native_home_waits_for_post_command_joint_pairs(self):
        clock = FakeClock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        robot = Mock()
        robot._parser = leader_pair_frames([0.2]*6, clock.now)
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0, motion_status=1)
        )
        def start_motion():
            # Ordinary J3-J6 feedback remains stale; fresh 155/156/157
            # drag input must still prove the native motion completed.
            stamp = clock.now + 0.01
            robot._parser.joint_12 = leader_pair_frames([0.0]*6, stamp).joint_12
            robot._parser.leader_joint_34 = SimpleNamespace(
                timestamp=stamp,
                msg=SimpleNamespace(joint_3=0.0, joint_4=0.0),
            )
            robot._parser.leader_joint_56 = SimpleNamespace(
                timestamp=stamp,
                msg=SimpleNamespace(joint_5=0.0, joint_6=0.0),
            )
        robot.move_leader_to_home.side_effect = start_motion
        def tick(seconds):
            clock.sleep(seconds)
            if robot.move_leader_to_home.called:
                for name in ('joint_12', 'leader_joint_34', 'leader_joint_56'):
                    getattr(robot._parser, name).timestamp = clock.now
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', tick), patch.object(
            lf.base, 'ensure_arm_status_normal'
        ), patch('builtins.print'):
            result = lf.home_leader_native(
                robot, config, 'can1', [0.2]*6, 20.0, lambda:False
            )
        self.assertEqual(result, [0.0]*6)
        robot.move_leader_to_home.assert_called_once()
        robot.move_j.assert_not_called()
        robot.set_speed_percent.assert_not_called()

    def test_native_home_uses_calibrated_motor_feedback_when_joint_frames_stop(self):
        clock = FakeClock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        robot = Mock()
        robot._parser = leader_pair_frames([0.2]*6, clock.now - 0.01)
        motor_states = [SimpleNamespace(
            timestamp=clock.now - 0.01,
            msg=SimpleNamespace(position=0.2),
        ) for _ in range(6)]
        moving = [False]
        def get_motor(index):
            state = motor_states[index - 1]
            if moving[0]:
                state.timestamp = clock.now
            return state
        robot.get_motor_states.side_effect = get_motor
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0, motion_status=0)
        )
        def start_motion():
            moving[0] = True
            for state in motor_states:
                state.msg.position = 0.0
        robot.move_leader_to_home.side_effect = start_motion
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', clock.sleep), patch.object(
            lf.base, 'ensure_arm_status_normal'
        ), patch('builtins.print') as output:
            result = lf.home_leader_native(
                robot, config, 'can1', [0.2]*6, 20.0, lambda:False
            )
        self.assertEqual(result, [0.0]*6)
        self.assertTrue(any('source=motor 0x251-0x256' in str(call)
                            for call in output.call_args_list))
        robot.move_leader_to_home.assert_called_once()

    def test_one_motor_zero_frame_then_feedback_stops_cannot_claim_home(self):
        clock = FakeClock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        robot = Mock()
        robot._parser = leader_pair_frames([0.2]*6, clock.now - 0.01)
        motor_states = [SimpleNamespace(
            timestamp=clock.now - 0.01,
            msg=SimpleNamespace(position=0.2),
        ) for _ in range(6)]
        robot.get_motor_states.side_effect = lambda index: motor_states[index - 1]
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0, motion_status=0)
        )
        def start_motion():
            for state in motor_states:
                state.timestamp = clock.now + 0.01
                state.msg.position = 0.0
        robot.move_leader_to_home.side_effect = start_motion
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', clock.sleep), patch.object(
            lf.base, 'ensure_arm_status_normal'
        ), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'no fresh position feedback'):
                lf.home_leader_native(
                    robot, config, 'can1', [0.2]*6, 20.0, lambda:False
                )

    def test_uncalibrated_motor_feedback_cannot_claim_home(self):
        clock = FakeClock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        robot = Mock()
        robot._parser = leader_pair_frames([0.2]*6, clock.now - 0.01)
        motor_states = [SimpleNamespace(
            timestamp=clock.now - 0.01,
            msg=SimpleNamespace(position=1.2),
        ) for _ in range(6)]
        robot.get_motor_states.side_effect = lambda index: motor_states[index - 1]
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0, motion_status=0)
        )
        def start_motion():
            for state in motor_states:
                state.timestamp = clock.now + 0.01
                state.msg.position = 0.0
        robot.move_leader_to_home.side_effect = start_motion
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', clock.sleep), patch.object(
            lf.base, 'ensure_arm_status_normal'
        ), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'no fresh position feedback'):
                lf.home_leader_native(
                    robot, config, 'can1', [0.2]*6, 20.0, lambda:False
                )

    def test_native_home_does_not_accept_precommand_cached_zero(self):
        clock = FakeClock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        robot = Mock()
        robot._parser = leader_pair_frames([0.0]*6, clock.now - 1.0)
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(arm_status=0, motion_status=1)
        )
        with patch.object(lf.time, 'monotonic', clock.time), patch.object(
            lf.time, 'time', clock.time
        ), patch.object(lf.time, 'sleep', clock.sleep), patch.object(
            lf.base, 'ensure_arm_status_normal'
        ), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'no fresh position feedback'):
                lf.home_leader_native(
                    robot, config, 'can1', [0.2]*6, 20.0, lambda:False
                )
        robot.move_leader_to_home.assert_called_once()

    def test_native_home_cancelled_before_send_does_not_send_command(self):
        robot = Mock()
        config = {'joint_limits': {f'joint{i}': [-3.0, 3.0] for i in range(1, 7)}}
        with patch.object(lf.base, 'ensure_arm_status_normal'):
            with self.assertRaises(InterruptedError):
                lf.home_leader_native(robot, config, 'can1', [0.2]*6,
                                      20.0, lambda:True)
        robot.move_leader_to_home.assert_not_called()

    def test_preflight_cancel_does_not_estop_unmoved_leader(self):
        follower, leader = Mock(), Mock()
        on_estop = Mock()
        with patch.object(lf, 'home_leader_native', side_effect=InterruptedError('before send')), patch(
            'builtins.print'
        ) as output:
            with self.assertRaises(InterruptedError):
                lf.home_both_sequentially(
                    follower, {}, 'can0', leader, {}, 'can1',
                    Mock(), Mock(), 1.0, 10.0, 60, lambda:True,
                    [0.2]*6, 0.02, on_estop=on_estop,
                )
        leader.electronic_emergency_stop.assert_not_called()
        follower.electronic_emergency_stop.assert_not_called()
        on_estop.assert_not_called()
        self.assertTrue(any('no E-stop sent' in str(call) for call in output.call_args_list))

    def run_keys(self, keys, *, control_only=False, leader_start=None):
        leader,follower = Mock(),Mock()
        leader_gripper,follower_gripper = Mock(),Mock()
        leader._parser = None; leader._ctx = None; follower._ctx = None
        follower.get_joint_angles.return_value = SimpleNamespace(msg=[0.0]*6)
        follower.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(arm_status=0))
        follower.fk.return_value = [0.0]*6
        leader.init_effector.return_value = leader_gripper
        follower.init_effector.return_value = follower_gripper
        recorder = Mock(); recorder.active = False
        recorder.start_episode.side_effect = lambda: setattr(recorder,'active',True)
        def save():
            recorder.active = False
            return Path('/tmp/fake-episode')
        recorder.save_episode.side_effect = save
        reader=Mock(); reader.__enter__=Mock(return_value=reader); reader.__exit__=Mock(return_value=False)
        reader.poll.side_effect = list(keys)
        config={'joint_limits':{f'joint{i}':[-3.,3.] for i in range(1,7)}}
        prefix='robotarm.teleop.leader_follower.runtime.'
        replacements={
            'sys.stdin.isatty':Mock(return_value=True),
            'base.connect_robot':Mock(side_effect=[(leader,config),(follower,config)]),
            'wait_initial_leader_joints':Mock(return_value=(leader_start or [0.0]*6,'feedback')),
            'wait_joint_message':Mock(return_value=([0.0]*6,0.0)),
            'validate_joint_feedback':Mock(),
            'ensure_startup_status':Mock(),
            'base.wait_gripper_width':Mock(return_value=.02),
            'base.get_all_joints_enabled_state':Mock(return_value=True),
            'base.ensure_arm_status_normal':Mock(),
            'arm_home.validate_joint_values':Mock(side_effect=lambda values, *_: values),
            'safe_joint_bounds':Mock(return_value=[(-1.,1.)]*6),
            'KeyReader':Mock(return_value=reader),
            'update_leader_joint_message':Mock(return_value=([0.0]*6,0.0)),
            'read_joint_message':Mock(side_effect=lambda _robot, _config, role: (
                leader_start if role == 'leader' and leader_start is not None else [0.0]*6,
                0.0,
            )),
            'read_gripper_message':Mock(return_value=(.02,0.0)),
            'read_leader_gripper_command':Mock(return_value=(.02,0.0)),
            'enter_follower_js':Mock(),
            'exit_leader_drag_mode':Mock(),
            'home_both_sequentially':Mock(return_value=[0.0]*6),
            'RawEpisodeRecorder':Mock(return_value=recorder),
            'time.monotonic':Mock(side_effect=(i*.005 for i in range(10000))),
            'time.sleep':Mock(),
            'base.disconnect_quietly':Mock(),
        }
        ordered=Mock()
        ordered.attach_mock(follower.move_js,'follow_move')
        ordered.attach_mock(replacements['exit_leader_drag_mode'],'exit_drag')
        ordered.attach_mock(replacements['home_both_sequentially'],'home_both')
        with ExitStack() as stack:
            for name,replacement in replacements.items():
                stack.enter_context(patch(prefix+name,replacement))
            output=stack.enter_context(patch('builtins.print'))
            arguments = ['--execute', '--allow-enabled-start']
            if control_only:
                arguments.append('--control-only')
            else:
                arguments.extend(['--setup-id', 'test', '--task', 'test collection'])
            result = lf.main(arguments)
        return result,leader,follower,recorder,replacements,ordered,output

    def test_saved_episode_then_h_uses_native_leader_home(self):
        result, leader, follower, recorder, replacements, ordered, output = self.run_keys(
            ['l', 'f', 'r', 'r', 'h', 'q']
        )
        self.assertEqual(result, 0)
        recorder.save_episode.assert_called_once()
        replacements['home_both_sequentially'].assert_called_once()
        self.assertEqual(
            replacements['home_both_sequentially'].call_args.args[2], 'can_follower1'
        )
        self.assertEqual(
            replacements['home_both_sequentially'].call_args.args[5], 'can_leader1'
        )
        self.assertEqual(replacements['home_both_sequentially'].call_args.args[-2], [0.0]*6)
        self.assertEqual(replacements['home_both_sequentially'].call_args.args[-1], .02)
        self.assertTrue(any('both arms homed' in str(call) for call in output.call_args_list))

    def test_h_is_available_immediately_and_after_save(self):
        for control_only in (False, True):
            result, leader, follower, recorder, replacements, ordered, _ = self.run_keys(
                ['h', 'q'], control_only=control_only
            )
            self.assertEqual(result, 0)
            leader.set_leader_mode.assert_not_called()
            replacements['home_both_sequentially'].assert_called_once()
            if control_only:
                replacements['RawEpisodeRecorder'].assert_not_called()
            else:
                recorder.start_episode.assert_not_called()

        result, leader, _, _, replacements, _, _ = self.run_keys(
            ['h', 'q'], leader_start=[0.2]*6
        )
        self.assertEqual(result, 0)
        leader.set_leader_mode.assert_called_once()
        replacements['home_both_sequentially'].assert_called_once()

        result, _, _, _, replacements, _, _ = self.run_keys(
            ['l', 'f', 'r', 'r', 'h', 'h', 'q']
        )
        self.assertEqual(result, 0)
        self.assertEqual(replacements['home_both_sequentially'].call_count, 2)

    def test_h_after_drag_before_recording_uses_native_home(self):
        result, _, _, recorder, replacements, _, _ = self.run_keys(['l', 'h', 'q'])
        self.assertEqual(result, 0)
        recorder.start_episode.assert_not_called()
        replacements['home_both_sequentially'].assert_called_once()

    def test_follower_failure_estops_only_follower_after_leader_hold(self):
        follower, leader = Mock(), Mock()
        with patch.object(lf, 'home_leader_native', return_value=[0.0]*6), patch.object(
            lf, 'exit_leader_drag_mode'
        ), patch.object(lf, 'close_gripper_slowly'), patch.object(
            lf, 'home_one_arm', side_effect=RuntimeError('follower fault')
        ):
            with self.assertRaisesRegex(RuntimeError, 'follower fault'):
                lf.home_both_sequentially(
                    follower, {}, 'can0', leader, {}, 'can1',
                    Mock(), Mock(), 1.0, 10.0, 60, lambda:False,
                    [0.2]*6, 0.02,
                )
        follower.electronic_emergency_stop.assert_called_once()
        leader.electronic_emergency_stop.assert_not_called()

    def test_home_one_arm_uses_slow_joint_mode_with_feedback_guard(self):
        robot=Mock(); gripper=Mock(); config={}
        with patch.object(lf.arm_home,'run_home') as run:
            lf.home_one_arm(robot,config,'can0',gripper,1.0,12.0,75,lambda:False)
        self.assertIs(run.call_args.args[0],robot)
        self.assertEqual(run.call_args.args[2].speed_percent,10)
        self.assertEqual(run.call_args.args[2].timeout,75)
        self.assertEqual(run.call_args.args[2].tolerance_deg,3)
        self.assertEqual(run.call_args.args[2].joint_limit_tolerance_deg,5)
        self.assertEqual(run.call_args.args[2].stale_timeout,.5)
        self.assertIs(run.call_args.kwargs['gripper'],gripper)
        self.assertEqual(run.call_args.kwargs['gripper_force'],1.0)
        self.assertEqual(run.call_args.kwargs['gripper_speed_mm_s'],12.0)

    def test_keyboard_interrupt_estops_active_leader_with_clear_message(self):
        follower, leader = Mock(), Mock()
        def interrupt_after_send(*_args, **kwargs):
            kwargs['on_command_start']()
            raise KeyboardInterrupt
        with patch.object(lf, 'home_leader_native', side_effect=interrupt_after_send), patch(
            'builtins.print'
        ) as output:
            with self.assertRaises(KeyboardInterrupt):
                lf.home_both_sequentially(
                    follower, {}, 'can0', leader, {}, 'can1',
                    Mock(), Mock(), 1.0, 10.0, 60, lambda:False,
                    [0.2]*6, 0.02,
                )
        leader.electronic_emergency_stop.assert_called_once()
        follower.electronic_emergency_stop.assert_not_called()
        self.assertTrue(any('interrupted by operator' in str(call)
                            for call in output.call_args_list))

    def test_h_while_recording_does_not_home(self):
        result, _, _, recorder, replacements, _, _ = self.run_keys(
            ['l', 'f', 'r', 'h', 'q']
        )
        self.assertEqual(result, 0)
        recorder.start_episode.assert_called_once()
        replacements['home_both_sequentially'].assert_not_called()

    def test_options_are_bounded(self):
        args=lf.make_parser().parse_args([])
        self.assertEqual(args.home_timeout,60)
        self.assertEqual(args.home_gripper_speed_mm_s,10)
        args.home_timeout=math.nan
        with self.assertRaisesRegex(ValueError,'numeric arguments'):
            lf.validate_args(args)
        args.home_timeout=181
        with self.assertRaisesRegex(ValueError,'home-timeout'):
            lf.validate_args(args)
