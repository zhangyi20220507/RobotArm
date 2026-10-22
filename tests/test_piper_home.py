"""Standalone homing tests using fake CAN and simulated clocks only."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from robotarm.hardware import home

CONFIG = {'joint_limits':{f'joint{i}':[-3,3] for i in range(1,7)}}


class Clock:
    def __init__(self): self.now = 1000.
    def time(self): return self.now
    def sleep(self,seconds): self.now += seconds


class HomeTests(unittest.TestCase):
    def args(self,*extra):
        return home.make_parser().parse_args(['--channel','can1',*extra])

    def test_preview_never_loads_sdk_or_connects(self):
        with patch.object(home,'connect_selected_arm') as connect:
            self.assertEqual(home.main(['--channel','can1']),0)
            connect.assert_not_called()

    def test_channel_is_required_and_options_finite(self):
        self.assertEqual(self.args().joint_limit_tolerance_deg, 5.0)
        for flag,value in (('--timeout','nan'),('--stale-timeout','0'),('--tolerance-deg','4'),
                           ('--joint-limit-tolerance-deg','nan'),('--joint-limit-tolerance-deg','6')):
            with self.assertRaises(ValueError): home.validate_args(self.args(flag,value))
        with self.assertRaises(SystemExit): home.make_parser().parse_args([])

    def simulate(self, complete=True, stop=None, timeout=4):
        clock = Clock()
        robot = Mock(_parser=None)
        robot.OPTIONS.MOTION_MODE.J = 'J'
        q = [.2]*6
        def move(values):
            if complete: q[:] = values
        robot.move_j.side_effect = move
        robot.get_joint_angles.side_effect = lambda: SimpleNamespace(msg=q.copy(),timestamp=clock.now)
        robot.get_joints_enable_status_list.return_value = [True]*6
        robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(arm_status=0,motion_status=0))
        robot.mock_clock = clock
        with patch.object(home.time,'monotonic',clock.time), patch.object(home.time,'time',clock.time), patch.object(home.time,'sleep',clock.sleep):
            home.run_home(robot,CONFIG,self.args('--timeout',str(timeout)),stop or (lambda:False))
        return robot

    def test_enable_holds_then_zeroes_and_remains_enabled_without_gripper(self):
        robot = self.simulate()
        calls = robot.mock_calls
        names = [c[0] for c in calls]
        self.assertLess(names.index('set_speed_percent'),names.index('set_motion_mode'))
        self.assertLess(names.index('move_j'),names.index('enable'))
        moves = robot.move_j.call_args_list
        self.assertEqual(moves[0].args[0],[.2]*6)
        self.assertEqual(moves[1].args[0],[0.0]*6)
        robot.disable.assert_not_called()
        robot.init_effector.assert_not_called()
        robot.electronic_emergency_stop.assert_not_called()

    def test_timeout_and_cancel_do_not_report_success(self):
        with self.assertRaises(TimeoutError): self.simulate(complete=False,timeout=1)
        with self.assertRaises(InterruptedError): self.simulate(stop=lambda:True)

    def test_stationary_arm_fails_early_with_controller_state(self):
        clock = Clock()
        robot = Mock(_parser=None)
        robot.OPTIONS.MOTION_MODE.J = 'J'
        joints = [.2] * 6
        robot.get_joint_angles.side_effect = lambda: SimpleNamespace(
            msg=joints.copy(), timestamp=clock.now
        )
        robot.get_joints_enable_status_list.return_value = [True] * 6
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(
                arm_status=0, motion_status=1, ctrl_mode=0, mode_feedback=0
            )
        )
        with patch.object(home.time, 'monotonic', clock.time), patch.object(
            home.time, 'time', clock.time
        ), patch.object(home.time, 'sleep', clock.sleep):
            with self.assertRaisesRegex(RuntimeError, 'homing made no meaningful progress for 8 s.*ctrl_mode=0'):
                home.run_home(
                    robot, CONFIG, self.args('--timeout', '20'), lambda: False
                )
        robot.disable.assert_not_called()

    def test_small_joint_jitter_does_not_hide_lack_of_homing_progress(self):
        import math
        clock = Clock()
        robot = Mock(_parser=None)
        robot.OPTIONS.MOTION_MODE.J = 'J'
        samples = [math.radians(17.5), math.radians(17.2), math.radians(17.9)]
        reads = [0]

        def joints():
            value = samples[reads[0] % len(samples)]
            reads[0] += 1
            return SimpleNamespace(
                msg=[0.0, 0.0, 0.0, 0.0, value, 0.0],
                timestamp=clock.now,
            )

        robot.get_joint_angles.side_effect = joints
        robot.get_joints_enable_status_list.return_value = [True] * 6
        robot.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(
                arm_status=0, motion_status=0, ctrl_mode=1, mode_feedback=1
            )
        )
        with patch.object(home.time, 'monotonic', clock.time), patch.object(
            home.time, 'time', clock.time
        ), patch.object(home.time, 'sleep', clock.sleep):
            with self.assertRaisesRegex(
                RuntimeError, 'homing made no meaningful progress for 8 s'
            ):
                home.run_home(
                    robot, CONFIG, self.args('--timeout', '30'), lambda: False
                )
        self.assertLess(clock.now - 1000.0, 12.0)

    def test_stale_and_invalid_feedback_rejected(self):
        robot = Mock(_parser=None)
        for values,stamp in (([0]*6,990),([float('nan')]*6,1000),([4]*6,1000)):
            robot.get_joint_angles.return_value = SimpleNamespace(msg=values,timestamp=stamp)
            with patch.object(home.time,'time',return_value=1000):
                with self.assertRaises(RuntimeError): home.read_joints(robot,CONFIG,.5)

    def test_waits_for_all_joint_pairs_before_accepting_startup_pose(self):
        clock = Clock()
        parser = SimpleNamespace(
            joint_12=SimpleNamespace(timestamp=clock.now),
            joint_34=None,
            joint_56=None,
        )
        robot = Mock(_parser=parser)
        robot.get_joint_angles.side_effect = lambda: SimpleNamespace(
            msg=[0.1] * 6, timestamp=clock.now
        )

        def arrive(seconds):
            clock.sleep(seconds)
            if clock.now >= 1000.04:
                parser.joint_34 = SimpleNamespace(timestamp=clock.now)
                parser.joint_56 = SimpleNamespace(timestamp=clock.now)

        with patch.object(home.time, 'monotonic', clock.time), patch.object(
            home.time, 'time', clock.time
        ), patch.object(home.time, 'sleep', arrive):
            joints = home.wait_complete_joints(robot, CONFIG, 1.0, 0.5)
        self.assertEqual(joints, [0.1] * 6)
        self.assertGreaterEqual(robot.get_joint_angles.call_count, 2)
        robot.move_j.assert_not_called()

    def test_incomplete_joint_pairs_timeout_without_control(self):
        clock = Clock()
        robot = Mock(_parser=SimpleNamespace(
            joint_12=SimpleNamespace(timestamp=clock.now),
            joint_34=None,
            joint_56=None,
        ))
        robot.get_joint_angles.side_effect = lambda: SimpleNamespace(
            msg=[0.0] * 6, timestamp=clock.now
        )
        with patch.object(home, 'connect_selected_arm', return_value=(robot, CONFIG)), patch.object(
            home.base, 'wait_for_feedback'
        ), patch.object(home.time, 'monotonic', clock.time), patch.object(
            home.time, 'time', clock.time
        ), patch.object(home.time, 'sleep', clock.sleep):
            self.assertEqual(home.main([
                '--channel', 'can1', '--feedback-timeout', '0.1', '--execute'
            ]), 1)
        robot.move_j.assert_not_called()
        robot.enable.assert_not_called()
        robot.electronic_emergency_stop.assert_not_called()

    def test_error_and_interrupt_after_control_estop_and_disconnect(self):
        for error,code in ((TimeoutError('timeout'),1),(InterruptedError('cancel'),130)):
            robot = Mock(_parser=None)
            with patch.object(home,'connect_selected_arm',return_value=(robot,CONFIG)), patch.object(home.base,'wait_for_feedback'), patch.object(home,'wait_complete_joints',return_value=[0]*6), patch.object(home.base,'ensure_arm_status_normal'), patch.object(home,'run_home',side_effect=error):
                self.assertEqual(home.main(['--channel','can1','--execute']),code)
            robot.electronic_emergency_stop.assert_called_once()
            robot.disconnect.assert_called_once()

    def test_initial_invalid_feedback_does_not_control(self):
        robot = Mock(_parser=None)
        with patch.object(home,'connect_selected_arm',return_value=(robot,CONFIG)),patch.object(home.base,'wait_for_feedback'),patch.object(home,'wait_complete_joints',side_effect=RuntimeError('stale')):
            self.assertEqual(home.main(['--channel','can1','--execute']),1)
        robot.move_j.assert_not_called()
        robot.enable.assert_not_called()
        robot.electronic_emergency_stop.assert_not_called()
        robot.disconnect.assert_called_once()

    def test_existing_emergency_stop_is_rejected_before_any_control(self):
        robot = Mock(_parser=None)
        with patch.object(home, 'connect_selected_arm', return_value=(robot, CONFIG)), patch.object(
            home.base, 'wait_for_feedback'
        ), patch.object(home, 'wait_complete_joints', return_value=[0.0] * 6), patch.object(
            home.base, 'ensure_arm_status_normal',
            side_effect=RuntimeError('Arm status is not normal (code=1)')
        ), patch.object(home, 'run_home') as run:
            self.assertEqual(
                home.main(['--channel', 'can1', '--firmware', 'v189', '--execute']),
                1,
            )
        run.assert_not_called()
        robot.enable.assert_not_called()
        robot.move_j.assert_not_called()
        robot.electronic_emergency_stop.assert_not_called()

    def test_autodetect_uses_selected_channel_and_numeric_version(self):
        probe,robot = Mock(_parser=None),Mock()
        probe.get_firmware.return_value = {'software_version':'S-V1.10-0'}
        factory = Mock(); factory.create_arm.side_effect = [probe,robot]
        firmware = SimpleNamespace(DEFAULT='default',V183='183',V188='188',V189='189')
        creator = Mock(return_value=CONFIG)
        with patch.object(home.base,'_load_sdk',return_value=(creator,factory,SimpleNamespace(PIPER='piper'),firmware)):
            result,_ = home.connect_selected_arm(self.args())
        self.assertIs(result,robot)
        self.assertTrue(all(c.kwargs['channel']=='can1' for c in creator.call_args_list))
        self.assertEqual(creator.call_args.kwargs['firmeware_version'],'189')
        probe.disconnect.assert_called_once()
        robot.move_j.assert_not_called()

    def boundary_config(self):
        config = {'joint_limits':{key:value.copy() for key,value in CONFIG['joint_limits'].items()}}
        config['joint_limits']['joint2'] = [0,3.141593]
        config['joint_limits']['joint3'] = [-2.967060,0]
        return config

    def test_zero_boundary_feedback_keeps_actual_angles_but_holds_within_limits(self):
        import math
        config = self.boundary_config()
        joints = [0,math.radians(-1.814),math.radians(2.253),0,0,0]
        robot = Mock(_parser=None)
        robot.get_joint_angles.return_value = SimpleNamespace(msg=joints,timestamp=1000)
        with patch.object(home.time,'time',return_value=1000):
            measured = home.read_joints(robot,config,.5)
        self.assertEqual(measured,joints)
        command = home.bounded_hold(measured,config)
        self.assertEqual(command,[0]*6)
        self.assertGreater(math.degrees(max(abs(v) for v in measured)),2)

    def test_real_limit_violation_still_fails_with_measured_value_and_range(self):
        import math
        robot = Mock(_parser=None)
        robot.get_joint_angles.return_value = SimpleNamespace(msg=[0,0,math.radians(5.5),0,0,0],timestamp=1000)
        with patch.object(home.time,'time',return_value=1000):
            with self.assertRaisesRegex(RuntimeError, r'J3 feedback=\+5.500 deg.*5 deg feedback tolerance'):
                home.read_joints(robot,self.boundary_config(),.5)
            with self.assertRaises(RuntimeError):
                home.read_joints(robot,self.boundary_config(),.5,0)

    def test_default_accepts_small_j5_overrun_without_relaxing_command_limit(self):
        import math
        config = self.boundary_config()
        config['joint_limits']['joint5'] = [-math.radians(70), math.radians(70)]
        robot = Mock(_parser=None)
        robot.get_joint_angles.return_value = SimpleNamespace(
            msg=[0,0,0,0,math.radians(73.596),0], timestamp=1000
        )
        with patch.object(home.time,'time',return_value=1000):
            measured = home.read_joints(robot,config,.5)
        self.assertAlmostEqual(math.degrees(measured[4]),73.596)
        self.assertAlmostEqual(math.degrees(home.bounded_hold(measured,config)[4]),70)

    def test_feedback_near_zero_does_not_abort_home(self):
        import math
        clock = Clock(); robot = Mock(_parser=None); config = self.boundary_config()
        joints = [0,math.radians(-1.53),math.radians(1.32),0,0,0]
        robot.OPTIONS.MOTION_MODE.J = 'J'
        robot.get_joint_angles.side_effect = lambda: SimpleNamespace(msg=joints.copy(),timestamp=clock.now)
        robot.get_joints_enable_status_list.return_value = [True]*6
        robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(arm_status=0,motion_status=0))
        with patch.object(home.time,'monotonic',clock.time),patch.object(home.time,'time',clock.time),patch.object(home.time,'sleep',clock.sleep):
            home.run_home(robot,config,self.args(),lambda:False)
        self.assertEqual(robot.move_j.call_count,3)
        self.assertTrue(all(call.args[0]==[0]*6 for call in robot.move_j.call_args_list))

    def test_outside_completion_tolerance_cannot_be_hidden_by_clamping(self):
        import math
        clock = Clock(); robot = Mock(_parser=None); config = self.boundary_config()
        robot.OPTIONS.MOTION_MODE.J = 'J'
        robot.get_joint_angles.side_effect = lambda: SimpleNamespace(msg=[0,0,math.radians(2.253),0,0,0],timestamp=clock.now)
        robot.get_joints_enable_status_list.return_value = [True]*6
        robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(arm_status=0,motion_status=0))
        with patch.object(home.time,'monotonic',clock.time),patch.object(home.time,'time',clock.time),patch.object(home.time,'sleep',clock.sleep):
            with self.assertRaises(TimeoutError):
                home.run_home(robot,config,self.args('--timeout','2'),lambda:False)
        self.assertEqual(robot.move_j.call_count,2)

    def test_joint_and_gripper_home_waits_for_both(self):
        clock=Clock(); robot=Mock(_parser=None); gripper=Mock()
        robot.OPTIONS.MOTION_MODE.J='J'
        joints=[.2]*6; width=[.02]
        robot.move_j.side_effect=lambda target: joints.__setitem__(slice(None), target)
        robot.get_joint_angles.side_effect=lambda: SimpleNamespace(msg=joints.copy(),timestamp=clock.now)
        robot.get_joints_enable_status_list.return_value=[True]*6
        robot.get_arm_status.return_value=SimpleNamespace(msg=SimpleNamespace(arm_status=0,motion_status=0))
        gripper.get_gripper_status.side_effect=lambda: SimpleNamespace(
            msg=SimpleNamespace(mode='width',value=width[0]),timestamp=clock.now)
        gripper.move_gripper_m.side_effect=lambda command,force: width.__setitem__(0,command)
        with patch.object(home.time,'monotonic',clock.time),patch.object(home.time,'time',clock.time),patch.object(home.time,'sleep',clock.sleep):
            home.run_home(robot,CONFIG,self.args('--timeout','5'),lambda:False,
                          gripper=gripper,gripper_force=1.0,gripper_speed_mm_s=10.)
        widths=[call.args[0] for call in gripper.move_gripper_m.call_args_list]
        self.assertGreater(len(widths),50)
        self.assertAlmostEqual(widths[-1],0.0)
        self.assertAlmostEqual(width[0],0.0)
        self.assertGreaterEqual(clock.now,1002.)
        self.assertTrue(all(a>=b for a,b in zip(widths,widths[1:])))
        self.assertTrue(all(a-b<=.00021 for a,b in zip(widths,widths[1:])))
        robot.electronic_emergency_stop.assert_not_called()

    def test_joint_zero_without_gripper_zero_times_out(self):
        clock=Clock(); robot=Mock(_parser=None); gripper=Mock()
        robot.OPTIONS.MOTION_MODE.J='J'
        robot.get_joint_angles.side_effect=lambda: SimpleNamespace(msg=[0.]*6,timestamp=clock.now)
        robot.get_joints_enable_status_list.return_value=[True]*6
        robot.get_arm_status.return_value=SimpleNamespace(msg=SimpleNamespace(arm_status=0,motion_status=0))
        gripper.get_gripper_status.side_effect=lambda: SimpleNamespace(
            msg=SimpleNamespace(mode='width',value=.02),timestamp=clock.now)
        with patch.object(home.time,'monotonic',clock.time),patch.object(home.time,'time',clock.time),patch.object(home.time,'sleep',clock.sleep):
            with self.assertRaises(TimeoutError):
                home.run_home(robot,CONFIG,self.args('--timeout','2'),lambda:False,
                              gripper=gripper,gripper_force=1.,gripper_speed_mm_s=10.)
        self.assertTrue(gripper.move_gripper_m.called)

    def test_stale_gripper_feedback_prevents_any_motion(self):
        clock=Clock(); robot=Mock(_parser=None); gripper=Mock()
        robot.get_joint_angles.return_value=SimpleNamespace(msg=[0.]*6,timestamp=clock.now)
        gripper.get_gripper_status.return_value=SimpleNamespace(
            msg=SimpleNamespace(mode='width',value=.02),timestamp=clock.now-1)
        with patch.object(home.time,'monotonic',clock.time),patch.object(home.time,'time',clock.time):
            with self.assertRaisesRegex(RuntimeError,'Gripper feedback is stale'):
                home.run_home(robot,CONFIG,self.args(),lambda:False,gripper=gripper)
        robot.move_j.assert_not_called()
        gripper.move_gripper_m.assert_not_called()
