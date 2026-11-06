"""Offline direct-control and controller-discontinuity regression tests."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from robotarm.teleop.xr.control import Clutch, PoseJumpGuard
from robotarm.teleop.xr.geometry import Limits, build_target
from robotarm.teleop.xr.piper_ik import PiperIK
from robotarm.teleop.xr.runtime import make_parser, validate_args


class DirectTests(unittest.TestCase):
    def test_no_grip_and_explicit_recovery_after_stale(self):
        c = Clutch(.9, .7, mode='direct')
        self.assertFalse(c.update(0, True))
        c.resume()
        self.assertTrue(c.update(0, True))
        self.assertTrue(c.update(1, True))
        self.assertFalse(c.update(0, False))
        self.assertFalse(c.update(0, True))
        c.resume()
        self.assertTrue(c.update(0, True))

    def test_jump_latches_even_in_grip_mode(self):
        for mode in ('direct', 'grip'):
            c = Clutch(.9, .7, mode=mode)
            c.resume(); c.pause()
            self.assertFalse(c.update(0, True))
            self.assertFalse(c.update(1, True))
            c.resume()
            self.assertTrue(c.update(1, True))

    def test_quaternion_sign_and_small_movements_are_not_jumps(self):
        g = PoseJumpGuard(.05,30)
        p = np.array([0.,0.,0.,0.,0.,0.,1.])
        self.assertIsNone(g.update(p))
        p[3:] *= -1
        self.assertIsNone(g.update(p))
        p[0] = .004
        p[3:] = Rotation.from_euler('x', 5, degrees=True).as_quat()
        self.assertIsNone(g.update(p))

    def test_translation_rotation_and_invalid_pose_detected(self):
        for displacement, angle in ((.051,0), (0,31)):
            g = PoseJumpGuard(.05,30)
            g.update([0,0,0,0,0,0,1])
            q = Rotation.from_euler('y',angle,degrees=True).as_quat()
            self.assertIn('controller jump', g.update(np.r_[displacement,0,0,q]))
        self.assertIn('invalid',g.update([0]*7))

    def test_input_options_validated(self):
        self.assertEqual(make_parser().parse_args([]).engagement_mode,'grip')
        self.assertEqual(make_parser().parse_args([]).hand_jump_action,'rebase')
        self.assertEqual(
            make_parser().parse_args(['--engagement-mode','direct']).engagement_mode,'direct')
        self.assertEqual(make_parser().parse_args([]).xr_mirror,'none')
        self.assertEqual(
            make_parser().parse_args(['--xr-mirror','lr']).xr_mirror,'lr')
        for flag, value in (('--max-hand-jump-m','nan'),('--max-hand-jump-deg','0')):
            with self.assertRaises(ValueError):
                validate_args(make_parser().parse_args([flag,value]),np)

    def test_single_axis_mirror_flips_only_that_axis(self):
        from robotarm.teleop.xr.geometry import build_target, mapping_matrix
        limits = Limits(np.array([-1.,-1.,-1.]),np.array([1.,1.,1.]),1.,.01,.1)
        ref = np.array([0.,0.,0.,0.,0.,0.,1.])
        flange = np.zeros(6)
        # hand right (+X), hand up (+Y), hand back (+Z) -> robot (+X forward, +Y left, +Z up)
        for hand, plain, lr, fb in (
            ([.1,0,0], [0.,-.1,0.], [0.,.1,0.], [0.,-.1,0.]),
            ([0,.1,0], [0.,0.,.1], [0.,0.,.1], [0.,0.,.1]),
            ([0,0,.1], [-.1,0.,0.], [-.1,0.,0.], [.1,0.,0.]),
        ):
            current = ref.copy(); current[:3] = hand
            for mirror, expected in (('none',plain),('lr',lr),('fb',fb)):
                target = build_target(ref,current,flange,1.,False,limits,mirror=mirror)
                np.testing.assert_allclose(target[:3],expected,atol=1e-9)
        with self.assertRaises(ValueError):
            mapping_matrix(0.,'ud')

    def test_posture_anchor_stops_null_space_drift(self):
        """Repeated moves must not ratchet the arm into unrelated postures."""
        start = np.array([.0674,.0687,.0099,-.1118,.1155,.1113])
        steps = list(range(1,31)) + list(range(29,-1,-1))

        def drift(anchor):
            ik = PiperIK(.02, False, 1e-5)
            ik.set_posture_anchor(start if anchor else None)
            seed = start.copy()
            reference = ik.flange_pose(seed)
            cycles = []
            for _ in range(2):
                for distance in steps:
                    target = reference.copy()
                    target[1] += distance * .001
                    seed = ik.solve_adaptive(seed, target).joints
                achieved = ik.flange_pose(seed)
                cycles.append(float(np.max(np.abs(seed-start))))
            return cycles

        anchored = drift(True)
        free = drift(False)
        self.assertLess(max(anchored), np.deg2rad(.2))
        self.assertLess(anchored[-1], free[-1])
        self.assertLess(free[0], free[-1])

    def test_bimanual_mirror_flags(self):
        from robotarm.teleop.xr.bimanual_runtime import (
            make_parser as bimanual_parser, validate_args as bimanual_validate)
        args = bimanual_parser().parse_args([])
        self.assertEqual((args.left_xr_mirror,args.right_xr_mirror),('none','none'))
        args = bimanual_parser().parse_args(
            ['--left-xr-mirror','lr','--right-xr-mirror','fb'])
        self.assertEqual((args.left_xr_mirror,args.right_xr_mirror),('lr','fb'))
        bimanual_validate(args)
        args.left_xr_mirror = 'ud'
        with self.assertRaises(ValueError):
            bimanual_validate(args)

    def test_wrist_round_trip_reuses_fixed_orientation_reference(self):
        limits = Limits(np.array([.1,-.3,.05]),np.array([.55,.3,.55]),.2,.0024,.02)
        ik = PiperIK(.02, True)
        initial = np.array([.1,.8,-1.2,.1,.5,.1])
        flange = ik.flange_pose(initial)
        ref = np.array([0.,0.,0.,0.,0.,0.,1.])
        current = initial.copy()
        for angle in list(np.linspace(0,8,30)) + list(np.linspace(8,0,30)):
            pose = np.r_[ref[:3], Rotation.from_euler('z',angle,degrees=True).as_quat()]
            target = build_target(ref,pose,flange,1.,True,limits)
            current = ik.solve_adaptive(current,target).joints
        actual = ik.flange_pose(current)
        self.assertLess(np.linalg.norm(actual[:3]-flange[:3]),.0008)
        self.assertLess(ik._orientation_error(actual[3:],flange[3:]),np.deg2rad(1))
        self.assertLess(np.max(np.abs(current-initial)),.03)

    def test_one_hand_jump_holds_both_and_automatically_rebases(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        from robotarm.teleop.xr.bimanual_runtime import pause_on_input_jump
        sides = [Mock(name='left'), Mock(name='right')]
        guards = {}
        state = {}
        for side, hand in zip(sides, ('left','right')):
            side.hand = hand
            side.clutch = Clutch(.9,.7,mode='direct',blocked=False)
            guards[hand] = PoseJumpGuard(.05,30)
            state[f'{hand}_controller_pose'] = [0,0,0,0,0,0,1]
        with patch('robotarm.teleop.xr.bimanual_runtime._hold') as hold:
            self.assertFalse(pause_on_input_jump(sides,state,guards,SimpleNamespace()))
            hold.assert_not_called()
            state['left_controller_pose'] = [.1,0,0,0,0,0,1]
            self.assertTrue(pause_on_input_jump(sides,state,guards,SimpleNamespace()))
            self.assertEqual(hold.call_count,2)
            for side in sides:
                side.reset_clutch.assert_called_once()
                self.assertTrue(side.clutch.update(0,True))

    def test_strict_jump_mode_still_latches_both_arms(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        from robotarm.teleop.xr.bimanual_runtime import pause_on_input_jump
        sides = [Mock(name='left'), Mock(name='right')]
        guards, state = {}, {}
        for side, hand in zip(sides, ('left','right')):
            side.hand = hand
            side.clutch = Clutch(.9,.7,mode='direct',blocked=False)
            guards[hand] = PoseJumpGuard(.05,30)
            state[f'{hand}_controller_pose'] = [0,0,0,0,0,0,1]
        args = SimpleNamespace(hand_jump_action='pause')
        with patch('robotarm.teleop.xr.bimanual_runtime._hold'):
            self.assertFalse(pause_on_input_jump(sides,state,guards,args))
            state['left_controller_pose'] = [.1,0,0,0,0,0,1]
            self.assertTrue(pause_on_input_jump(sides,state,guards,args))
        for side in sides:
            self.assertFalse(side.clutch.update(0,True))

    def test_paused_arm_does_not_update_gripper_or_solve_ik(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        from robotarm.teleop.xr.bimanual_runtime import ArmSide, _compute_arm
        c = Clutch(.9,.7,mode='direct'); c.pause()
        side = ArmSide('left','left','can0',Mock(),Mock(),None,0,np.zeros(6),c,
                       gripper=Mock(),ready=True,enabled=True)
        state = {'left_controller_pose':[0,0,0,0,0,0,1],
                 'right_controller_pose':[0,0,0,0,0,0,1],
                 'left_grip':0,'left_trigger':1}
        self.assertIsNone(_compute_arm(side,state,True,.02,1,SimpleNamespace(live=True)))
        self.assertEqual(side.gripper.mock_calls,[])
        self.assertEqual(side.ik.mock_calls,[])

    def test_bimanual_engage_frame_only_captures_reference(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        from robotarm.teleop.xr.bimanual_runtime import ArmSide, _compute_arm
        from robotarm.teleop.xr.geometry import Limits

        joints = np.zeros(6)
        flange = np.array([0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
        ik = Mock()
        ik.flange_pose.return_value = flange.copy()
        side = ArmSide(
            "right", "right", "can1", Mock(), ik,
            Limits(np.array([0.1, -0.3, 0.05]), np.array([0.55, 0.3, 0.55]),
                   0.2, 0.0024, np.deg2rad(1.2)),
            0.0, joints.copy(), Clutch(0.9, 0.7, blocked=False, mode="grip"),
            ready=True, enabled=True,
        )
        state = {
            "headset_pose": [0, 0, 0, 0, 0, 0, 1],
            "left_controller_pose": [0, 0, 0, 0, 0, 0, 1],
            "right_controller_pose": [0, 0, 0, 0, 0, 0, 1],
            "right_grip": 1.0,
            "right_trigger": 0.0,
            "A_button": False,
            "B_button": False,
        }
        args = SimpleNamespace(
            live=False, command_modifier_threshold=0.8, posture_anchor="engage",
            scale=2.0, max_target_lead=0.02, orientation=False,
            tracking_frame="head", engage_settle_seconds=0.25,
            engage_deadband_m=0.002,
        )
        with patch(
            "robotarm.teleop.xr.bimanual_runtime.wait_for_joint_angles",
            return_value=joints,
        ), patch("builtins.print"):
            self.assertIsNone(_compute_arm(side, state, True, 0.02, 1.0, args))
        ik.solve_adaptive.assert_not_called()
        np.testing.assert_allclose(side.xr_reference, state["right_controller_pose"])
        np.testing.assert_allclose(side.flange_reference, flange)

        # Motion caused while squeezing Grip is continuously rebased.
        state["right_controller_pose"][0] = 0.0015
        with patch(
            "robotarm.teleop.xr.bimanual_runtime.wait_for_joint_angles",
            return_value=joints,
        ):
            self.assertIsNone(_compute_arm(side, state, True, 0.02, 1.1, args))
        np.testing.assert_allclose(side.xr_reference[:3], [0.0015, 0, 0])
        self.assertFalse(side.motion_armed)

        # After settling, sub-threshold jitter still sends no command. Crossing
        # the threshold only arms/rebases; it does not move the robot either.
        with patch(
            "robotarm.teleop.xr.bimanual_runtime.wait_for_joint_angles",
            return_value=joints,
        ):
            state["right_controller_pose"][0] = 0.0025
            self.assertIsNone(_compute_arm(side, state, True, 0.02, 1.3, args))
            state["right_controller_pose"][0] = 0.004
            self.assertIsNone(_compute_arm(side, state, True, 0.02, 1.32, args))
        self.assertTrue(side.motion_armed)
        ik.solve_adaptive.assert_not_called()
