"""Offline XR control regression checks; never connect to a robot."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from robotarm.teleop.xr.geometry import Limits, build_target, clamp_rotation_step
from robotarm.teleop.xr.piper_ik import PiperIK
from robotarm.teleop.xr.runtime import wait_for_xr, make_parser, validate_args


class PicoControlTests(unittest.TestCase):
    def tracker(self, lead=.02):
        from robotarm.teleop.xr.geometry import TranslationTracker
        limits = Limits(np.array([.1,-.3,.05]),np.array([.55,.3,.55]),.2,.0024,.014)
        return TranslationTracker(np.zeros(3),np.array([.3,0.,.3]),1.,limits,lead)

    def test_large_hand_motion_does_not_queue_long_chase(self):
        t = self.tracker(); feedback = np.array([.3,0.,.3])
        hand = np.array([0.,0.,-.5])
        target,reasons = t.update(hand,feedback)
        self.assertAlmostEqual(np.linalg.norm(target-feedback),.02)
        self.assertTrue(any('FOLLOW_LIMIT' in x for x in reasons))
        # Once feedback catches up, a stationary hand must not reveal old debt.
        for _ in range(100):
            next_target,_ = t.update(hand,target)
            np.testing.assert_allclose(next_target,target)

    def test_workspace_saturation_has_no_reverse_dead_zone(self):
        t=self.tracker(1.)
        target,reasons=t.update(np.array([0.,1.,0.]),np.array([.3,0.,.3]))
        self.assertTrue(any('RELATIVE_LIMIT' in x for x in reasons))
        target2,_=t.update(np.array([0.,.99,0.]),target)
        self.assertAlmostEqual(target2[2],target[2]-.01)

    def test_incremental_small_motion_remains_one_to_one(self):
        t=self.tracker(); feedback=np.array([.3,0.,.3])
        for i in range(1,11):
            target,reasons=t.update(np.array([0.,0.,-.001*i]),feedback)
            self.assertFalse(reasons)
            feedback=target
        np.testing.assert_allclose(target,[.31,0.,.3])

    def test_workspace_notice_and_fixed_clutch_boundary(self):
        t=self.tracker(1.)
        target,reasons=t.update(np.array([0.,-.5,0.]),np.array([.3,0.,.3]))
        # Relative sphere still limits travel after any number of updates.
        for i in range(10):target,reasons=t.update(np.array([0.,-.6-i,0.]),target)
        self.assertLessEqual(np.linalg.norm(target-t.reference),.200001)
        t=self.tracker(1.); t.maximum[0]=.305
        target,reasons=t.update(np.array([0.,0.,-.1]),t.reference)
        self.assertAlmostEqual(target[0],.305)
        self.assertTrue(any('WORKSPACE_LIMIT' in x for x in reasons))

    def test_normal_exit_holds_without_estop_or_disable(self):
        from robotarm.teleop.xr.arm_ops import hold_arm_on_exit
        from unittest.mock import Mock
        for mode in ('js','j'):
            robot = Mock()
            robot.get_joint_angles.return_value.msg = [.1,.2,-.3,.1,.2,.1]
            hold_arm_on_exit(robot,mode)
            getattr(robot,'move_'+mode).assert_called_once_with([.1,.2,-.3,.1,.2,.1])
            robot.electronic_emergency_stop.assert_not_called()
            robot.disable.assert_not_called()

    def test_yaw_calibration_preserves_vertical(self):
        limits = Limits(np.full(3,-1.),np.full(3,1.),.2,.0024,.014)
        ref = np.array([0.,0.,0.,0.,0.,0.,1.])
        moved = ref.copy(); moved[2] = -.1; moved[1] = .02
        target = build_target(ref,moved,np.zeros(6),1.,False,limits,yaw_degrees=90)
        np.testing.assert_allclose(target[:3],[0,.1,.02],atol=1e-10)

    def test_bent_posture_tracks_three_cm_lateral_in_one_second(self):
        from robotarm.teleop.xr.control import limit_joint_step
        from robotarm.teleop.xr.geometry import clamp_vector_step
        from robotarm.teleop.xr.piper_ik import PIPER_JOINT_MIN, PIPER_JOINT_MAX
        args = make_parser().parse_args([])
        ik = PiperIK(1/args.rate,False,args.orientation_weight)
        q = np.array([0.,.6,-.8,0.,.2,0.]); pose = ik.flange_pose(q); ref = pose.copy()
        steps = int(args.rate)
        for i in range(steps):
            goal = ref.copy(); goal[1] += .03*(i+1)/steps
            target = goal.copy()
            target[:3] = clamp_vector_step(pose[:3],goal[:3],args.translation_speed/args.rate)
            sol = ik.solve(q,target)
            q = limit_joint_step(q,sol,q,args.joint_speed/args.rate,args.max_joint_following_error,PIPER_JOINT_MIN,PIPER_JOINT_MAX)
            pose = ik.flange_pose(q)
        self.assertLess(np.linalg.norm(pose[:3]-goal[:3]),.001)

    def test_reject_unsafe_or_nonfinite_speed(self):
        for option,value in [('--joint-speed','nan'),('--joint-speed','2'),('--translation-speed','0'),('--orientation-weight','-1')]:
            with self.assertRaises(ValueError):
                validate_args(make_parser().parse_args([option,value]),np)

    def test_enable_waits_for_six_confirmed_joints(self):
        from robotarm.teleop.xr.arm_ops import enable_arm
        from unittest.mock import Mock
        robot = Mock()
        robot.enable.return_value = True
        robot.get_joints_enable_status_list.side_effect = [[], [True]*5+[False], [True]*6]
        with patch('robotarm.teleop.xr.arm_ops.time.sleep'):
            enable_arm(robot)
        self.assertEqual(robot.enable.call_count,3)

    def test_enable_timeout_does_not_report_success(self):
        from robotarm.teleop.xr.arm_ops import enable_arm
        from unittest.mock import Mock
        robot = Mock()
        robot.get_joints_enable_status_list.return_value = [False]*6
        with patch('robotarm.teleop.xr.arm_ops.time.sleep'), patch(
            'robotarm.teleop.xr.arm_ops.time.monotonic', side_effect=[0,.1,2]
        ):
            with self.assertRaises(RuntimeError): enable_arm(robot,1)

    def test_recording_cannot_reclaim_b_button(self):
        args = make_parser().parse_args(['--live','--record'])
        with self.assertRaisesRegex(ValueError,'hold B'):
            validate_args(args,np)

    def test_joint_limits_preserve_ratios(self):
        from robotarm.teleop.xr.control import limit_joint_step
        previous = np.zeros(6)
        target = np.array([.02,-.01,.005,0,0,0])
        result = limit_joint_step(previous,target,previous,.002,.05,
                                  np.full(6,-1.),np.full(6,1.))
        np.testing.assert_allclose(result,target*.1)
        result = limit_joint_step(previous,target,previous,.1,.001,
                                  np.full(6,-1.),np.full(6,1.))
        np.testing.assert_allclose(result,target*.05)

    def test_infeasible_following_bound_rejects(self):
        from robotarm.teleop.xr.control import limit_joint_step
        with self.assertRaises(RuntimeError):
            limit_joint_step(np.zeros(6),np.ones(6)*.01,np.ones(6),.002,.05,
                             np.full(6,-2.),np.full(6,2.))

    def test_zero_pose_lateral_motion_does_not_become_vertical(self):
        from robotarm.teleop.xr.control import limit_joint_step
        from robotarm.teleop.xr.piper_ik import PIPER_JOINT_MIN, PIPER_JOINT_MAX
        ik = PiperIK(.02,False)
        q = np.zeros(6)
        ref = ik.flange_pose(q); target = ref.copy(); target[1] += .0012
        solution = ik.solve(q,target)
        cmd = limit_joint_step(q,solution,q,.002,.05,PIPER_JOINT_MIN,PIPER_JOINT_MAX)
        delta = ik.flange_pose(cmd)[:3] - ref[:3]
        self.assertGreater(delta[1],.00008)
        self.assertLess(abs(delta[2]),delta[1]*.01)

    def test_adaptive_ik_reports_residual_and_condition_numbers(self):
        ik = PiperIK(.02, False)
        seed = np.array([0., .6, -.8, 0., .2, 0.])
        target = ik.flange_pose(seed)
        target[1] += .0024
        result = ik.solve_adaptive(seed, target)
        self.assertEqual(result.diagnostics.target_scale, 1.0)
        self.assertLess(result.diagnostics.position_error_m, .0008)
        self.assertTrue(np.isfinite(result.diagnostics.position_condition))
        self.assertTrue(np.isfinite(result.diagnostics.pose_condition))

    def test_adaptive_ik_shrinks_target_until_residual_is_acceptable(self):
        ik = PiperIK(.02, False)
        seed = np.array([0., .6, -.8, 0., .2, 0.])
        target = ik.flange_pose(seed)
        target[0] += .004
        with patch.object(ik, "_solve_once", return_value=seed.copy()):
            result = ik.solve_adaptive(seed, target)
        self.assertEqual(result.diagnostics.target_scale, .125)
        self.assertEqual(result.diagnostics.attempts, 4)
        self.assertLessEqual(result.diagnostics.position_error_m, .0008)

    def test_joint_limit_diagnostics_names_active_sources(self):
        from robotarm.teleop.xr.control import joint_limit_diagnostics
        previous = np.zeros(6)
        requested = np.full(6, .1)
        feedback = np.zeros(6)
        command = np.full(6, .002)
        ratio, sources = joint_limit_diagnostics(
            previous, requested, feedback, command, .002, .05,
            np.full(6, -1.), np.full(6, 1.),
        )
        self.assertLess(ratio, 1.0)
        self.assertIn("JOINT_STEP", sources)
        self.assertIn("JOINT_FOLLOW", sources)

        ratio, sources = joint_limit_diagnostics(
            previous, previous, feedback, previous, .002, .05,
            np.full(6, -1.), np.full(6, 1.),
        )
        self.assertEqual(ratio, 1.0)
        self.assertEqual(sources, [])

    def test_clutch_hysteresis_and_disconnect_rearm(self):
        from robotarm.teleop.xr.control import Clutch
        c = Clutch(.9,.7)
        self.assertFalse(c.update(1., True))  # held at startup
        self.assertFalse(c.update(0., True))
        self.assertTrue(c.update(.95, True))
        self.assertTrue(c.update(.8, True))  # no chatter below engage threshold
        self.assertFalse(c.update(.95, False))
        self.assertFalse(c.update(.95, True))  # reconnection must not resume
        self.assertFalse(c.update(.6, True))
        self.assertTrue(c.update(.95, True))

    def test_translation_holds_orientation(self):
        ik = PiperIK(.02, False)
        q = np.array([0., .6, -.8, 0., .2, 0.])
        ref = ik.flange_pose(q)
        target = ref.copy()
        target[0] += .01
        for _ in range(100):
            q = ik.solve(q, target)
        actual = ik.flange_pose(q)
        self.assertLess(np.linalg.norm(actual[:3] - target[:3]), 1e-4)
        error = Rotation.from_euler('xyz', actual[3:]) * Rotation.from_euler('xyz', ref[3:]).inv()
        self.assertLess(error.magnitude(), 1e-3)

    def test_rotation_uses_short_path_across_wrap(self):
        current = np.deg2rad([0, 0, 179])
        requested = np.deg2rad([0, 0, -179])
        limited = clamp_rotation_step(current, requested, np.deg2rad(.8))
        delta = Rotation.from_euler('xyz', limited) * Rotation.from_euler('xyz', current).inv()
        self.assertAlmostEqual(delta.magnitude(), np.deg2rad(.8))
        self.assertGreater(delta.as_rotvec()[2], 0)

    def test_relative_motion_and_orientation_reference(self):
        limits = Limits(np.array([.1,-.3,.05]),np.array([.55,.3,.55]),.2,.0012,.014)
        ref = np.array([0.,0.,0.,0.,0.,0.,1.])
        pose = ref.copy(); pose[2] = -.1
        flange = np.array([.3,0.,.2,.1,.2,.3])
        target = build_target(ref,pose,flange,.5,False,limits)
        np.testing.assert_allclose(target, [.35,0.,.2,.1,.2,.3])

    def test_startup_rejects_frozen_nonzero_timestamp(self):
        class Frozen:
            def get_time_stamp_ns(self): return 100
            def get_right_controller_pose(self): return [0,0,0,0,0,0,1]
        with patch('robotarm.teleop.xr.runtime.time.sleep'), patch(
            'robotarm.teleop.xr.runtime.time.monotonic', side_effect=[0,.01,.02,.03,2]
        ):
            with self.assertRaisesRegex(RuntimeError, 'No valid XR'):
                wait_for_xr(Frozen(),1,lambda: False)

    def test_defaults_bound_speed_and_no_recording(self):
        args = make_parser().parse_args([])
        validate_args(args,np)
        self.assertFalse(args.live)
        self.assertFalse(args.record)
        self.assertEqual(args.rate,100)
        self.assertAlmostEqual(
            min(args.rate * args.max_step_translation, args.translation_speed), .12
        )
        self.assertAlmostEqual(
            min(args.rate * args.max_step_joint, args.joint_speed), .45
        )
        args.grip_release_threshold = args.grip_threshold
        with self.assertRaises(ValueError): validate_args(args,np)


if __name__ == '__main__':
    unittest.main()
