"""Offline checks for PICO bimanual control; never connects to CAN hardware."""

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from robotarm.teleop.xr.bimanual_runtime import (
    ArmSide,
    LongPress,
    _estop_all,
    _enable,
    _home_both_sequentially,
    home_grip_released,
    make_parser,
    simultaneous_home_pressed,
    snapshot_xr,
    validate_args,
    wait_for_bimanual_xr,
    main,
)
from robotarm.teleop.xr.arm_ops import update_force_limited_gripper


class FakeXr:
    def get_time_stamp_ns(self): return 123
    def get_headset_pose(self): return [0, 0, 0, 0, 0, 0, 1]
    def get_left_controller_pose(self): return [1, 2, 3, 0, 0, 0, 1]
    def get_right_controller_pose(self): return [4, 5, 6, 0, 0, 0, 1]
    def get_left_grip(self): return .25
    def get_right_grip(self): return .75
    def get_left_trigger(self): return .1
    def get_right_trigger(self): return .9
    def get_left_axis_click(self): return False
    def get_right_axis_click(self): return True
    def get_X_button(self): return True
    def get_Y_button(self): return False
    def get_A_button(self): return False
    def get_B_button(self): return True


class PicoBimanualTests(unittest.TestCase):
    def test_default_channels_are_distinct_and_dry_run(self):
        args = make_parser().parse_args([])
        validate_args(args)
        self.assertEqual(args.left_channel, "can_leader1")
        self.assertEqual(args.right_channel, "can_follower1")
        self.assertFalse(args.live)
        self.assertEqual(args.stale_timeout, 0.2)
        self.assertEqual(args.homing_stale_timeout, 1.0)
        self.assertEqual(args.gripper_force_hold_threshold, 0.8)
        self.assertEqual(args.command_modifier_threshold, 0.8)
        self.assertEqual(args.scale, 2.0)
        self.assertEqual(args.tracking_frame, "head")
        self.assertEqual(args.engage_settle_seconds, 0.25)
        self.assertEqual(args.engage_deadband_m, 0.002)
        self.assertFalse(args.verbose_status)

    def test_same_can_channel_is_rejected(self):
        args = make_parser().parse_args(["--left-channel", "can0", "--right-channel", "can0"])
        with self.assertRaisesRegex(ValueError, "must be different"):
            validate_args(args)

    def test_direct_snapshot_contains_both_controllers(self):
        state = snapshot_xr(FakeXr(), False)
        self.assertEqual(state["headset_pose"][6], 1)
        self.assertEqual(state["left_controller_pose"][:3], [1, 2, 3])
        self.assertEqual(state["right_controller_pose"][:3], [4, 5, 6])
        self.assertAlmostEqual(state["left_grip"], .25)
        self.assertTrue(state["right_axis_click"])
        self.assertTrue(state["X_button"])
        self.assertTrue(state["B_button"])

    def test_xr_wait_requires_both_poses_and_advancing_timestamp(self):
        valid = [0, 0, 0, 0, 0, 0, 1]
        xrt = Mock()
        xrt.snapshot.side_effect = [
            {"timestamp_ns": 1, "right_controller_pose": valid},
            {"timestamp_ns": 2, "left_controller_pose": [0]*7, "right_controller_pose": valid},
            {"timestamp_ns": 3, "left_controller_pose": valid, "right_controller_pose": valid},
        ]
        with patch("robotarm.teleop.xr.bimanual_runtime.time.sleep"), patch("builtins.print"):
            wait_for_bimanual_xr(xrt, True, 1.0, lambda: False)
        self.assertEqual(xrt.snapshot.call_count, 3)

    def test_xr_wait_reports_missing_left_pose_instead_of_key_error(self):
        xrt = Mock()
        xrt.snapshot.return_value = {"timestamp_ns": 1, "right_controller_pose": [0,0,0,0,0,0,1]}
        with patch("robotarm.teleop.xr.bimanual_runtime.time.monotonic", side_effect=[0,0,0,0,2]), patch(
            "robotarm.teleop.xr.bimanual_runtime.time.sleep"
        ), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "left_pose=missing"):
                wait_for_bimanual_xr(xrt, True, 1.0, lambda: False)

    def test_xr_wait_rejects_frozen_timestamp_even_with_valid_poses(self):
        xrt = Mock()
        valid = [0,0,0,0,0,0,1]
        xrt.snapshot.return_value = {"timestamp_ns": 1, "left_controller_pose": valid, "right_controller_pose": valid}
        with patch("robotarm.teleop.xr.bimanual_runtime.time.monotonic", side_effect=[0,0,0,0,.1,.1,2]), patch(
            "robotarm.teleop.xr.bimanual_runtime.time.sleep"
        ), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "not changing"):
                wait_for_bimanual_xr(xrt, True, 1.0, lambda: False)

    def test_xr_startup_failure_does_not_connect_or_estop_arms(self):
        factory, create_config, xrt = Mock(), Mock(), Mock()
        model = SimpleNamespace(PIPER="piper")
        firmware = SimpleNamespace(DEFAULT="default", V183="v183", V188="v188", V189="v189")
        with patch("robotarm.core.robot.piper._load_sdk", return_value=(create_config, factory, model, firmware)), patch(
            "robotarm.teleop.xr.bridge.XrBridgeClient", return_value=xrt
        ), patch("robotarm.teleop.xr.bimanual_runtime.wait_for_bimanual_xr", side_effect=RuntimeError("no XR data")), patch(
            "robotarm.teleop.xr.bimanual_runtime.signal.signal"
        ), patch("robotarm.teleop.xr.bimanual_runtime._estop_all") as estop, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "no XR data"):
                main(["--live", "--xr-bridge"])
        create_config.assert_not_called()
        factory.create_arm.assert_not_called()
        estop.assert_not_called()
        xrt.close.assert_called_once_with()

    def test_long_press_fires_once_until_release(self):
        press = LongPress()
        self.assertFalse(press.update(True, 0.0, 1.0))
        self.assertFalse(press.update(True, .9, 1.0))
        self.assertTrue(press.update(True, 1.0, 1.0))
        self.assertFalse(press.update(True, 2.0, 1.0))
        self.assertFalse(press.update(False, 2.1, 1.0))
        self.assertFalse(press.update(True, 3.0, 1.0))
        self.assertTrue(press.update(True, 4.0, 1.0))

    def test_home_only_requires_the_corresponding_grip_to_be_released(self):
        state = {"left_grip": 1.0, "right_grip": 0.0}
        self.assertTrue(home_grip_released(state, "right", 0.7))
        self.assertFalse(home_grip_released(state, "left", 0.7))

        state = {"left_grip": 0.0, "right_grip": 1.0}
        self.assertTrue(home_grip_released(state, "left", 0.7))
        self.assertFalse(home_grip_released(state, "right", 0.7))

    def test_simultaneous_home_requires_both_home_buttons(self):
        self.assertTrue(simultaneous_home_pressed({
            "left_grip": 0.8, "right_grip": 0.9,
            "left_trigger": 0.0, "right_trigger": 0.0,
            "X_button": True, "A_button": True,
        }))
        self.assertTrue(simultaneous_home_pressed({
            "left_grip": 0.0, "right_grip": 0.0,
            "left_trigger": 0.8, "right_trigger": 0.9,
            "X_button": True, "A_button": True,
        }))
        self.assertFalse(simultaneous_home_pressed({
            "left_grip": 0.0, "right_grip": 0.0,
            "left_trigger": 0.79, "right_trigger": 0.9,
            "X_button": True, "A_button": True,
        }))
        self.assertFalse(simultaneous_home_pressed({
            "left_grip": 0.0, "right_grip": 0.0,
            "left_trigger": 0.9, "right_trigger": 0.9,
            "X_button": True, "A_button": False,
        }))

    def test_both_arm_home_is_sequential_left_then_right(self):
        left, right, args, xrt = Mock(name="left"), Mock(name="right"), Mock(), Mock()
        calls = []

        def record_home(side, other, *_):
            calls.append((side, other))

        with patch(
            "robotarm.teleop.xr.bimanual_runtime._home", side_effect=record_home
        ), patch("builtins.print"):
            _home_both_sequentially(left, right, args, xrt, lambda: False)
        self.assertEqual(calls, [(left, right), (right, left)])

    def test_resume_with_grip_held_does_not_require_regrip(self):
        from robotarm.teleop.xr.control import Clutch

        clutch = Clutch(
            0.9, 0.7, engaged=True, blocked=True, mode="grip", fault_latched=True
        )
        side = ArmSide(
            "left", "left", "can0", Mock(), Mock(), Mock(), 0.0,
            np.zeros(6), clutch, ready=True,
        )
        with patch(
            "robotarm.teleop.xr.bimanual_runtime.wait_for_joint_angles",
            return_value=np.zeros(6),
        ), patch("builtins.print"):
            _enable(side, SimpleNamespace(engagement_mode="grip"))
        self.assertFalse(clutch.blocked)
        self.assertFalse(clutch.fault_latched)
        self.assertTrue(clutch.update(1.0, True))

    def test_resume_blocks_out_of_range_wrist_and_allows_homing(self):
        from robotarm.teleop.xr.control import Clutch

        current = np.zeros(6)
        current[4] = np.deg2rad(70.0)
        side = ArmSide(
            "right", "right", "can1", Mock(), Mock(), Mock(), 0.0,
            current.copy(), Clutch(0.9, 0.7, mode="grip"),
            ready=True, enabled=True, gripper=Mock(),
        )
        with patch(
            "robotarm.teleop.xr.bimanual_runtime.wait_for_joint_angles",
            return_value=current,
        ), patch("builtins.print") as output:
            _enable(side, SimpleNamespace())
        self.assertTrue(side.clutch.fault_latched)
        self.assertTrue(side.ready)
        self.assertIn("J5=70.00 deg", output.call_args.args[0])
        self.assertIn("RIGHT Grip+A", output.call_args.args[0])

    def test_trigger_opens_and_released_trigger_closes_gripper(self):
        gripper = Mock()
        gripper.get_gripper_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(mode="width", value=0.05, force=0.0)
        )
        closed, hold, event, _ = update_force_limited_gripper(
            gripper, 0.05, None, 0.0, 0.07, 0.001, 1.0, 0.8
        )
        opened, _, _, _ = update_force_limited_gripper(
            gripper, 0.05, None, 1.0, 0.07, 0.001, 1.0, 0.8
        )
        self.assertAlmostEqual(closed, 0.049)
        self.assertAlmostEqual(opened, 0.051)
        self.assertIsNone(hold)
        self.assertIsNone(event)

    def test_contact_force_latches_width_until_trigger_opens(self):
        gripper = Mock()
        gripper.get_gripper_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(mode="width", value=0.032, force=0.9)
        )
        command, hold, event, force = update_force_limited_gripper(
            gripper, 0.025, None, 0.0, 0.07, 0.001, 1.0, 0.8
        )
        self.assertAlmostEqual(command, 0.032)
        self.assertAlmostEqual(hold, 0.032)
        self.assertEqual(event, "latched")
        self.assertAlmostEqual(force, 0.9)

        command, hold, event, _ = update_force_limited_gripper(
            gripper, command, hold, 1.0, 0.07, 0.001, 1.0, 0.8
        )
        self.assertAlmostEqual(command, 0.033)
        self.assertIsNone(hold)
        self.assertEqual(event, "released")

    def test_global_estop_reaches_both_enabled_arms(self):
        sides = []
        for name in ("left", "right"):
            robot = Mock()
            side = ArmSide(
                name=name, hand=name, channel=f"can-{name}", robot=robot,
                ik=Mock(), limits=Mock(), yaw_degrees=0.0,
                joints=np.zeros(6), clutch=Mock(), enabled=True,
            )
            sides.append(side)
        _estop_all(sides)
        for side in sides:
            side.robot.electronic_emergency_stop.assert_called_once_with()
            self.assertTrue(side.emergency_stopped)

    def test_each_arm_has_independent_limit_notice_timer(self):
        left = ArmSide(
            "left", "left", "can0", Mock(), Mock(), Mock(), 0.0,
            np.zeros(6), Mock(),
        )
        right = ArmSide(
            "right", "right", "can1", Mock(), Mock(), Mock(), 0.0,
            np.zeros(6), Mock(),
        )
        left.last_limit_notice = 10.0
        self.assertEqual(right.last_limit_notice, 0.0)

    def test_unsafe_rates_and_workspace_are_rejected(self):
        args = make_parser().parse_args(["--joint-speed", "2"])
        with self.assertRaises(ValueError):
            validate_args(args)
        args = make_parser().parse_args([
            "--left-xyz-min", ".5,0,0", "--left-xyz-max", ".4,1,1"
        ])
        with self.assertRaisesRegex(ValueError, "left workspace"):
            validate_args(args)
        args = make_parser().parse_args(["--min-flange-separation", "-0.1"])
        with self.assertRaisesRegex(ValueError, "min-flange-separation"):
            validate_args(args)
        args = make_parser().parse_args([
            "--stale-timeout", "0.2", "--homing-stale-timeout", "0.1"
        ])
        with self.assertRaisesRegex(ValueError, "homing-stale-timeout"):
            validate_args(args)
        args = make_parser().parse_args([
            "--gripper-force", "1.0", "--gripper-force-hold-threshold", "1.1"
        ])
        with self.assertRaisesRegex(ValueError, "gripper-force-hold-threshold"):
            validate_args(args)
        args = make_parser().parse_args(["--command-modifier-threshold", "0"])
        with self.assertRaisesRegex(ValueError, "command-modifier-threshold"):
            validate_args(args)
        args = make_parser().parse_args(["--engage-settle-seconds", "1.1"])
        with self.assertRaisesRegex(ValueError, "engage-settle-seconds"):
            validate_args(args)
        args = make_parser().parse_args(["--engage-deadband-m", "0.021"])
        with self.assertRaisesRegex(ValueError, "engage-deadband-m"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
