"""Dual-Piper software reset tests; no CAN device is opened."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from robotarm import data_acquisition
from robotarm.hardware import reset_both


def normal_status(timestamp: float = 101.0):
    return SimpleNamespace(
        timestamp=timestamp,
        msg=SimpleNamespace(arm_status=0),
    )


class ResetBothArmsTests(unittest.TestCase):
    def test_execute_requires_drop_confirmation_without_connecting(self):
        with patch.object(reset_both.base, "connect_robot") as connect:
            with self.assertRaisesRegex(RuntimeError, "confirm-drop"):
                reset_both.main(["--execute"])
        connect.assert_not_called()

    def test_dry_run_preflights_both_without_reset(self):
        follower, leader = Mock(), Mock()
        with patch.object(
            reset_both.base,
            "connect_robot",
            side_effect=[(follower, {}), (leader, {})],
        ), patch.object(reset_both, "_wait_controller_feedback"), patch.object(
            reset_both.base, "disconnect_quietly"
        ) as disconnect, patch("builtins.print"):
            self.assertEqual(reset_both.main([]), 0)
        follower.reset.assert_not_called()
        leader.reset.assert_not_called()
        self.assertEqual(disconnect.call_args_list, [call(leader), call(follower)])

    def test_execute_resets_and_confirms_both_controllers(self):
        follower, leader = Mock(), Mock()
        follower.get_arm_status.return_value = normal_status()
        leader.get_arm_status.return_value = normal_status()
        with patch.object(
            reset_both.base,
            "connect_robot",
            side_effect=[(follower, {}), (leader, {})],
        ), patch.object(reset_both, "_wait_controller_feedback"), patch.object(
            reset_both.base, "disconnect_quietly"
        ), patch.object(reset_both.time, "time", return_value=100.0), patch(
            "builtins.print"
        ):
            self.assertEqual(
                reset_both.main(["--confirm-drop", "--execute"]), 0
            )
        follower.reset.assert_called_once_with()
        leader.reset.assert_called_once_with()

    def test_channels_must_be_distinct(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            reset_both.main([
                "--follower-channel", "can0", "--leader-channel", "can0"
            ])

    def test_preflight_accepts_fresh_drag_feedback_without_normal_joints(self):
        robot = Mock()
        robot.get_arm_status.return_value = None
        robot.get_joint_angles.return_value = None
        robot.get_leader_joint_angles.return_value = SimpleNamespace(timestamp=99.9)
        with patch.object(reset_both.time, "time", return_value=100.0):
            source = reset_both._wait_controller_feedback(robot, "can_leader1", 3.0)
        self.assertEqual(source, "leader drag feedback")
        robot.reset.assert_not_called()

    def test_no_controller_is_reset_when_leader_preflight_fails(self):
        follower, leader = Mock(), Mock()
        with patch.object(
            reset_both.base, "connect_robot", side_effect=[(follower, {}), (leader, {})]
        ), patch.object(
            reset_both, "_wait_controller_feedback", side_effect=["arm status", TimeoutError("no leader feedback")]
        ), patch.object(reset_both.base, "disconnect_quietly"), patch("builtins.print"):
            with self.assertRaisesRegex(TimeoutError, "no leader feedback"):
                reset_both.main(["--confirm-drop", "--execute"])
        follower.reset.assert_not_called()
        leader.reset.assert_not_called()

    def test_preflight_rejects_stale_cached_drag_feedback(self):
        robot = Mock()
        robot.get_arm_status.return_value = None
        robot.get_joint_angles.return_value = None
        robot.get_leader_joint_angles.return_value = SimpleNamespace(timestamp=90.0)
        with patch.object(reset_both.time, "time", return_value=100.0), patch.object(
            reset_both.time, "monotonic", side_effect=[0.0, 0.1, 4.0]
        ), patch.object(reset_both.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "can_leader1.*no fresh"):
                reset_both._wait_controller_feedback(robot, "can_leader1", 3.0)

    def test_cli_workflow_is_registered(self):
        self.assertIn("reset-both-arms", data_acquisition.WORKFLOWS)


if __name__ == "__main__":
    unittest.main()
