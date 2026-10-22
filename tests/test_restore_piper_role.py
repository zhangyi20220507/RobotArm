"""Offline checks for role recovery; no CAN connection is opened."""
import unittest
from unittest.mock import Mock, patch

from robotarm import data_acquisition
from robotarm.hardware import restore_role


class RestorePiperRoleTests(unittest.TestCase):
    def test_preview_and_missing_confirmation_do_not_connect(self):
        with patch.object(restore_role.base, '_load_sdk') as load, patch('builtins.print'):
            self.assertEqual(restore_role.main(['--channel', 'can1']), 0)
            with self.assertRaisesRegex(RuntimeError, 'confirm-drop'):
                restore_role.main(['--channel', 'can1', '--execute'])
        load.assert_not_called()

    def test_execute_only_changes_selected_arm_role(self):
        arm = Mock()
        factory = Mock()
        factory.create_arm.return_value = arm
        with patch.object(restore_role.base, '_load_sdk', return_value=(None, factory, None, None)), patch.object(
            restore_role.base, 'build_config', return_value={'channel': 'can1'}
        ), patch.object(restore_role.base, 'disconnect_quietly') as disconnect, patch('builtins.print'):
            self.assertEqual(restore_role.main([
                '--channel', 'can1', '--firmware', 'v189',
                '--confirm-drop', '--execute',
            ]), 0)
        factory.create_arm.assert_called_once_with({'channel': 'can1'})
        arm.connect.assert_called_once()
        arm.set_follower_mode.assert_called_once()
        arm.move_j.assert_not_called()
        arm.move_js.assert_not_called()
        arm.enable.assert_not_called()
        arm.reset.assert_not_called()
        disconnect.assert_called_once_with(arm)

    def test_cli_workflow_is_registered(self):
        self.assertIn('restore-piper-role', data_acquisition.WORKFLOWS)


if __name__ == '__main__':
    unittest.main()
