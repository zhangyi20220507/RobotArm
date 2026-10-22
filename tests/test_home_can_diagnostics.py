import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from robotarm.teleop.leader_follower.can_diagnostics import home_receive_diagnostics


class ReceiveDiagnosticsTests(unittest.TestCase):
    def test_reports_unknown_frames_and_removes_observer_on_failure(self):
        original = lambda frame: None
        callbacks = [original]
        robot = SimpleNamespace(_ctx=SimpleNamespace(_parser_packet_fun_list=callbacks))
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'missing feedback'):
            with home_receive_diagnostics(robot, 'can1'):
                for data in (b'\x01', b'\x01', b'\x00'):
                    callbacks[-1](SimpleNamespace(arbitration_id=0x3A5, data=data))
                raise RuntimeError('missing feedback')
        self.assertEqual(callbacks, [original])
        self.assertIn('0x3A5 count=3 changes=1', output.getvalue())
        self.assertIn('first=01 last=00', output.getvalue())

    def test_success_is_quiet_and_cleans_up(self):
        callbacks = []
        robot = SimpleNamespace(_ctx=SimpleNamespace(_parser_packet_fun_list=callbacks))
        output = io.StringIO()
        with redirect_stdout(output), home_receive_diagnostics(robot, 'can1'):
            callbacks[-1](SimpleNamespace(arbitration_id=0x155, data=b'\x00'))
        self.assertEqual(callbacks, [])
        self.assertEqual(output.getvalue(), '')

    def test_interrupt_without_frames_still_cleans_up(self):
        callbacks = []
        robot = SimpleNamespace(_ctx=SimpleNamespace(_parser_packet_fun_list=callbacks))
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(KeyboardInterrupt):
            with home_receive_diagnostics(robot, 'can1'):
                raise KeyboardInterrupt()
        self.assertEqual(callbacks, [])
        self.assertIn('no CAN frames received', output.getvalue())
