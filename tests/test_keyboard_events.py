"""Key decoding and isolated pseudo-terminal restoration; no hardware."""
from __future__ import annotations

import unittest
import os
from unittest.mock import patch

from robotarm.teleop.keyboard import TerminalKeyEventReader


class TerminalKeyEventReaderTests(unittest.TestCase):
    def test_adjacent_escape_s_is_alt_s(self):
        events, pending = TerminalKeyEventReader._decode_bytes(
            b"\x1bs", 10.0, None
        )
        self.assertEqual(events, [("alt+s", 10.0)])
        self.assertIsNone(pending)

    def test_expired_escape_does_not_turn_later_s_into_alt_s(self):
        events, pending = TerminalKeyEventReader._decode_bytes(
            b"\x1b", 10.0, None
        )
        self.assertEqual(events, [])
        self.assertEqual(pending, 10.0)

        events, pending = TerminalKeyEventReader._decode_bytes(
            b"s", 10.0 + TerminalKeyEventReader.ALT_SEQUENCE_TIMEOUT + 0.01,
            pending,
        )
        self.assertEqual(events, [("s", 10.06)])
        self.assertIsNone(pending)


@unittest.skipUnless(os.name == "posix", "POSIX terminal settings")
class TerminalRestoreTests(unittest.TestCase):
    def setUp(self):
        import pty
        import termios

        self.master, self.slave = pty.openpty()
        self.addCleanup(os.close, self.master)
        self.addCleanup(os.close, self.slave)
        self.stdin = os.fdopen(os.dup(self.slave), "r")
        self.addCleanup(self.stdin.close)
        self.original = termios.tcgetattr(self.slave)

    def test_policy_restores_echo_after_late_cleanup_on_every_exit_path(self):
        import termios
        import tty
        from robotarm.modeling import online_policy

        for error in (None, RuntimeError("cleanup failed"), KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                def run(_):
                    reader = TerminalKeyEventReader()
                    reader.start()
                    try:
                        self.assertFalse(termios.tcgetattr(self.slave)[3] & termios.ECHO)
                    finally:
                        reader.close()
                    # Simulate a later component leaving cbreak/raw mode after
                    # the reader's own restoration has already taken place.
                    tty.setraw(self.slave)
                    if error is not None:
                        raise error
                    return 0

                with patch("sys.stdin", self.stdin), patch.object(online_policy, "_main", side_effect=run):
                    if error is None:
                        self.assertEqual(online_policy.main([]), 0)
                    else:
                        with self.assertRaises(type(error)):
                            online_policy.main([])
                self.assertEqual(termios.tcgetattr(self.slave), self.original)

    def test_restoration_survives_stdin_replacement_and_is_immediate(self):
        import io
        import termios
        import tty
        from robotarm.teleop.keyboard import preserve_terminal_settings

        with patch("sys.stdin", self.stdin), patch("termios.tcsetattr", wraps=termios.tcsetattr) as restore:
            with preserve_terminal_settings():
                tty.setraw(self.slave)
                with patch("sys.stdin", io.StringIO()):
                    self.stdin.close()
            self.assertEqual(restore.call_args.args[1], termios.TCSANOW)
        self.assertEqual(termios.tcgetattr(self.slave), self.original)

    def test_shell_restore_repairs_inherited_no_echo_state(self):
        import termios
        import tty
        from robotarm.teleop.keyboard import preserve_terminal_settings

        tty.setcbreak(self.slave)
        self.assertFalse(termios.tcgetattr(self.slave)[3] & termios.ECHO)
        with patch("sys.stdin", self.stdin):
            with preserve_terminal_settings(ensure_shell_input=True):
                pass
        flags = termios.tcgetattr(self.slave)[3]
        self.assertTrue(flags & termios.ECHO)
        self.assertTrue(flags & termios.ICANON)
        self.assertTrue(flags & termios.ISIG)


if __name__ == "__main__":
    unittest.main()
