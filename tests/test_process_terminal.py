"""CLI child lifetime and TTY ownership under a real terminal Ctrl+C."""
import os
import sys
import unittest
from unittest.mock import Mock, patch

from robotarm.legacy import run_process


class ProcessWaitTests(unittest.TestCase):
    def test_repeated_interrupts_wait_for_child_without_signalling_it_again(self):
        process = Mock()
        process.wait.side_effect = [KeyboardInterrupt(), KeyboardInterrupt(), 0]
        with patch("robotarm.legacy.subprocess.Popen", return_value=process), patch(
            "robotarm.legacy.preserve_terminal_settings"
        ) as terminal:
            self.assertEqual(run_process(["fake-child"]), 130)
        self.assertEqual(process.wait.call_count, 3)
        process.send_signal.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        terminal.assert_called_once_with(ensure_shell_input=True)

    def test_normal_child_exit_code_is_preserved(self):
        process = Mock()
        process.wait.return_value = 7
        with patch("robotarm.legacy.subprocess.Popen", return_value=process), patch(
            "robotarm.legacy.preserve_terminal_settings"
        ):
            self.assertEqual(run_process(["fake-child"]), 7)


@unittest.skipUnless(os.name == "posix", "POSIX pseudo-terminal required")
class ProcessTerminalTests(unittest.TestCase):
    def test_ctrl_c_waits_for_cleanup_and_restores_after_child_atexit(self):
        import pty
        import select
        import signal
        import termios
        import time

        child = """
import atexit, time, tty
atexit.register(lambda: tty.setcbreak(0))
tty.setcbreak(0)
print('CHILD_READY', flush=True)
try:
    while True:
        time.sleep(.02)
except KeyboardInterrupt:
    time.sleep(.8)  # Simulated homing/recorder shutdown, no robot hardware.
    print('CHILD_CLEANUP_DONE', flush=True)
"""
        launcher = (
            "import sys\nfrom robotarm.legacy import run_process\n"
            f"result = run_process([sys.executable, '-c', {child!r}])\n"
            "print('WRAPPER_RETURNED', result, flush=True)\n"
        )
        pid, master = pty.fork()
        if pid == 0:
            os.execv(sys.executable, [sys.executable, "-c", launcher])
        reaped = False
        output = b""
        deadline = time.monotonic() + 10

        def read_until(marker):
            nonlocal output
            while marker not in output:
                if time.monotonic() >= deadline:
                    self.fail(f"terminal child timed out: {output!r}")
                ready, _, _ = select.select([master], [], [], .1)
                if ready:
                    try:
                        data = os.read(master, 4096)
                    except OSError:
                        data = b""
                    if not data:
                        self.fail(f"terminal closed early: {output!r}")
                    output += data

        try:
            read_until(b"CHILD_READY")
            self.assertFalse(termios.tcgetattr(master)[3] & termios.ECHO)
            os.write(master, b"\x03")  # Signal the foreground group, as a user does.
            read_until(b"WRAPPER_RETURNED 130")
            self.assertIn(b"CHILD_CLEANUP_DONE", output)
            self.assertLess(output.index(b"CHILD_CLEANUP_DONE"), output.index(b"WRAPPER_RETURNED"))
            flags = termios.tcgetattr(master)[3]
            self.assertTrue(flags & termios.ECHO)
            self.assertTrue(flags & termios.ICANON)
            _, status = os.waitpid(pid, 0)
            reaped = True
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        finally:
            if not reaped:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
            os.close(master)
