"""Mode selection tests; mock runtimes so no XR or CAN connection is opened."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

from robotarm.teleop.xr import runtime, bimanual_runtime


class PicoModeTests(unittest.TestCase):
    def test_default_remains_single_arm_and_preserves_options(self):
        with patch.object(runtime, "_main_single", return_value=7) as single, patch.object(
            bimanual_runtime, "main"
        ) as dual:
            self.assertEqual(runtime.main(["--channel", "can0", "--xr-bridge"]), 7)
        single.assert_called_once_with(["--channel", "can0", "--xr-bridge"])
        dual.assert_not_called()

    def test_explicit_single_arm_only_runs_single_runtime(self):
        with patch.object(runtime, "_main_single", return_value=0) as single, patch.object(
            bimanual_runtime, "main"
        ) as dual:
            runtime.main(["--live", "--arms", "1", "--channel", "can1"])
        single.assert_called_once_with(["--live", "--channel", "can1"])
        dual.assert_not_called()

    def test_dual_arm_forwards_both_channels_and_shared_options(self):
        options = ["--left-channel", "can1", "--right-channel", "can0", "--xr-bridge", "--live", "--joint-speed", "0.15"]
        with patch.object(runtime, "_main_single") as single, patch.object(
            bimanual_runtime, "main", return_value=9
        ) as dual:
            self.assertEqual(runtime.main(["--arms=2", *options]), 9)
        dual.assert_called_once_with(options)
        single.assert_not_called()

    def test_sys_argv_module_invocation_uses_selector(self):
        with patch.object(runtime.sys, "argv", ["piper-xr", "--arms", "2", "--xr-bridge"]), patch.object(
            bimanual_runtime, "main", return_value=0
        ) as dual:
            runtime.main()
        dual.assert_called_once_with(["--xr-bridge"])

    def test_invalid_arm_count_is_rejected_before_runtime_starts(self):
        with patch.object(runtime, "_main_single") as single, patch.object(
            bimanual_runtime, "main"
        ) as dual, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                runtime.main(["--arms", "3"])
        self.assertEqual(error.exception.code, 2)
        single.assert_not_called()
        dual.assert_not_called()

    def test_same_can_channels_are_rejected_before_loading_sdk(self):
        with patch("robotarm.core.robot.piper._load_sdk") as sdk:
            with self.assertRaisesRegex(ValueError, "must be different"):
                runtime.main(["--arms", "2", "--left-channel", "can0", "--right-channel", "can0"])
        sdk.assert_not_called()

    def test_single_arm_flags_do_not_silently_select_dual_channels(self):
        with patch("robotarm.core.robot.piper._load_sdk") as sdk, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                runtime.main(["--arms", "2", "--channel", "can0"])
        self.assertEqual(error.exception.code, 2)
        sdk.assert_not_called()

    def test_help_for_each_mode_does_not_load_sdk(self):
        for arms, expected_flag in ((1, "--channel"), (2, "--left-channel")):
            output = StringIO()
            with patch("robotarm.core.robot.piper._load_sdk") as sdk, redirect_stdout(output):
                with self.assertRaises(SystemExit) as error:
                    runtime.main(["--arms", str(arms), "--help"])
            self.assertEqual(error.exception.code, 0)
            self.assertIn(expected_flag, output.getvalue())
            self.assertIn("--arms", output.getvalue())
            sdk.assert_not_called()

    def test_parser_defaults_keep_preview_and_expected_arm_count(self):
        single = runtime.make_parser().parse_args([])
        dual = bimanual_runtime.make_parser().parse_args([])
        self.assertEqual(single.arms, 1)
        self.assertEqual(dual.arms, 2)
        self.assertFalse(single.live)
        self.assertFalse(dual.live)


if __name__ == "__main__":
    unittest.main()
