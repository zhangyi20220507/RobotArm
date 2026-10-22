"""Exercise runtime policy registration in a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys
import unittest


class ValidateRuntimeTests(unittest.TestCase):
    def test_policy_registration_and_pi05_guard_in_fresh_processes(self):
        # A subprocess prevents imports from other tests from filling the
        # registry and concealing a missing registration in validate-runtime.
        script = '''
import json
from pathlib import Path
import tempfile
from unittest import mock
from robotarm.modeling import approve, online_policy, validate_runtime

with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory)
    (checkpoint / "config.json").write_text(json.dumps({"type": POLICY_TYPE}))
    report = checkpoint / "shadow.jsonl"
    approval = checkpoint / "approval.json"
    with (
        mock.patch.object(online_policy, "checkpoint_digest", return_value="0" * 64),
        mock.patch.object(online_policy, "runtime_artifact_paths", return_value=(report, report, approval)),
        mock.patch.object(online_policy, "main", return_value=0) as shadow,
        mock.patch.object(approve, "main", return_value=0) as approve_run,
    ):
        if POLICY_TYPE == "pi05":
            try:
                validate_runtime.main(["--checkpoint", str(checkpoint)])
            except RuntimeError as error:
                assert "cannot receive execution approval" in str(error), str(error)
            else:
                raise AssertionError("Pi0.5 approval was allowed")
            shadow.assert_not_called()
            approve_run.assert_not_called()
        else:
            assert validate_runtime.main([
                "--checkpoint", str(checkpoint), "--interface", "socketcan",
                "--channel", "can1", "--firmware", "v189",
                "--feedback-timeout", "4",
            ]) == 0
            shadow.assert_called_once()
            assert "--execute" not in shadow.call_args.args[0]
            forwarded = shadow.call_args.args[0]
            assert forwarded[forwarded.index("--channel") + 1] == "can1"
            assert forwarded[forwarded.index("--interface") + 1] == "socketcan"
            assert forwarded[forwarded.index("--firmware") + 1] == "v189"
            assert forwarded[forwarded.index("--feedback-timeout") + 1] == "4.0"
            approve_run.assert_called_once()
'''
        for policy_type in ("act", "pi05"):
            with self.subTest(policy_type=policy_type):
                result = subprocess.run(
                    [sys.executable, "-c", f"POLICY_TYPE = {policy_type!r}\n" + script],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
