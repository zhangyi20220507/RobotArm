"""Control-period deadline gating in shadow approval."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotarm.modeling import approve


def _row(**overrides) -> dict:
    """One valid ACT shadow row, complete enough to reach the approval output."""
    row = {
        "mode": "shadow", "command_sent": False, "healthy": True,
        "checkpoint_sha256": "0" * 64, "policy_type": "act",
        "inference_ms": 3.0, "camera_age_ms": 20.0,
        "control_period_ms": 50.0, "control_loop_ms": 5.0,
        "control_deadline_missed": False,
        "device": "cuda", "interface": "socketcan", "channel": "can1",
        "firmware": "v189", "camera_serial": "347522071736", "camera_fps": 20,
        "task": None, "chunk_size": 100, "temporal_ensemble_coeff": None,
        "n_action_steps": 50, "num_flow_steps": None, "rate_hz": 20.0,
        "planner_mode": None, "max_chunk_inference_ms": 60.0,
        "max_smolvla_plan_age_ms": 1250.0, "smolvla_replan_lead_steps": None,
        "max_joint_step_deg": 1, "max_gripper_step_mm": 1.0,
        "max_joint_lead_deg": 1.0, "max_gripper_lead_mm": 2.0,
        "max_act_raw_joint_gap_deg": 9.0,
        "max_act_raw_gripper_gap_mm": 18.0,
        "max_act_raw_gripper_bound_tolerance_mm": 1.0,
        "max_act_clamp_hold_steps": 10,
        "gripper_max_mm": 100.0, "gripper_force": 1.5, "max_camera_age_ms": 50.0,
        "raw_action": [0.0] * 7, "safe_action": [0.0] * 7,
        "act_raw_action_ok": True,
    }
    row.update(overrides)
    return row


class DeadlineGateTests(unittest.TestCase):
    def _approve(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "shadow.jsonl"
            report.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            return approve.build_approval(
                report, Path(directory) / "approval.json", 1, 0.05, **kwargs
            )

    def test_isolated_miss_is_approved_and_recorded(self):
        # 1/1000 mirrors the observed benign transient: one iteration at 62 ms
        # against a 50 ms period, with every row still watchdog-healthy.
        rows = [_row(control_loop_ms=62.0, control_deadline_missed=True)]
        rows += [_row() for _ in range(999)]
        approval = self._approve(rows)
        self.assertEqual(approval["deadline_misses"], 1)
        self.assertAlmostEqual(approval["deadline_miss_fraction"], 0.001)
        self.assertAlmostEqual(approval["max_control_loop_ms"], 62.0)
        self.assertAlmostEqual(approval["max_control_loop_periods"], 1.24)

    def test_clustered_misses_are_rejected(self):
        rows = [_row(control_loop_ms=80.0, control_deadline_missed=True)] * 50
        rows += [_row() for _ in range(950)]
        with self.assertRaisesRegex(SystemExit, "deadline miss fraction"):
            self._approve(rows)

    def test_single_stall_is_rejected_even_within_fraction(self):
        # The 1118 ms iteration from the older dp report stays fatal on its own.
        rows = [_row(control_loop_ms=1118.0, control_deadline_missed=True)]
        rows += [_row() for _ in range(999)]
        with self.assertRaisesRegex(SystemExit, "longest control iteration"):
            self._approve(rows)

    def test_reports_without_loop_timings_fall_back_to_the_miss_flag(self):
        rows = [_row(control_loop_ms=None, control_period_ms=None)] * 10
        approval = self._approve(rows)
        self.assertEqual(approval["deadline_misses"], 0)
        self.assertIsNone(approval["max_control_loop_ms"])
        self.assertIsNone(approval["max_control_loop_periods"])

    def test_thresholds_are_overridable(self):
        rows = [_row(control_loop_ms=62.0, control_deadline_missed=True)]
        rows += [_row() for _ in range(999)]
        with self.assertRaisesRegex(SystemExit, "deadline miss fraction"):
            self._approve(rows, max_control_deadline_miss_fraction=0.0)
        with self.assertRaisesRegex(SystemExit, "longest control iteration"):
            self._approve(rows, max_control_loop_periods=1.0)

    def test_watchdog_violations_still_stop_the_run(self):
        rows = [_row(control_loop_ms=62.0, control_deadline_missed=True)]
        rows += [_row(healthy=False) for _ in range(999)]
        with self.assertRaisesRegex(SystemExit, "watchdog violations"):
            self._approve(rows)

    def test_act_raw_action_watchdog_failure_rejects_approval(self):
        rows = [_row(act_raw_action_ok=False)]
        with self.assertRaisesRegex(
            SystemExit, "ACT raw-action watchdog rejected 1/1"
        ):
            self._approve(rows)

    def test_act_clamp_fraction_and_new_limits_are_recorded(self):
        clamped = _row(
            raw_action=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            safe_action=[0.0] * 7,
        )
        approval = self._approve([_row(), clamped])
        self.assertEqual(approval["clamp_fraction"], 0.5)
        self.assertEqual(approval["severe_clamp_fraction"], 0.5)
        self.assertIs(approval["act_raw_action_watchdog_passed"], True)
        self.assertEqual(approval["runtime"]["max_act_raw_joint_gap_deg"], 9.0)
        self.assertEqual(
            approval["runtime"]["max_act_raw_gripper_bound_tolerance_mm"], 1.0
        )
        self.assertEqual(approval["runtime"]["max_act_clamp_hold_steps"], 10)


if __name__ == "__main__":
    unittest.main()
