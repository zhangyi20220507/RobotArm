"""Create checkpoint-bound approval only for a passing shadow report.

迁移自 Modeling/lerobot_training/scripts/approve_shadow.py。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build_approval(
    report: Path,
    output: Path,
    min_samples: int,
    max_smolvla_severe_clamp_fraction: float,
    max_control_deadline_miss_fraction: float = 0.005,
    max_control_loop_periods: float = 2.0,
) -> dict:
    if not 0.0 <= max_smolvla_severe_clamp_fraction <= 1.0:
        raise SystemExit("FAIL: max-smolvla-severe-clamp-fraction must be in [0, 1]")
    if not 0.0 <= max_control_deadline_miss_fraction <= 1.0:
        raise SystemExit("FAIL: max-control-deadline-miss-fraction must be in [0, 1]")
    if max_control_loop_periods <= 0.0:
        raise SystemExit("FAIL: max-control-loop-periods must be positive")
    rows = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
    if len(rows) < min_samples:
        raise SystemExit(f"FAIL: {len(rows)} samples; require {min_samples}")
    if any(row["mode"] != "shadow" or row["command_sent"] for row in rows):
        raise SystemExit("FAIL: report is not a pure shadow run")
    if any(not row["healthy"] for row in rows):
        raise SystemExit("FAIL: shadow run contains watchdog violations")
    # The control loop re-paces every tick, so one over-period iteration only
    # makes a single command a few ms late and does not accumulate.  What has
    # to be caught is sustained overload (misses clustering) or an outright
    # stall, so bound the miss fraction and the worst iteration instead of
    # rejecting any run that ever crossed the period.  Rows whose watchdog
    # tolerances were breached are already rejected above and stay the hard
    # stop; both statistics below are recorded in the approval either way.
    # Reports predating per-iteration telemetry carry no control_loop_ms and
    # fall back to the miss flag alone.
    miss_count = sum(1 for row in rows if row.get("control_deadline_missed", False))
    miss_fraction = miss_count / len(rows)
    loop_ms = [
        float(row["control_loop_ms"])
        for row in rows
        if row.get("control_loop_ms") is not None
    ]
    loop_periods = [
        float(row["control_loop_ms"]) / float(row["control_period_ms"])
        for row in rows
        if row.get("control_loop_ms") is not None and row.get("control_period_ms")
    ]
    if miss_fraction > max_control_deadline_miss_fraction:
        raise SystemExit(
            f"FAIL: control-period deadline miss fraction {miss_fraction:.3%} "
            f"exceeds {max_control_deadline_miss_fraction:.3%}"
        )
    if loop_periods and max(loop_periods) > max_control_loop_periods:
        raise SystemExit(
            f"FAIL: longest control iteration {max(loop_periods):.2f} control "
            f"periods exceeds {max_control_loop_periods:.2f}"
        )
    digests = {row["checkpoint_sha256"] for row in rows}
    if len(digests) != 1:
        raise SystemExit("FAIL: report contains multiple checkpoints")
    inference = sorted(row["inference_ms"] for row in rows)
    camera = sorted(row["camera_age_ms"] for row in rows)
    policy_types = {row.get("policy_type") for row in rows}
    if len(policy_types) != 1:
        raise SystemExit("FAIL: report contains multiple policy types")
    policy_type = policy_types.pop()
    if policy_type == "pi05":
        raise SystemExit(
            "FAIL: Pi0.5 base shadow output is uncalibrated 32D and cannot "
            "receive a CAN execution approval"
        )
    if policy_type in {"act", "pi0"} and any(
        row.get("act_raw_action_ok") is not True for row in rows
    ):
        bad_count = sum(
            row.get("act_raw_action_ok") is not True for row in rows
        )
        raise SystemExit(
            "FAIL: ACT raw-action watchdog rejected "
            f"{bad_count}/{len(rows)} shadow samples; execution would stop "
            "before sending a command"
        )
    # SmolVLA and Diffusion run behind the async planner; their reports must
    # carry the planner telemetry and are gated on it. ACT stays synchronous.
    async_planned = policy_type in ("smolvla", "diffusion", "pi0")

    def one_value(key):
        values = {json.dumps(row.get(key), sort_keys=True) for row in rows}
        if len(values) != 1 or next(iter(values)) == "null":
            raise SystemExit(f"FAIL: report has missing or inconsistent {key}")
        return json.loads(values.pop())

    def optional_value(key):
        """Like one_value, but a consistent null is a valid value (e.g. the
        task of a Diffusion run, SmolVLA-only flow steps)."""
        values = {json.dumps(row.get(key), sort_keys=True) for row in rows}
        if len(values) != 1:
            raise SystemExit(f"FAIL: report has inconsistent {key}")
        return json.loads(values.pop())

    runtime = None
    clamp_fraction = None
    severe_clamp_fraction = None
    chunk_inference_p95 = None
    if async_planned:
        if policy_type == "pi0" and one_value("pi0_clock_aligned") is not True:
            raise SystemExit("FAIL: Pi0 report has no clock alignment")
        if one_value("planner_mode") != "async_chunked":
            raise SystemExit("FAIL: report was not produced by async planner")
        if any(row.get("planner_underflow", True) for row in rows):
            raise SystemExit("FAIL: async planner ran out of actions")
        chunk_timings = [
            float(row["chunk_inference_ms"])
            for row in rows if row.get("chunk_inference_ms") is not None
        ]
        if not chunk_timings:
            raise SystemExit("FAIL: report contains no completed action chunks")
        chunk_inference_p95 = sorted(chunk_timings)[
            min(len(chunk_timings) - 1, round((len(chunk_timings) - 1) * 0.95))
        ]
        if any(
            timing > float(row["max_chunk_inference_ms"])
            for row in rows
            for timing in ([row["chunk_inference_ms"]]
                           if row.get("chunk_inference_ms") is not None else [])
        ):
            raise SystemExit("FAIL: chunk inference exceeded its budget")
        if any(
            row.get("plan_observation_age_ms") is None
            or float(row["plan_observation_age_ms"])
            > float(row["max_smolvla_plan_age_ms"])
            for row in rows
        ):
            raise SystemExit("FAIL: action plan exceeded its observation-age limit")
    # Clamp telemetry is useful for ACT too: a low validation loss does not
    # reveal that every online target is being rewritten by the safety layer.
    if policy_type in ("act", "smolvla", "diffusion", "pi0"):
        clamped = 0
        severe_clamped = 0
        for row in rows:
            raw = row.get("raw_action")
            safe = row.get("safe_action")
            if (
                not isinstance(raw, list)
                or not isinstance(safe, list)
                or len(raw) != len(safe)
            ):
                raise SystemExit("FAIL: report has invalid action vectors")
            residual = [
                abs(float(before) - float(after))
                for before, after in zip(raw, safe)
            ]
            if any(value > 1e-6 for value in residual):
                clamped += 1
            if (
                any(value > math.radians(5.0) for value in residual[:6])
                or residual[6] > 0.005
            ):
                severe_clamped += 1
        clamp_fraction = clamped / len(rows)
        severe_clamp_fraction = severe_clamped / len(rows)
        # The severe-clamp threshold is calibrated for SmolVLA's conservative
        # per-tick limits; Diffusion's 1 deg/tick absolute targets make >5 deg
        # residuals routine, so the fraction is recorded but not gated.
        if (
            policy_type == "smolvla"
            and severe_clamp_fraction > max_smolvla_severe_clamp_fraction
        ):
            raise SystemExit(
                "FAIL: SmolVLA severe safety-clamp fraction "
                f"{severe_clamp_fraction:.3%} exceeds "
                f"{max_smolvla_severe_clamp_fraction:.3%}"
            )

    camera_serials = optional_value("camera_serials")
    runtime = {
        "device": one_value("device"),
        "interface": one_value("interface"),
        "channel": one_value("channel"),
        "firmware": one_value("firmware"),
        "camera_serial": one_value("camera_serial"),
        **({"camera_serials": camera_serials} if camera_serials is not None else {}),
        "camera_fps": one_value("camera_fps"),
        "task": optional_value("task"),
        **({
            "chunk_size": one_value("chunk_size"),
            "temporal_ensemble_coeff": optional_value("temporal_ensemble_coeff"),
        } if policy_type in {"act", "pi0"} else {}),
        **({"pi0_clock_aligned": one_value("pi0_clock_aligned"),
            "num_inference_steps": one_value("num_inference_steps")}
           if policy_type == "pi0" else {}),
        "n_action_steps": one_value("n_action_steps"),
        "num_flow_steps": optional_value("num_flow_steps"),
        "rate_hz": one_value("rate_hz"),
        "planner_mode": optional_value("planner_mode"),
        "max_chunk_inference_ms": one_value("max_chunk_inference_ms"),
        # ACT runs synchronously and does not use an async plan-age watchdog.
        # Match online_policy's runtime binding, which uses None for ACT.
        "max_smolvla_plan_age_ms": (
            one_value("max_smolvla_plan_age_ms") if async_planned else None
        ),
        "smolvla_replan_lead_steps": optional_value("smolvla_replan_lead_steps"),
        "max_joint_step_deg": one_value("max_joint_step_deg"),
        "max_gripper_step_mm": one_value("max_gripper_step_mm"),
        "max_joint_lead_deg": one_value("max_joint_lead_deg"),
        "max_gripper_lead_mm": one_value("max_gripper_lead_mm"),
        **({
            "max_act_raw_joint_gap_deg": one_value(
                "max_act_raw_joint_gap_deg"
            ),
            "max_act_raw_gripper_gap_mm": one_value(
                "max_act_raw_gripper_gap_mm"
            ),
            "max_act_raw_gripper_bound_tolerance_mm": one_value(
                "max_act_raw_gripper_bound_tolerance_mm"
            ),
            "max_act_clamp_hold_steps": one_value(
                "max_act_clamp_hold_steps"
            ),
        } if policy_type in {"act", "pi0"} else {}),
        "gripper_max_mm": one_value("gripper_max_mm"),
        "gripper_force": one_value("gripper_force"),
        "max_camera_age_ms": one_value("max_camera_age_ms"),
    }

    def percentile(values, q):
        return values[min(len(values) - 1, round((len(values) - 1) * q))]

    return {
        "approved": True, "checkpoint_sha256": digests.pop(), "samples": len(rows),
        "policy_type": policy_type,
        "act_raw_action_watchdog_passed": (
            True if policy_type in {"act", "pi0"} else None
        ),
        "inference_ms_p95": percentile(inference, 0.95),
        "chunk_inference_ms_p95": chunk_inference_p95,
        "camera_age_ms_p95": percentile(camera, 0.95),
        "deadline_misses": miss_count,
        "deadline_miss_fraction": miss_fraction,
        "max_control_loop_ms": max(loop_ms) if loop_ms else None,
        "max_control_loop_periods": max(loop_periods) if loop_periods else None,
        "clamp_fraction": clamp_fraction,
        "severe_clamp_fraction": severe_clamp_fraction,
        "runtime": runtime,
        "source_report": str(report.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    from ..config import defaults
    from .online_policy import workspace_path

    cfg = defaults("modeling", "approve")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=workspace_path(cfg["output"]),
                        help="审批文件输出（默认由 YAML 管理）")
    parser.add_argument("--min-samples", type=int, default=cfg["min-samples"])
    parser.add_argument(
        "--max-smolvla-severe-clamp-fraction", type=float,
        default=cfg["max-smolvla-severe-clamp-fraction"],
        help=(
            "maximum fraction of SmolVLA rows whose clamp residual exceeds "
            "5 degrees or 5 mm"
        ),
    )
    parser.add_argument(
        "--max-control-deadline-miss-fraction", type=float,
        default=cfg["max-control-deadline-miss-fraction"],
        help=(
            "maximum fraction of control iterations allowed to exceed one "
            "control period"
        ),
    )
    parser.add_argument(
        "--max-control-loop-periods", type=float,
        default=cfg["max-control-loop-periods"],
        help=(
            "maximum duration of the longest control iteration, in control "
            "periods (20 Hz -> 2.0 periods = 100 ms)"
        ),
    )
    args = parser.parse_args(argv)
    approval = build_approval(
        args.report, args.output, args.min_samples,
        args.max_smolvla_severe_clamp_fraction,
        args.max_control_deadline_miss_fraction,
        args.max_control_loop_periods,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(approval, indent=2) + "\n")
    print(json.dumps(approval, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
