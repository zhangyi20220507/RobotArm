"""Run shadow telemetry and automatically approve a checkpoint if it passes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--camera-serial", default=None,
                        help="override the global camera serial during shadow validation")
    parser.add_argument("--wrist-camera-serial", default=None,
                        help="override the wrist camera serial during shadow validation")
    parser.add_argument("--interface", default=None,
                        help="override the robot connection interface")
    parser.add_argument("--channel", default=None,
                        help="override the robot CAN channel")
    parser.add_argument(
        "--firmware", choices=("default", "v183", "v188", "v189"),
        default=None, help="override the Piper firmware API variant",
    )
    parser.add_argument("--feedback-timeout", type=float, default=None,
                        help="override the robot feedback timeout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from . import approve, online_policy

    args = parse_args(argv)
    if args.seconds <= 0 or args.min_samples < 1:
        raise RuntimeError("seconds and min-samples must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    # Import built-in policy configs before resolving the checkpoint's type.
    # Importing the config base class alone leaves LeRobot's registry empty.
    import lerobot.policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig

    policy_type = PreTrainedConfig.from_pretrained(checkpoint).type
    if policy_type == "pi05":
        raise RuntimeError(
            "Pi0.5 base cannot receive execution approval. Run "
            "online-policy without --execute for shadow telemetry."
        )
    if policy_type == "pi0":
        from ..paths import ARTIFACTS_ROOT
        base = (args.base_checkpoint or ARTIFACTS_ROOT / "pretrained/pi0_base").resolve()
        digest = online_policy.pi0_checkpoint_digest(checkpoint, base)
    else:
        digest = online_policy.checkpoint_digest(checkpoint)
    report, _, approval = online_policy.runtime_artifact_paths(checkpoint, digest)
    # The standard runtime is 20 Hz. Keep a time margin for scheduling jitter;
    # users may request a longer observation window explicitly.
    seconds = max(args.seconds, math.ceil(args.min_samples / 20.0) + 5.0)
    print(f"Runtime validation: {seconds:.0f}s shadow run; no motion commands will be sent.")
    shadow_args = [
        "--checkpoint", str(checkpoint),
        "--report", str(report),
        "--duration", str(seconds),
    ]
    if args.base_checkpoint is not None:
        shadow_args.extend(("--base-checkpoint", str(args.base_checkpoint)))
    if args.camera_serial is not None:
        shadow_args.extend(("--camera-serial", args.camera_serial))
    if args.wrist_camera_serial is not None:
        shadow_args.extend(("--wrist-camera-serial", args.wrist_camera_serial))
    for flag, value in (
        ("--interface", args.interface),
        ("--channel", args.channel),
        ("--firmware", args.firmware),
        ("--feedback-timeout", args.feedback_timeout),
    ):
        if value is not None:
            shadow_args.extend((flag, str(value)))
    result = online_policy.main(shadow_args)
    if result != 0:
        return result
    print("Shadow run complete; checking telemetry and creating approval.")
    result = approve.main([
        "--report", str(report),
        "--output", str(approval),
        "--min-samples", str(args.min_samples),
    ])
    if result == 0:
        print(f"Runtime validation passed. Approval: {approval}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
