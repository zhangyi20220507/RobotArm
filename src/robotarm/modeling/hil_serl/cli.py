"""Piper HIL residual SAC preparation, training, actor and CPU verification."""
import argparse
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="cache ACT features from reviewed replay; preview by default")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--act-checkpoint", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--seed", type=int, default=1000)
    prepare.add_argument("--batch-size", type=int, default=8, help="offline ACT encoding batch size; live actor remains single-frame")
    prepare.add_argument("--temporal-ensemble-coeff", type=float, default=0.01,
                         help="ACT execution reference; v2 caches encode frames sequentially")
    prepare.add_argument("--max-joint-delta-deg", type=float, default=1.0)
    prepare.add_argument("--max-gripper-delta-mm", type=float, default=2.0)
    prepare.add_argument("--gripper-max-mm", type=float, default=100.0)
    prepare.add_argument("--execute", action="store_true", help="encode the offline dataset; never connects hardware")
    learner = commands.add_parser("learner", help="offline warm-up, then train as labelled online trials arrive")
    learner.add_argument("--run-dir", type=Path, required=True)
    learner.add_argument("--device", default="cuda")
    learner.add_argument("--steps", type=int, default=100000)
    learner.add_argument("--offline-steps", type=int, default=1000)
    learner.add_argument("--updates-per-transition", type=int, default=1)
    learner.add_argument("--publish-every", type=int, default=100)
    learner.add_argument("--save-every", type=int, default=1000)
    learner.add_argument("--cpu-threads", type=int, default=4)
    learner.add_argument("--resume", action="store_true")
    recovery = commands.add_parser("recover", help="recover a completed RAW trial after outcome submission failed")
    recovery.add_argument("--run-dir", type=Path, required=True)
    recovery.add_argument("--raw-episode", type=Path, required=True)
    recovery.add_argument("--outcome", choices=("success", "failure"), required=True)
    recovery.add_argument("--device", default="cuda")
    recovery.add_argument("--execute", action="store_true")
    actor = commands.add_parser("actor", help="run the existing guarded controller with residual SAC")
    actor.add_argument("--run-dir", type=Path, required=True)
    actor.add_argument("--execute", action="store_true", help="authorize real arm control; starts paused until X")
    actor.add_argument("--deterministic", action="store_true", help="evaluate the residual mean without exploration")
    actor.add_argument("--mode", choices=("baseline", "residual", "explore"), default="baseline",
                       help="baseline: original ACT only (default); residual: deterministic SAC; explore: stochastic SAC")
    actor.add_argument("--minimum-updates", type=int, default=100)
    actor.add_argument("online_args", nargs=argparse.REMAINDER, help="online-policy options after --")
    smoke = commands.add_parser("smoke", help="CPU SAC/actor/replay integration test; no cameras or CAN")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--steps", type=int, default=20)
    sequential = commands.add_parser("pipeline", help="HG ACT fine-tune -> frozen features -> HIL learner; preview by default")
    sequential.add_argument("--act-dataset", type=Path, required=True)
    sequential.add_argument("--hil-dataset", type=Path, required=True)
    sequential.add_argument("--act-checkpoint", type=Path, required=True)
    sequential.add_argument("--run-dir", type=Path, required=True)
    sequential.add_argument("--device", default="cuda")
    sequential.add_argument("--act-steps", type=int, default=20000)
    sequential.add_argument("--act-checkpoint-selection", choices=("best-validation", "last"), default="best-validation")
    sequential.add_argument("--hil-steps", type=int, default=100000)
    sequential.add_argument("--offline-steps", type=int, default=1000)
    sequential.add_argument("--execute", action="store_true", help="launch training jobs; never launches the physical actor")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from ...paths import WORKSPACE_ROOT
    cache = WORKSPACE_ROOT / ".cache/huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    if args.command == "prepare":
        from .prepare import prepare
        return prepare(args)
    if args.command == "learner":
        from .config import Config
        if Config.load(args.run_dir / "config.json").format_version.endswith("v1"):
            raise ValueError("legacy stateless HIL run: preserve it for analysis; prepare a new v2 run before training")
        from .learner import learn
        return learn(args)
    if args.command == "recover":
        from .recover import recover
        return recover(args)
    if args.command == "smoke":
        from .smoke import smoke
        return smoke(args)
    if args.command == "pipeline":
        from .pipeline import pipeline
        return pipeline(args)
    from .config import Config
    from ..online_policy import main as online_main

    cfg = Config.load(args.run_dir / "config.json")
    extra = args.online_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    reserved = {"--checkpoint", "--hil-run", "--execute", "--no-execute", "--hil-min-updates",
                "--hil-deterministic", "--hil-explore", "--n-action-steps", "--temporal-ensemble-coeff"}
    if any(token.split("=", 1)[0] in reserved for token in extra):
        raise ValueError("checkpoint, HIL run and execute mode must be set through the HIL actor command")
    coefficient = cfg.temporal_ensemble_coeff if cfg.temporal_ensemble_coeff is not None else 0.01
    online = [*extra, "--checkpoint", cfg.act_checkpoint,
              "--n-action-steps", "1", "--temporal-ensemble-coeff", str(coefficient)]
    if args.mode == "baseline":
        print("[HIL] BASELINE evaluation: original ACT execution, no SAC residual or learner submission.", flush=True)
    else:
        if cfg.format_version.endswith("v1"):
            raise ValueError("old HIL weights/cache use a different ACT reference; use --mode baseline or prepare a new v2 run")
        if args.mode == "explore" and args.deterministic:
            raise ValueError("--mode explore cannot be combined with --deterministic")
        online += ["--hil-run", str(args.run_dir), "--hil-min-updates", str(args.minimum_updates)]
        if args.mode == "residual":
            online.append("--hil-deterministic")
        else:
            online.append("--hil-explore")
    if args.execute:
        online += ["--execute", "--hg-dagger", "--hg-label-outcome", "--hg-task", cfg.task]
    return online_main(online)


if __name__ == "__main__":
    raise SystemExit(main())
