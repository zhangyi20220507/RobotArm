"""Sequential HG fine-tuning -> ACT feature preparation -> HIL learner."""
from pathlib import Path
import shlex
import subprocess
import sys


def select_act_checkpoint(act_run, mode="best-validation"):
    """Pick an actually evaluated saved checkpoint; never infer best from last."""
    if mode == "last":
        return (act_run / "checkpoints/last/pretrained_model").resolve(), {"selection": "last"}
    from ..training_report import read_run_history, summarize_history
    history, sources = read_run_history(act_run)
    summary = summarize_history(history, act_run)
    if not summary["recommended_checkpoint"]:
        raise ValueError("no saved ACT checkpoint has validation loss; cannot choose the HIL prior")
    summary.update(selection="best-validation", history_sources=sources)
    return Path(summary["recommended_checkpoint"]).resolve(), summary


def commands(args):
    root = args.run_dir.resolve()
    act = root / "act"
    hil = root / "hil"
    checkpoint = act / "checkpoints/last/pretrained_model"
    base = [sys.executable, "-m", "robotarm"]
    train = [*base, "modeling", "train", "--", str(args.act_dataset.resolve()), str(act), "act",
             "--policy.path=" + str(args.act_checkpoint.resolve()), "--policy.device=" + args.device,
             "--quality-filter", "--quality-shuffle-seed", "1000",
             "--dataset.eval_split=0.1", "--eval_steps=1000", "--max_eval_samples=0",
             "--batch_size=16", f"--steps={args.act_steps}", "--save_freq=2000", "--seed=1000"]
    prepare = [*base, "modeling", "hil-serl", "--", "prepare", "--dataset", str(args.hil_dataset.resolve()),
               "--act-checkpoint", str(checkpoint), "--run-dir", str(hil), "--device", args.device, "--execute"]
    learner = [*base, "modeling", "hil-serl", "--", "learner", "--run-dir", str(hil),
               "--device", args.device, "--steps", str(args.hil_steps),
               "--offline-steps", str(args.offline_steps)]
    return [train, prepare, learner], hil


def pipeline(args):
    if args.act_steps < 1 or args.hil_steps < 1 or args.offline_steps < 1:
        raise ValueError("training step counts must be positive")
    if args.run_dir.exists():
        raise FileExistsError("pipeline requires a new run directory; resume individual stages explicitly")
    for root in (args.act_dataset, args.hil_dataset):
        if not (root / "meta/info.json").is_file():
            raise ValueError(f"dataset is not ready: {root}")
    if not (args.act_checkpoint / "model.safetensors").is_file():
        raise ValueError("ACT initialization checkpoint is missing")
    from .prepare import inspect_dataset
    inspect_dataset(args.hil_dataset)
    steps, hil = commands(args)
    for index, command in enumerate(steps, 1):
        print(f"Stage {index}: {shlex.join(command)}", flush=True)
    print("After ACT training, the preparation checkpoint is resolved using "
          + getattr(args, "act_checkpoint_selection", "best-validation") + ".", flush=True)
    print("Separate physical actor (only after learner publishes weights):\n"
          + shlex.join([*steps[0][:3], "modeling", "hil-serl", "--", "actor",
                        "--run-dir", str(hil), "--execute"]), flush=True)
    if not args.execute:
        print("PREVIEW: no data writes, training or hardware. Add --execute to start the sequential training jobs.")
        return 0
    for index, command in enumerate(steps):
        if index == 1:
            from .config import atomic_json
            selected, summary = select_act_checkpoint(
                args.run_dir.resolve() / "act", getattr(args, "act_checkpoint_selection", "best-validation")
            )
            if not (selected / "model.safetensors").is_file():
                raise FileNotFoundError(f"selected ACT weights missing: {selected}")
            command[command.index("--act-checkpoint") + 1] = str(selected)
            atomic_json(args.run_dir / "act_checkpoint_selection.json", summary)
            print(f"HIL ACT prior selected: {selected}", flush=True)
        subprocess.run(command, check=True)
    return 0
