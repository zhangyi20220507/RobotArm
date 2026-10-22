"""模型训练、评估与推理。

Python 工作流（robotarm.modeling / robotarm.record 包内实现）：
- check_environment：统一运行环境与训练依赖自检
- evaluate：checkpoint 离线评估
- online_policy：键盘控制的在线策略执行（含 shadow 门控）
- trajectory_studio：浏览器中的 open-loop 模型验证平台
- dataset_browser：本地多 episode 数据质量浏览与标记
- validate_runtime：自动完成 shadow 遥测与 checkpoint 绑定审批
- normalize_metadata：piper_pick_up 数据集任务/装定归一化
- pi05_offline：使用本地数据集做完全离线的 Pi0.5 延迟测试
- approve：shadow 报告审批，生成 checkpoint 绑定的批准文件
- convert-dataset：piper_raw 会话转 LeRobot 数据集（robotarm.record.convert）

Bash 编排工作流（需先激活 robotarm 环境）：
- train：训练
- train-with-holdout：带留出集训练
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..legacy import run_python_module, run_shell_script
from ..paths import WORKSPACE_ROOT
from ..workflows import WorkflowSpec

MODELING_SCRIPTS = Path(__file__).resolve().parent / "scripts"


def _run_module(module: str):
    def handler(extra_args: Sequence[str] | None = None) -> int:
        return run_python_module(module, extra_args)

    return handler


def _run_bash(script_name: str):
    def handler(extra_args: Sequence[str] | None = None) -> int:
        # Run from the workspace root so relative dataset/output paths and
        # wandb offline runs resolve as documented in the README.
        return run_shell_script(
            MODELING_SCRIPTS / script_name, extra_args, cwd=WORKSPACE_ROOT
        )

    return handler


def _run_dataset_browser(extra_args: Sequence[str] | None = None) -> int:
    from .dataset_browser import main

    return main(list(extra_args or []))


def _run_online_policy(extra_args: Sequence[str] | None = None) -> int:
    # Handle Ctrl+C once in the policy safety state machine rather than in a
    # wrapper process and its child at the same time.
    from .online_policy import main

    return main(list(extra_args or []))


def _run_validate_runtime(extra_args: Sequence[str] | None = None) -> int:
    from .validate_runtime import main

    return main(list(extra_args or []))


def _run_trajectory_studio(extra_args: Sequence[str] | None = None) -> int:
    from .trajectory_studio import main

    return main(list(extra_args or []))


WORKFLOWS = {
    "pi0-training-report": WorkflowSpec("Compare Pi0 training logs and sampled offline predictions.", _run_module("robotarm.modeling.reports.pi0_training")),
    "pi0-denoising-report": WorkflowSpec("Compare paired Pi0 5/10-step denoising offline and write HTML.", _run_module("robotarm.modeling.reports.pi0_denoising")),
    "act-training-report": WorkflowSpec("Combine ACT resume history and matched latent reports into HTML.", _run_module("robotarm.modeling.reports.act_training")),
    "train-pi0-lora": WorkflowSpec("Preview or launch isolated Pi0 LoRA training on a local Piper dataset.", _run_module("robotarm.modeling.pi0_lora")),
    "check-pi0-lora": WorkflowSpec("Reload a Pi0 LoRA adapter and predict one offline dataset frame.", _run_module("robotarm.modeling.pi0_lora_check")),
    "act-latent-report": WorkflowSpec("Compare ACT posterior/zero latent L1 on train/validation and write HTML.", _run_module("robotarm.modeling.act_latent_report")),
    "train": WorkflowSpec("Train a policy with the Lerobot pipeline.", _run_bash("train.bash")),
    "train-with-holdout": WorkflowSpec("Train with a holdout split.", _run_bash("train_with_holdout.bash")),
    "training-report": WorkflowSpec("Generate loss curves and a local training report.", _run_module("robotarm.modeling.training_report")),
    "convert-dataset": WorkflowSpec("Convert a raw Piper session into a Lerobot dataset.", _run_module("robotarm.record.convert")),
    "check-environment": WorkflowSpec("Check the training environment.", _run_module("robotarm.modeling.check_environment")),
    "evaluate": WorkflowSpec("Evaluate a checkpoint.", _run_module("robotarm.modeling.evaluate")),
    "online-policy": WorkflowSpec("Run the online policy loop.", _run_online_policy),
    "pi05-offline-benchmark": WorkflowSpec(
        "Benchmark Pi0.5 on one local dataset frame without hardware I/O.",
        _run_module("robotarm.modeling.pi05_offline"),
    ),
    "pi-offline-benchmark": WorkflowSpec(
        "Benchmark Pi0 or Pi0.5 on one local dataset frame without hardware I/O.",
        _run_module("robotarm.modeling.pi05_offline"),
    ),
    "validate-runtime": WorkflowSpec("Validate runtime telemetry and create approval.", _run_validate_runtime),
    "trajectory-studio": WorkflowSpec("Launch the open-loop validation studio.", _run_trajectory_studio),
    "dataset-browser": WorkflowSpec("Browse, review and trim episodes into a new local LeRobot dataset.", _run_dataset_browser),
    "normalize-piper-metadata": WorkflowSpec("Normalize Piper pick-up metadata.", _run_module("robotarm.modeling.normalize_metadata")),
    "approve": WorkflowSpec("Approve a shadow report and bind it to a checkpoint.", _run_module("robotarm.modeling.approve")),
}
