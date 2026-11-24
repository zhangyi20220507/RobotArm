# Piper ACT 残差 HIL-SERL

入口：`robotarm modeling hil-serl -- prepare|learner|actor|pipeline|smoke`。
这是面向本项目 Piper 的 HIL 残差 SAC 实现：冻结 ACT，复用其双相机视觉骨干，
SAC 学习受限的 7 维关节／夹爪连续动作修正。人工接管、人工结果和物体复位沿用现有流程。
它采用双 Q、目标网络、自动熵温度、约 1:1 专家与 replay 采样以及异步 actor/learner。
本实现使用连续夹爪与固定 ACT 视觉特征；这些是本项目的实现选择。
[原始 HIL-SERL 项目](https://github.com/rail-berkeley/hil-serl)提供算法背景。

当前先评估原 ACT 基线：`actor --mode baseline` 是默认模式，不加载残差或提交在线 replay。
旧 `hil_act010000` v1 learner 已保存并停止；其缓存与新执行参考不兼容，保留用于分析。
当前完整命令见 [README 的运行与续训](README.md#hil-serl当前010000模型的运行与续训)。

## 一次性 CPU 自检

```bash
robotarm modeling hil-serl -- smoke \
  --output artifacts/logs/hil_serl_smoke_new --steps 20
```

该命令不连接相机或 CAN。它验证学习更新、replay 混合、原子权重发布、残差界限及续训，
结果写入输出目录的 `smoke_result.json`。自检模型绑定虚拟 ACT，不能用于物理 actor。

## 顺序执行 HG 微调与 HIL 学习

前提是两个数据集都已完成转换，目录没有 `CONVERSION_INCOMPLETE.txt`：

- `pick_up_flvv_bc_hg_round1`：原 BC 加已审核 HG 人工片段，供 ACT 监督训练。
- `flvv_hil_seed_round1`：BC/HG 成功及 HG 失败的完整试次，包含 reward/done/接管标记。

先预览：

```bash
scripts/run_hg_then_hil.sh \
  --act-dataset data/lerobot/pick_up_flvv_bc_hg_round1 \
  --hil-dataset data/lerobot/flvv_hil_seed_round1 \
  --act-checkpoint artifacts/checkpoints/20261115_act_flvv_267eps_layer4_chunk50_bs16_160k_v1/checkpoints/160000/pretrained_model \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1 \
  --act-steps 20000 --hil-steps 100000 --offline-steps 1000
```

添加 `--execute` 后依次：

1. 从旧 ACT 权重初始化新训练任务，进行 HG 聚合数据微调。
2. 从本次日志中选择已保存且验证 loss 最低的 ACT，重算完整 replay 每帧的 ACT 基线及视觉特征。选择结果写入 `act_checkpoint_selection.json`；可显式传 `--act-checkpoint-selection last` 改用最终检查点。
3. 启动 SAC learner，先更新 1000 步，随后等待有结果的新在线试次；每条 transition 默认允许一次更新。

后续阶段仅在上阶段退出码为零时执行。流水线不自动连接机械臂；要在另一个终端启动 actor。
目录必须是新的；某阶段中断后，用下面的独立入口恢复对应阶段，避免误覆盖旧训练。
`--act-steps` 是这次新微调任务的优化步数，`--hil-steps` 是 SAC 的总优化步数上限。

## 单独准备特征与训练

如果 HG 微调已完成，可以跳过流水线第一步：

```bash
robotarm modeling hil-serl -- prepare \
  --dataset data/lerobot/flvv_hil_seed_round1 \
  --act-checkpoint artifacts/checkpoints/act_flvv_bc_hg_round1_finetune/checkpoints/last/pretrained_model \
  --run-dir artifacts/checkpoints/flvv_hil_round1 \
  --device cuda
```

默认仅预览，添加 `--execute` 才提取特征。无需相机或机械臂。
默认残差上限：每个关节 1°、夹爪 2 mm，可在准备时通过
`--max-joint-delta-deg`、`--max-gripper-delta-mm` 设置。物理控制的限位、加速度、步长及超前
限制仍然生效，残差上限不替代这些限制。

```bash
robotarm modeling hil-serl -- learner \
  --run-dir artifacts/checkpoints/flvv_hil_round1 \
  --device cuda --steps 100000 --offline-steps 1000
```

初始离线更新预算用完后，learner 等待新标注试次，避免在固定历史数据上无限执行普通 SAC 更新。
每秒检查新 episode，默认每 100 步发布 actor，每 1000 步保存完整训练状态。
同一目录只有一个 learner 可持锁运行。Ctrl+C 保存后退出；续训在同一命令中添加 `--resume`。
可用 `--updates-per-transition` 调整更新预算；它不是成功／失败采样比例。

## 真机 actor

原 ACT 基线无需启动 learner，也无需 `policy.pt`。残差模式要求新 v2 run，并先由 learner
产生至少 100 次更新的 `policy.pt`。先运行只读基线 shadow：

```bash
robotarm modeling hil-serl -- actor \
  --run-dir artifacts/checkpoints/flvv_hil_round1 \
  --mode baseline -- --duration 30
```

Shadow 仍读取相机和从臂反馈，发送运动命令的门控始终关闭。
以下明确使用确定性残差（要求 v2 run）；原 ACT 基线将 `--mode residual` 改为 `--mode baseline`：

```bash
robotarm modeling hil-serl -- actor \
  --run-dir artifacts/checkpoints/flvv_hil_round1 \
  --mode residual --execute -- \
  --hg-leader-interface socketcan --hg-leader-channel can_leader1 \
  --hg-setup-id flvv_hil_round1
```

流水线产生的 HIL run 是 `artifacts/checkpoints/flvv_hg_then_hil_round1/hil`，使用流水线时
将上面的 `--run-dir` 换为该目录。actor 自动绑定准备阶段的 ACT 检查点和任务。
actor 默认 `--mode baseline` 严格使用原 ACT 执行链，不加载 SAC。
`--mode residual` 使用残差均值；只有显式 `--mode explore` 才随机采样。
旧 `--deterministic` 仍兼容，但不能与 `--mode explore` 同用。

按键：

| 输入 | 行为 |
|---|---|
| X | 准备完成后启动；再次 X 保存完整试次并等待结果 |
| I | 人工接管；再次 I 交还 ACT+SAC |
| 1 / 0 | 将刚结束的试次标为成功／失败，并提交给 learner |
| R / 长按 Y | 回零；需人工复位物体，然后 X 开始下一条 |
| D | 丢弃本条；不进入 learner |
| Ctrl+C / Alt+S | 沿用原控制程序的退出回零／急停流程 |

新权重只在按 X 开始下一条之前加载，当前试次固定一个版本。人工接管完全绕过 RL 输出。
相同版本不重复加载；新权重的 SAC 头在暂停状态下预热，启动预热也覆盖完整 ACT＋SAC。
相机取帧年龄默认上限为 50 ms；包含推理的观测到命令总年龄通过独立的
`--hil-max-action-age-ms` 检查，默认 100 ms。两项都满足才允许动作，避免将取帧预算误用于整条推理链。
若仅观测到命令总年龄超限，模型控制和人工接管都丢弃该帧，不发送新指令，也不写入 RAW／replay；
保留最后安全目标，下一轮获取新观测、重新推理，不执行过期动作。提示最多每秒一次。
连续第 3 帧超限仍停止当前试次，需要人工重新按 X；不会在真正停止后自动重启。
相机取帧过期、反馈／模式／关节异常、推理超过硬上限仍立即停止，不进入慢帧重试。
人工夹爪接管保持起始宽度连续，到达开合边界后更新夹爪离合偏置，避免多次 I 接管后
从臂只能在偏置附近开合。关节离合偏置保持原逻辑，所有指令仍经过步长、超前和加速度限幅。
每帧保存的是经过所有限幅后实际发送的动作，而不是未执行的策略建议。
在线 RAW 使用 `hil_serl_` 前缀，保留完整视频和接管来源；不能误认为旧 ACT 的 HG 采集。

### 标注或提交中断后的恢复

如果 X 已保存完整 RAW，但 1/0 标注或向 learner 提交时进程异常退出，可以从视频与
NPZ 重建该试次的 ACT 特征，并补交结果。下例的目录需替换为对应的 run 和已完成 episode：

```bash
robotarm modeling hil-serl -- recover \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1/hil_act010000 \
  --raw-episode data/raw/piper_raw/SESSION/episodes/episode_000000 \
  --outcome failure
```

默认只预检；添加 `--execute` 才重建特征、保存结果并提交。无相机或 CAN 连接。
只有同一 HIL 配置绑定的完整非空 RAW 可以恢复；已有相反的人工标注会被拒绝。
以 RAW 路径生成稳定提交 ID，重复执行不会重复入库。原视频和 NPZ 保留，
缺失的人工结果写入原 RAW 的统一审核 sidecar；无法还原的策略版本记为未知。

## 数据与训练约定

- v2 ACT 参考使用原 `select_action` 时间集成（默认 0.01），以单帧、episode 顺序准备缓存；
  episode 与接管／交还边界重置时间集成。现场安全等待冻结时间轴，丢弃慢帧则撤销时间集成更新。
  旧 v1 的无时间集成首动作参考只保留用于历史分析，不再允许继续训练或执行残差。
- v2 RAW 保存每帧实际 ACT 参考；异常补交使用记录值恢复，避免丢失安全等待的时间轴信息。
- 观测包含 ACT 冻结视觉特征（每路池化至 2×2）、7 维关节状态、7 维 ACT 参考目标。
- critic 的动作是实际命令的绝对关节／夹爪目标。专家动作即使超出残差范围也保留原值，
  不通过裁剪伪造已执行动作。actor 始终受残差范围限制。
- BC 全部帧是专家；HG 只有专家帧进入额外的专家采样池，完整 HG 都能进入 replay。
- 成功试次最后一个已执行动作 reward=1，其余为 0；失败全部为 0。done 只在末 transition 为真。
  无命令的帧不生成虚假动作 transition；下一状态不跨 episode。
- X 保存后仍未标结果、D 丢弃、watchdog/超时/退出中断的数据均不提交在线 learner。
- `config.json`、缓存及权重绑定 ACT 权重、配置、归一化器摘要与相机顺序。换 ACT 后重新建 run。
- 同机进程通过 `online/episode_*.npz` 原子提交试次，通过 `policy.pt` 原子同步权重；没有网络服务。
- 反馈过期、非有限动作、USB-CAN 重新枚举都会阻止物理动作；异常停止不会在清理时重新使能主臂。

本模块的 CPU 测试和真实 ACT 离线特征检查不代表真机闭环已经验证，也不保证训练收敛。
首次执行应先确认 shadow 延迟和反馈检查通过，再开始短试次验证。
