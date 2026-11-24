# ACT、HG-DAgger 与 HIL-SERL 数据流

## 当前整理后的RAW目录（2026-11-24）

已将分散session整理为以下独立数据组，位于`data/raw/piper_raw/`。HG、旧版HIL、
v2评估与v2训练保留各自来源和审核结果；合并未重新编码视频或改写NPZ。

| 目录 | 完整episode | 人工成功/失败标签 |
|---|---:|---|
| `merged_hg_20261116_pick_place_flvv` | 112 | 73成功、39失败 |
| `merged_hil_v1_20261117_pick_place_flvv` | 59 | 50成功、8失败、1条未标注 |
| `merged_hil_v2_eval_20261117_pick_place_flvv` | 10 | 10成功 |
| `merged_hil_v2_20261118_pick_place_flvv` | 40 | 29成功、11失败 |

独有的14条未完成轨迹保存在`archive/piper_raw_incomplete_20261124/`。
原始session和审核元数据备份、逐文件哈希与旧路径到新episode的对应关系保存在
`artifacts/reports/piper_raw_cleanup_20261124/audit.json`及同级`original_metadata/`。
原路径仍作为`merge_source`中的历史身份；HIL recover会沿用该身份，防止已提交试次因合并而重复入库。
评估组已有人工接管和在线replay提交记录，不能据此当作独立未见测试集。

## 统一审核入口

RAW 是长期保存的事实来源；BC 示教与 HG-DAgger 试次可以在同一个浏览器页面审核：

```bash
robotarm modeling dataset-browser -- \
  --dataset-root data/raw/piper_raw/BC_SESSION data/raw/piper_raw/hg_dagger_SESSION
```

页面显示每条轨迹来自哪个 session。审核决定仍分别写在各自的
`annotations/reviews.json`，绑定源 NPZ 内容；不创建第二套互相冲突的标签。
为 HIL replay 选择的完整 trial 应手工标为 `keep`，并标注 `task_outcome=success/failure`。
未手工确认的页面默认值不能当作 RL 奖励标签。

采集完整 ACT+人工接管 trial 时，可在现有 ACT 真机命令后加 `--hg-dagger
--hg-label-outcome`。X 结束试次后，按 `1` 标成功或 `0` 标失败；结果立即写回
同一个 RAW 审核 sidecar。手工复位物体后，R/Y 回零机械臂，再按 X 开始下一条。
试次没有人工结果时，程序不会启动下一条；急停与退出仍然可用。这个开关目前
负责 ACT 试次及结果标注，不会开启 RL 动作。这批试次可作为 DAgger 专家纠正数据
以及以后 RL 的历史 replay；**它不是 HIL-SERL 在线训练时由 RL actor 新采的试次**。
真正运行 HIL-SERL 时，要另开由 RL 策略控制的新 session，并继续记录人工接管和结果。
不同 session 可在同一 RAW 审核平台统一管理。
正在进行的HG trial可按`D`直接放弃；该条临时视频和NPZ会被删除，程序保持暂停，
必须R/Y回零后才能开始下一条。额外按`V`录制的评估视频是独立文件，不随HG trial删除。

一个完整 HG trial 包括模型、待接管、人工接管三个控制阶段。ACT 监督数据只使用
`expert_mask=1` 的连续片段；HIL replay 使用完整 trial 并保留控制来源、接管标记、
`reward` 和 `done`。普通 BC 示教整段是专家数据。
完整 HG trial 还导出 `complementary_info.novice_action`；只有
`complementary_info.novice_action_valid=true` 的已执行自主控制帧可把它当作 ACT 基线。
人工接管期间保存的上一条 ACT 动作不能当作该帧的基线。

## 两种按需导出

已有 BC LeRobot 数据集不需要重新解码或编码。只向新聚合集增量转换已审核的 HG
人工片段；旧数据集先以文件系统写时复制方式克隆到临时目录，不支持时自动回退为普通复制。
这个过程仍会编码**新增片段**的视频：

```bash
robotarm modeling extend-dataset -- \
  --mode new \
  --base-dataset data/lerobot/pick_up_flvv_20261115_curated \
  --new-raw data/raw/piper_raw/hg_dagger_SESSION \
  --output data/lerobot/pick_up_flvv_hg_round1 \
  --new-status review --new-label none

# 核对预检数量后，给同一命令添加 --execute
```

同一个审核 sidecar 也可供 HIL-SERL replay 使用。完整 HG trial 只在准备在线 RL
时转换一次；`--dry-run` 先统计待编码帧数：

```bash
robotarm modeling prepare-hil-serl -- \
  data/raw/piper_raw/hg_dagger_SESSION \
  --buffer-role replay --space joint --frame absolute \
  --output data/lerobot/hil_replay_round1 --dry-run
```

正式运行时去掉 `--dry-run`。普通 BC RAW 可用 `--buffer-role offline-demo`
导出到另一个 buffer。完整 HG trial 不能导入 offline-demo；其中模型动作不是专家动作。

若要准备“全部 HG 失败＋抽样 HG 成功＋抽样 BC 成功”的历史 replay seed，
可以向同一入口传入 HG 与 BC 的原始 session：

```bash
robotarm modeling prepare-hil-serl -- \
  data/raw/piper_raw/merged_hg_20261116_pick_place_flvv \
  data/raw/piper_raw/20261115_flvv_layer4_newview_v1/20261115_114407_135992 \
  --selection hg-failures-plus-success \
  --hg-success-count 30 --bc-success-count 50 --seed 1000 \
  --buffer-role replay --space joint --frame absolute \
  --output data/lerobot/flvv_hil_seed_round1 --dry-run
```

去掉 `--dry-run` 才编码输出。筛选只使用审核为 keep 且有成功／失败结果的完整试次，
保留全部合格 HG 失败，不选 BC 失败；成功数量不足时取全部可用试次，不重复抽样。
固定 seed 可复现选择；来源身份、实际数量及 seed 写入转换清单。源 session 和标注不修改。
混合数据只用于 replay：HG 保留全程动作与 expert_mask，BC 的全部帧标记为人工示范，
不能将整个混合集作为全专家 offline-demo。不要同时传入原始 session 及其合并副本。

若 learner 直接在本机读取 RAW，连 HG 的 LeRobot 导出也可以省去：

```python
from robotarm.modeling.raw_hil_replay import RawHILReplayDataset
from lerobot.rl.buffer import ReplayBuffer

dataset = RawHILReplayDataset("data/raw/piper_raw/hg_dagger_SESSION")
replay = ReplayBuffer.from_lerobot_dataset(
    dataset, device="cuda:0", storage_device="cpu",
    state_keys=["observation.state", "observation.images.global",
                "observation.images.wrist"], use_drq=False,
)
dataset.close()
```

RAW 读取器逐帧解码原 MP4，支持随机跳转，当前只面向单进程读取和 Piper 7 维绝对
关节动作。它在末帧补人工审核的稀疏奖励；失败末帧奖励为零。它没有替代在线奖励
采集和 SAC learner。

若现有 BC LeRobot 数据集的 `meta/piper_conversion.json` 已包含逐条人工审核的
`task_outcome`，可直接读取为 RL 离线示教，**不必重转 BC 视频**：

```python
from robotarm.modeling.reviewed_offline_rl import open_reviewed_offline_rl
from lerobot.rl.buffer import ReplayBuffer

dataset = open_reviewed_offline_rl("data/lerobot/pick_up_flvv_20261115_curated")
offline_buffer = ReplayBuffer.from_lerobot_dataset(
    dataset, device="cuda:0", storage_device="cpu",
    state_keys=["observation.state", "observation.images.global",
                "observation.images.wrist"], use_drq=False,
)
```

适配器只在读取时补上 `next.reward` 和 `next.done`，来源是审核过的成功/失败标签。
这仍会逐帧解码图像并占用 replay 内存，但不会重新编码或复制 MP4。

多 session 审核页面的“导出”会为每个 RAW session 分别生成一个 LeRobot 目录。
若已有 BC LeRobot 数据，优先使用上面的 `extend-dataset`，省去重转旧 BC 数据。

## ACT 加 RL 的训练与执行

`robotarm modeling hil-serl` 已提供冻结 ACT 的残差 SAC learner、复用原有控制门控的
Piper actor、标注结果后提交的在线 replay、试次边界权重同步和 CPU 自检。
`scripts/run_hg_then_hil.sh` 可顺序执行 HG 微调、ACT 特征缓存及 HIL learner。
命令和算法约定见 [HIL_SERL.md](HIL_SERL.md)。

任务结果与物体复位仍由人工完成。转换数据、离线更新和 CPU 测试不等于真机闭环验证；
首次上线先运行 actor shadow，再验证短试次的接管、标注和权重同步。
