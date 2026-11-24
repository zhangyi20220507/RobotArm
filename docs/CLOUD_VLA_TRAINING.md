# π0 / SmolVLA 双云服务器训练包

两个展开目录使用同一环境，只是 `job.env` 中的 `POLICY`、基础模型目录、batch size 和输出目录不同。
目录只包含代码和配置，不包含数据、基础权重、checkpoint 或账号凭据。可以直接用 `rsync`/`scp`
上传整个目录，不要求压缩。

## 1. 云服务器要求

- Linux x86_64、Python 3.12、`python3.12-venv`、git、足够的本地SSD空间。
- NVIDIA GPU及能运行CUDA 12.8版PyTorch的驱动；以 `nvidia-smi` 和环境自检结果为准。
- π0默认LoRA、batch 1，建议至少24GB显存；显存不足不能靠环境脚本保证解决。
- SmolVLA默认冻结VLM并训练完整动作专家、batch 4；若OOM先把 `BATCH_SIZE` 降到2或1，仍不足时
  将 `TRAINING_MODE` 改为 `lora`。
- 首次下载基础模型需要访问Hugging Face。π0依赖的PaliGemma可能要求先在网页接受许可并执行
  `hf auth login`；不要把token写入压缩包。

两台服务器必须使用同一份压缩包版本和同一套安装命令。不要直接复用另一台机器打包的虚拟环境。

## 2. 直接上传目录并创建通用环境

```bash
# 在本地RobotArm/artifacts/cloud_exports中执行；替换云服务器地址和目录
rsync -a --info=progress2 \
  RobotArm_vla_cloud_20261115_pi0/ \
  USER@SERVER:/path/RobotArm_vla_cloud_20261115_pi0/

# 登录π0服务器
cd RobotArm_vla_cloud_20261115_pi0
bash scripts/cloud_vla/setup_env.sh
source scripts/cloud_vla/cloud_env.sh
```

SmolVLA服务器把文件名和目录名换成 `smolvla`，其余命令相同。安装器固定：

- Python 3.12虚拟环境 `.venv-cloud`
- PyTorch 2.11.0、torchvision 0.26.0、CUDA 12.8 wheel
- LeRobot 0.6.1
- π0、SmolVLA、PEFT、Transformers和数据集依赖

如果云平台必须使用其他官方PyTorch CUDA wheel，先修改 `PYTORCH_INDEX_URL`；同时需要修改并验证
与之匹配的Torch、torchvision和torchao版本，不要只替换一个包。当前两个包以默认锁定版本为验收口径。

如果上传工具要求单文件，可以在本地**按需**压缩；这不是训练必需步骤：

```bash
cd /home/dm/RobotArm/artifacts/cloud_exports
tar -czf RobotArm_vla_cloud_20261115_pi0.tar.gz RobotArm_vla_cloud_20261115_pi0
sha256sum RobotArm_vla_cloud_20261115_pi0.tar.gz \
  > RobotArm_vla_cloud_20261115_pi0.tar.gz.sha256

tar -czf RobotArm_vla_cloud_20261115_smolvla.tar.gz RobotArm_vla_cloud_20261115_smolvla
sha256sum RobotArm_vla_cloud_20261115_smolvla.tar.gz \
  > RobotArm_vla_cloud_20261115_smolvla.tar.gz.sha256
```

云端解压前先在压缩包所在目录校验：

```bash
sha256sum -c RobotArm_vla_cloud_20261115_pi0.tar.gz.sha256
tar -xzf RobotArm_vla_cloud_20261115_pi0.tar.gz
```

## 3. 下载各自基础模型

π0服务器：

```bash
source scripts/cloud_vla/cloud_env.sh
python scripts/cloud_vla/prepare_assets.py --policy pi0
```

SmolVLA服务器：

```bash
source scripts/cloud_vla/cloud_env.sh
python scripts/cloud_vla/prepare_assets.py --policy smolvla
```

基础模型保存在包内 `artifacts/pretrained/`，VLM/tokenizer进入包内 `.cache/huggingface/`。
下载完成后可设置 `export ROBOTARM_OFFLINE=1` 再重新 source 环境脚本，验证训练不再访问网络。

## 4. 后续放入同一份数据集

把最终转换完成的整个LeRobot目录分别上传为：

```text
data/lerobot/task_dataset/
├── meta/
├── data/
└── videos/
```

不要只上传视频或parquet。两台服务器的数据目录内容应完全一致；推荐在本地和两台云端分别保存并核对：

```bash
find data/lerobot/task_dataset -type f -print0 | sort -z | xargs -0 sha256sum > dataset.sha256
python scripts/cloud_vla/preflight.py --dataset-root data/lerobot/task_dataset
```

自检要求Piper绝对关节表示、7维state/action、`global + wrist` 双相机，以及非空的语言任务
`meta/tasks.parquet`。它会打印实际episode数和任务文字；应与你清洗导出的内容一致，不要只假定必然是270。

## 5. 先试跑，再正式训练

先打开 `job.env` 检查路径、chunk、步数和batch。默认是固定seed、10% episode留出、chunk 50、
30,000步，每2,000步保存和验证。它是第一轮可比较基线，不代表最优超参数。

```bash
# 只检查训练计划，不加载模型
bash scripts/cloud_vla/run_smoke.sh

# 实际执行20步、保存并验证；确认显存、loss和checkpoint完整
bash scripts/cloud_vla/run_smoke.sh --execute

# 正式训练仍先dry-run核对，再启动
bash scripts/cloud_vla/run_train.sh
bash scripts/cloud_vla/run_train.sh --execute
```

π0使用预训练基础权重上的LoRA；SmolVLA默认采用其官方结构的动作专家微调，冻结视觉/语言主干，
也可切换为LoRA。启动器会验证基础权重、Piper 7维输出和实际可训练参数，拒绝意外训练VLM主干。
如果正式输出目录或同名launch manifest已经存在，脚本会拒绝覆盖；更换 `OUTPUT_DIR` 后重新运行。

## 6. 后台守护运行、停止和断点续训

先在前台完成20步smoke test，确认能够保存checkpoint，再使用后台服务。后台启动正式训练：

```bash
bash scripts/cloud_vla/service.sh start
bash scripts/cloud_vla/service.sh status
bash scripts/cloud_vla/service.sh logs
```

`logs` 使用 `tail -f`，按 `Ctrl-C` 只会退出日志查看，不会停止训练。训练日志保存在
`.runtime/train.log`，PID保存在 `.runtime/train.pid`。

优雅停止训练：

```bash
bash scripts/cloud_vla/service.sh stop
```

它向整个训练进程组发送 `SIGINT` 并等待最多30秒，不会删除已有checkpoint。当前保存点之后、下一个保存点
之前的更新可能丢失，因此不要把“停止进程”当成“立即保存”。只有普通停止确实无效、且已经检查日志时才用：

```bash
bash scripts/cloud_vla/service.sh stop-force
```

从某个完整checkpoint继续时，在 `job.env` 增加或修改以下一行；路径必须指向
`pretrained_model/train_config.json`，同级 `training_state/` 必须保留：

```bash
RESUME_CONFIG=artifacts/checkpoints/RUN/checkpoints/020000/pretrained_model/train_config.json
```

确认 `job.env` 中 `STEPS` 是希望达到的**总步数**，然后后台续训：

```bash
bash scripts/cloud_vla/service.sh resume
bash scripts/cloud_vla/service.sh status
bash scripts/cloud_vla/service.sh logs
```

例如从20,000步继续到30,000步，`STEPS=30000`，不是 `10000`。不要同时启动 `start` 和 `resume`；
服务脚本检测到已有训练PID会拒绝第二个进程。

## 7. 网页监控与日志监控

网页监控可与训练独立后台运行：

```bash
bash scripts/cloud_vla/service.sh monitor-start
bash scripts/cloud_vla/service.sh monitor-status
bash scripts/cloud_vla/service.sh monitor-logs
```

默认监听 `8420` 端口。若云服务器不直接开放端口，在本地电脑建立SSH隧道：

```bash
ssh -N -L 8420:127.0.0.1:8420 USER@SERVER
```

然后本地浏览器打开 <http://127.0.0.1:8420>。停止网页监控不会停止训练：

```bash
bash scripts/cloud_vla/service.sh monitor-stop
```

也可以不启动网页，只查看训练原始日志：

```bash
tail -n 100 -f .runtime/train.log
```

## 8. 需要带回本地的文件

带回整个运行目录及其同名 `.launch.json`，不要只取 `model.safetensors`。每个checkpoint至少保留：

- `pretrained_model/`：LoRA adapter、policy config、processor及数据统计；
- `training_state/`：优化器、scheduler和续训步数；
- 运行根目录日志和WandB offline目录，用于比较loss和选择checkpoint。

两个模型的flow-matching loss不能与ACT的L1直接比较；最终仍应分别做固定留出集离线检查和真机成功率测试。
