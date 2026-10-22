# RobotArm

Piper 主从臂／手柄采集、LeRobot 训练、离线分析与在线执行工作区。
本页是可直接查命令的操作手册：功能用途、前置条件、运行命令、参数和输出位置。
文档集中在 `docs/`；以下命令仍从项目根目录执行，不要在 `docs/` 中运行。
实验参数、结果、失败分析见
[BestPractice_and_LessonLearn.md](BestPractice_and_LessonLearn.md)。

**使用方式：选择需要的小节，不要从头到尾整段执行。** 标注`--execute`、`--live`的硬件命令可能运动；
训练、转换、报告命令即使没有这些开关也可能写文件。示例目录已有内容时换新名字，不随手加`--overwrite`。

| 我现在要做什么 | 直接跳转 |
|---|---|
| 打开全局／腕部／双相机 | [相机画面](#31-相机画面全局腕部双相机) |
| 查看CAN、使能、回零 | [硬件检查与维护](#3-相机与硬件) |
| 主从／手柄／PICO遥操 | [采集与遥操](#4-采集与遥操) |
| 原始session合并、转成LeRobot | [数据处理](#5-数据合并转换与审查) |
| 训练、续训、实时loss | [训练](#6-act训练续训与loss页面) |
| Open-loop、A/B/C/D、HTML报告 | [离线分析](#7-分析模型) |
| π0微调、迁回、延迟测试 | [π0工作流](#8-π0与π05离线训练测试) |
| 真机执行、审批、按键 | [在线运行](#9-在线运行先shadow再执行) |
| 仿真、PPO、模型查看器 | [仿真](#10-仿真与模型可视化) |
| 查全部功能名／脚本／常见问题 | [命令清单](#11-全部cli功能清单)、[辅助脚本](#12-辅助脚本)、[排查](#13-常见问题) |

## 1. 从这里开始

```bash
conda activate robotarm
cd /home/dm/RobotArm
robotarm --help
robotarm data-acquisition --help
robotarm modeling --help
robotarm simulation --help
robotarm modeling online-policy -- --help
```

第一次配置环境：`bash scripts/create_unified_env.sh`，详见[环境手册](#11-环境版本与验收)。
统一格式是 `robotarm <组> <功能> -- <参数>`；后面的 `--help` 才是功能本身的帮助。
修改代码或 YAML 后必须退出旧进程再启动。云端只训练 π0 时用独立环境，不安装整套硬件依赖。

操作说明统一维护在本页；按顶部导航找常用命令，详细控制行为和排查说明在各节后半部分。
测试文件用途另见[测试说明](../tests/README.md)。`docs/best_practice_assets/`继续保存台账插图。

### 1.1 环境版本与验收

项目统一使用名为 `robotarm` 的 Conda 环境，Python 版本为 3.12。数据采集、
LeRobot 训练与推理、XR 遥操作和 ManiSkill 仿真不再切换解释器。

项目以 editable 模式安装；改 Python 文件无需重装，但须重启进程。修改依赖、命令入口或本地原生扩展后，需要重新执行对应安装步骤。

#### 版本基线

- Python 3.12
- PyTorch 2.11 / CUDA 12.8
- LeRobot 0.6.1
- NumPy 2.2.6
- ManiSkill 3.0.1 / SAPIEN 3.0.3
- Stable-Baselines3 2.9.0
- Placo 0.9.23
- XRoboToolkit SDK 1.0.2（由 `vendor/xrobotoolkit_sdk` 编译）

#### 已知的依赖元数据例外

LeRobot 0.6.1 要求 NumPy `<2.3`。Placo 0.9.23 间接安装的
`cmeel-boost 1.89.0` 在 Python 3.12 上声明 NumPy `>=2.3,<2.4`。统一环境固定
NumPy 2.2.6；Placo 的 Piper IK 已通过运行测试，但 `pip check` 会继续报告这一条
元数据冲突。升级相关组件前必须同时回归测试 LeRobot 数据加载和 Piper IK。

LeRobot 同时安装 `opencv-python-headless`，仿真查看器使用同版本
`opencv-python` 提供 Qt GUI。两个发行包必须保持完全相同的版本。

#### 验收

```bash
robotarm --help
robotarm data-acquisition collect-piper -- --help
robotarm data-acquisition piper-xr -- --help
robotarm modeling check-environment
robotarm simulation viewer -- --help
```

`modeling check-environment` 包含 CUDA 矩阵计算，因此需要 NVIDIA 驱动正常工作。

## 2. 默认参数怎么改

| 配置文件 | 管理范围 |
|---|---|
| `src/robotarm/config/data_acquisition.yaml` | 手柄、主从、数据转换 |
| `src/robotarm/config/xr.yaml` | PICO 单／双臂 |
| `src/robotarm/config/modeling.yaml` | 训练包装、在线控制、策略覆盖、审批 |

在线参数优先级：**显式 CLI > `online-policy.policies.<类型>` > 通用 YAML**。
模型结构和归一化器来自 checkpoint；`n_action_steps` 等运行参数可以被覆盖，实际值看启动日志。
训练入口采用 LeRobot 的下划线参数，如 `--batch_size`、`--policy.chunk_size`；
在线入口常用连字符，如 `--n-action-steps`。

2026-10-19核对：ACT 的 YAML 覆盖为 `n_action_steps: 30`、关节 step/lead=2.5°/6°、
原始关节差门槛12°、原始夹爪差门槛30mm；π0 的原始关节差门槛15°。这些是配置快照，**不是新推荐值**。
本次未修改这些值。要消费20步请显式加 `--n-action-steps 20`；不要将当前默认值当成历史实验参数。

`approval-required: false` 当前跳过审批文件检查，不代表验证通过；仍需执行开关和运行时门控。
开启审批请按[部署流程](#9-在线运行先shadow再执行)操作，不为绕过失败日志而放宽门槛。

查看当前在线配置（只读，不连接硬件）：

```bash
python -c "from robotarm.config import defaults; import pprint; pprint.pp(defaults('modeling', 'online-policy'))"
```

示例使用global D435 `347522071736`、wrist D405 `260322272554`。
CAN示例使用`can_leader1`/`can_follower1`；若系统显示`can0`/`can1`，确认对应机械臂后替换，不能凭编号猜角色。
手柄采集默认wrist为空，因此双相机采集命令必须显式传`--wrist-camera-serial`。

## 3. 相机与硬件

### 3.1 相机画面：全局、腕部、双相机

先退出占用相机的采集/推理程序。以下只预览真实相机，不连接机械臂；入口虽然在`simulation`组，但不是仿真画面。

```bash
# 只打开全局相机
robotarm simulation realsense-stream -- --serials 347522071736 --color-only

# 只打开腕部相机
robotarm simulation realsense-stream -- --serials 260322272554 --color-only

# 指定同时打开这两个相机，序列号用空格分开
robotarm simulation realsense-stream -- --serials 347522071736 260322272554 --color-only

# 打开所有已连接相机，只显示RGB
robotarm simulation realsense-stream -- --color-only

# 全局相机全屏；不需要全屏就去掉最后的参数
robotarm simulation realsense-stream -- --serials 347522071736 --color-only --fullscreen

# 自动选择RGB+深度；USB 2.x默认只开RGB
robotarm simulation realsense-stream

# 按正式采集设置检查双相机，不连接机械臂
robotarm data-acquisition collect-piper -- --camera-check \
  --global-camera-serial 347522071736 --wrist-camera-serial 260322272554
```

预览窗口按`Q`退出。`--serials`选择设备，`--color-only`仅RGB，`--fullscreen`启动全屏。
独立相机预览与online-policy是不同窗口实现，不要假设所有窗口都有相同快捷键。
拆装相机后要与原采集视野核对；序列号相同不等于安装角度相同。

### 3.2 CAN与反馈

先确认设备名及角色，不只凭 CAN 名称判断主／从：

```bash
ip -details -brief link show type can
lsusb
robotarm data-acquisition read-status -- \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --feedback-timeout 3 --interval 0.1
```

确认接口对应关系后，若接口尚未启用，可配置Piper的1Mbps CAN：

```bash
sudo ip link set can_leader1 up type can bitrate 1000000
sudo ip link set can_follower1 up type can bitrate 1000000
```

`read-status`只读反馈，Ctrl+C退出；接口已UP但需更改配置时，先停控制程序再按现场情况处理。

### 3.3 使能、失能、复位、独立回零

先退出其他控制程序，确认对象及路径。**失能/复位/角色切换前托稳机械臂，可能失去保持力。**
下列命令包含执行开关；仅检查参数时去掉`--execute`。

```bash
# 使能从臂
robotarm data-acquisition enable-control -- \
  enable --interface socketcan --channel can_follower1 --firmware v189 --execute

# 失能：先托稳
robotarm data-acquisition enable-control -- \
  disable --interface socketcan --channel can_follower1 --firmware v189 --confirm-drop --execute

# 软件复位：不等同于解除实体急停
robotarm data-acquisition enable-control -- \
  reset --interface socketcan --channel can_follower1 --firmware v189 --confirm-drop --execute

# 六关节回零，夹爪保持开度
robotarm data-acquisition home-piper -- \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --speed-percent 10 --timeout 60 --execute

# 需要退出主臂角色时，切回Follower角色；不发送关节运动/使能命令
robotarm data-acquisition restore-piper-role -- \
  --interface socketcan --channel can_leader1 --firmware v189 --confirm-drop --execute
```

角色切换后按提示重新上电。`--speed-percent`是独立回零速度，不能当作模型/遥操的执行速度。

## 4. 采集与遥操

### 主从臂：当前自定义实现

```bash
robotarm data-acquisition collect-leader-follower -- \
  --execute --allow-enabled-start \
  --leader-channel can_leader1 --follower-channel can_follower1 \
  --global-camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --setup-id leader_follower_v1 \
  --task "Pick up the cup and place it in the target container"
```

去掉 `--execute` 为预览。确认从臂状态后才使用 `--allow-enabled-start`。
终端单键：`l` 请求零力 → `a` 对齐并等待完成 → `f` 跟随 → `r` 开始/保存录制；
`h` 非录制时回零，`q` 保持退出，`x` 急停。不要同时运行另一个控制程序。
WeGo 对照入口已移除，历史行为仅保留在实验台账。

其他模式：

```bash
# 只读主从反馈，检查CAN角色与输入，不跟随
robotarm data-acquisition collect-leader-follower -- \
  --check-only --leader-channel can_leader1 --follower-channel can_follower1

# 纯遥操，不打开相机或录制
robotarm data-acquisition collect-leader-follower -- \
  --execute --control-only --allow-enabled-start \
  --leader-channel can_leader1 --follower-channel can_follower1
```

`s`查看状态、`d`丢弃录制。默认跟随50Hz、录制20FPS，`--control-rate`与`--record-fps`不是一个参数。
`--setup-id`标记装定批次；`--task`必须写实际示教任务，例如仅抓取与抓取后放入孔中不要误用同一个描述。
新会话默认写入`data/raw/piper_raw/<时间戳>/`。

#### 主从对齐、参数与动作记录

软件桥接读取零力主臂，再用 MOVE_JS 控制从臂；不要让另一程序或固件直连同时控制从臂。
终端单键，不用回车。

| 键 | 功能 |
|---|---|
| l | 请求主臂零力拖动 |
| a | 固定按键时的主臂姿态，从臂慢速对齐，完成后保持 |
| f | 开始/暂停跟随；对齐中取消对齐 |
| s | 打印两臂模式、角度、目标、误差和反馈年龄 |
| r / d | 开始/保存 episode / 丢弃 episode |
| h | 非录制时先主臂后从臂回零，闭合两侧夹爪 |
| q | 保持位置退出 |
| x | 从臂急停，主臂尝试使能保持后退出 |

一条 episode：l → a → 等待对齐完成 → f → r → 完成任务 → r。
启动或保存后可按 h，完成后保持使能，不自动录下一条。主臂用固件回零指令，
速度由控制器决定；从臂用 10% MOVE_J。回零中退出急停当前臂。

| 参数 | 含义 |
|---|---|
| `--joint-signs` / `--joint-offsets-deg` | 关节映射方向 / 零点偏移，默认同向、零偏移 |
| `--control-rate` | 默认跟随 50 Hz，录制另为 20 FPS |
| `--max-step-joint-deg` | 每周期关节目标最大变化，当前默认 3° |
| `--max-following-error-deg` | 从臂反馈与目标的误差上限 |
| `--joint-limit-margin-deg` | 命令距标称限位裕量，至少 0.5° |
| `--stale-timeout` | 从臂反馈超时，默认 0.2 秒 |
| `--leader-init-timeout` | 等待主臂完整初值，默认 30 秒 |
| `--home-timeout` / `--home-gripper-speed-mm-s` | 回零各阶段超时 / 夹爪闭合速度，默认 60 秒 / 10 mm/s |

主臂已在零力模式时按提示轻移三个关节对并操作夹爪。主臂静止可保持最后输入，
从臂始终检查反馈新鲜度。a 对齐使用 10% MOVE_J、目标最大 5°/s；跟随前误差须在 2° 内
（配置阈值更小时取更小值），按 s 看详情。

记录 state 为从臂六关节 rad + 夹爪 m，action 为实际下发的限速/限幅后目标。
leader_state、跟随误差、反馈年龄保留作诊断。目标越限自动限幅，主臂返回范围后自动恢复。
实测反馈限位容差为 5°，不放宽命令限位。回零异常保留 home-rx 摘要，只有新反馈确认到位
才继续下一阶段；角色恢复见上文硬件维护小节。

### 手柄：DLS 默认后端

#### 完整命令

```bash
robotarm data-acquisition collect-piper -- \
  --execute --allow-enabled-start \
  --interface socketcan --channel can_follower1 --firmware v189 --feedback-timeout 3 \
  --ik-backend dls --step-deg 0.5 --max-lead-deg 5 \
  --xy-step-mm 5 --z-step-mm 2 --angle-step-deg 1 \
  --max-position-lead-mm 20 --max-angle-lead-deg 5 --max-joint-command-deg 10 \
  --cartesian-smoothing 0.35 --tcp-offset-mm 142.5 \
  --preferred-joints-deg 0 45 -90 0 30 0 \
  --speed 10 --home-timeout 60 \
  --gripper-step-mm 2 --gripper-max-mm 100 --gripper-force 1 --gripper-home-speed-mm-s 10 \
  --gamepad-index 0 --deadzone 0.18 --trigger-deadzone 0.12 \
  --control-rate 30 --hold-seconds 1 \
  --global-camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --record-width 640 --record-height 480 --record-fps 20 --encoder-queue-size 60 \
  --setup-id table_left_v1 --task "Pick up the cup and place it in the target container"

# 手柄诊断，不连接机械臂
robotarm data-acquisition collect-piper -- \
  --gamepad-monitor --gamepad-index 0 --gamepad-connection wired
```

去掉 execute 可预览参数。不传 gamepad-connection 时交互选择 wired/bluetooth。
从臂已经使能时确认状态后使用 allow-enabled-start。默认末端模式、控制暂停。

#### 按键与采集顺序

PS 的 Cross/Circle/Square/Triangle 对应 A/B/X/Y。

| 输入 | 关节模式 | 末端模式 |
|---|---|---|
| 左摇杆左右 / 上下 | J1 / J2 | Y / X |
| 右摇杆上下 / 左右 | J3 / J5 | Z / Yaw |
| 十字键左右 / 上下 | J4 / J6 | Roll / Pitch |
| LT / RT | 夹爪闭合 / 打开 | 同左 |
| LB / RB | 降低 / 提高 Jog 档位 | 同左 |
| A | 切换关节/末端模式 | 同左 |
| 长按 Start | 使能关节和夹爪，保持暂停 | 同左 |
| 长按 X | 开始/暂停控制 | 同左 |
| 短按并松开 B | 开始 episode；再次按下保存 | 同左 |
| 录制时长按 B | 丢弃 episode | 同左 |
| 长按 Y | 关节与夹爪回零，完成后暂停 | 同左 |
| 长按 Back | 失能，操作前托稳机械臂 | 同左 |
| L3 + R3 | 电子急停退出，录制标为 incomplete | 同左 |

顺序：启动 → Start 使能 → X 控制 → B 录制 → 完成任务 → B 保存。
控制与录制独立，暂停运动不会自动保存。右摇杆 Z 优先：竖直轴出死区时控制 Z，
回到死区后水平轴才控制 Yaw。XYZ 平移不叠加姿态输入。

#### 参数与修改方法

| 参数 | 含义 |
|---|---|
| `--ik-backend dls\|placo` | 末端逆解；默认 dls，placo 用于对照 |
| `--step-deg`、LB/RB | 关节 Jog 增量/档位，不改变回零速度 |
| `--xy-step-mm`、`--z-step-mm`、`--angle-step-deg` | 末端满摇杆步长，按控制频率换算 |
| `--max-lead-deg` | 最终关节命令相对反馈超前量 |
| `--max-position-lead-mm`、`--max-angle-lead-deg` | 末端目标相对反馈超前量 |
| `--max-joint-command-deg` | 逆解相对反馈的最大关节变化，超限拒绝 |
| `--cartesian-smoothing` | 关节命令低通系数，越大响应越直接；按频率换算 |
| `--tcp-offset-mm` | 法兰本地 Z 到控制点的偏移；142.5 mm 为模型参考，换工具后实测 |
| `--preferred-joints-deg` | DLS 软参考姿态；Placo 不使用此参数 |
| `--speed` | MOVE_J 回零速度百分比，默认 10；不控制摇杆速度 |
| `--gripper-home-speed-mm-s` | 夹爪回零闭合速度，默认 10 mm/s |
| `--deadzone` / `--trigger-deadzone` | 摇杆 / 扳机死区 |
| `--control-rate` / `--record-fps` | 控制频率 / 数据采样频率 |
| `--setup-id`、`--task` | 装定标识与任务文字，同批保持一致 |
| `--no-preview` | 关闭窗口，双相机仍录制 |

预览最多 10 FPS，末端状态最多 5 Hz。每 2 秒 timing 中 input_max 为输入读取，
control_max 为控制段，record_max 为录制/预览段；ik_max 包含于 control_max。
这些是各段峰值，不能相加当作一拍总耗时。拒绝时保留完整错误。

### PICO单臂与双臂遥操

先运行XRoboToolkit PC Service和头显应用。PICO当前不提供有效录制入口，正式采集用主从或手柄。

```bash
# 终端1：桥接服务
robotarm data-acquisition xr-bridge -- --endpoint tcp://127.0.0.1:60062 --rate 100

# 终端2：单臂只读；核对坐标/工作空间后，同参数加 --live 才允许遥操
robotarm data-acquisition piper-xr -- \
  --arms 1 --xr-bridge --xr-bridge-endpoint tcp://127.0.0.1:60062 \
  --channel can_follower1 --firmware v189 --scale 0.4

# 双臂只读；确认后加 --live
robotarm data-acquisition piper-xr -- \
  --arms 2 --xr-bridge --left-channel can_leader1 --right-channel can_follower1 \
  --firmware v189 --scale 0.4
```

单臂长按B使能、Grip接合/松开保持、松开Grip后长按A回零、右摇杆按下急停。
双臂Y/B使能、X/A回零，任一摇杆按下急停两臂。
`--scale`调位移比例，方向与安全工作空间配置见[PICO手册](#pico控制参数与排查)。
独立双臂入口`robotarm data-acquisition piper-bimanual-xr`也保留；优先用上面的`--arms 2`统一入口。

### PICO控制参数与排查

统一入口 `piper-xr` 使用 `--arms 1`（默认）控制单臂，`--arms 2` 控制双臂。
单臂使用右手柄和 `--channel`；双臂使用左/右手柄与各自的
`--left-channel` / `--right-channel`。双臂模式读取 `xr.yaml` 的
`piper-bimanual-xr` 配置，单臂模式读取 `piper-xr` 配置。

保持XRoboToolkit PC Service和bridge运行，启动命令见上节。无`--live`仍需读取CAN反馈，但不使能、不发送运动目标；确认后在同参数命令加`--live`，长按B才使能。

`--firmware v189` 跳过逐通道固件探测（S-V1.8-9 / S-V1.9-0 都用 v189 驱动），
启动更快，也不再打印探测线程的 `Bad file descriptor` 回溯。`--scale 0.4` 表示
手走 10 cm、末端走 4 cm，减少撞工作空间边界和丢位移；需要 1:1 时用 `1.0`。

启动时松开右握持键和 A，长按 B 约 1 秒；看到 `Piper enabled. Joint motor enables: [True, True, True, True, True, True]` 后，再松开并握紧右握持键接合。不加 `--live` 时 B 只打印提示，不操作硬件。短按 B 无动作，`--record` 已禁用。握持达到 0.9 接合，低于 0.7 释放；默认以位置为优先、尽量保持接合时的法兰朝向（奇异位形附近允许朝向让步）。添加 `--orientation` 可让朝向随右手柄旋转。右扳机松开时夹爪闭合、按下时张开；长按 A 回零并保持使能，右摇杆按下急停。

PICO 默认配置独立存放在 `src/robotarm/config/xr.yaml`，不再读取共享文件的 `piper-xr` 段。当前单臂默认控制频率 100 Hz、双臂 50 Hz，平移目标限速 0.12 m/s，单关节命令限速 0.45 rad/s（约 25.8°/秒）。使用 `--translation-speed`（m/s）和 `--joint-speed`（rad/s）调整；同时保留每步限幅，取更严格者。控制周期延误时不补发大步长。不要复制旧示例中的每步限幅到新的频率上，否则会改变速度上限。目标推进基于限幅后关节命令的正解，减少目标积累。


| PICO参数 | 含义 |
|---|---|
| `--translation-speed` / `--joint-speed` | 平移m/s、单关节rad/s上限 |
| `--max-step-translation` / `--max-step-joint` | 单周期m/rad上限，与速度×周期取较小值 |
| `--max-target-lead` / `--max-relative-translation` | 相对反馈超前／单次接合范围，单位m |
| `--xyz-min` / `--xyz-max` | 基座坐标系工作空间；双臂分别用left-/right-前缀 |
| `--posture-anchor engage` / `seed` | 固定接合姿态参考／随上一周期种子变化 |

XR 超过 0.2 秒不更新时脱离；恢复后必须松手再接合。松手/断流/逆解拒绝时发送当前关节位置作为保持目标，正常退出（包括回零期间 Ctrl+C）时发送当前位置保持命令，保持使能，不发送软件急停；运行异常或右摇杆主动急停仍发送软件急停。停止效果仍需在真实控制器上验证。回零期间也监测 XR 断流、退出信号和右摇杆急停。

离线验证：

```bash
python -m unittest discover -s tests -p test_pico_control.py -v
```

#### 双手柄控制双臂

通过统一入口 `piper-xr -- --arms 2` 启动双臂控制：默认左手柄控制 `can_leader1`，右手柄控制 `can_follower1`。
旧入口 `piper-bimanual-xr` 仍可使用，行为相同。
升级代码后必须重启 XR bridge，因为旧 bridge 消息不包含左手柄字段。

确认两个手柄方向和两台机械臂反馈后再增加 `--live`。首次真机建议缩小映射比例、
一次只接合一只手；方向和镜像按实测填写（`--left/right-xr-yaw-deg`、
`--left/right-xr-mirror`）：

```bash
robotarm data-acquisition piper-xr -- \
  --arms 2 --xr-bridge --left-channel can_leader1 --right-channel can_follower1 --firmware v189 \
  --left-xr-mirror lr --right-xr-mirror lr \
  --scale 0.4 \
  --live
```

降低映射比例可减少目标位移；降低速度上限也会使快速手部输入更容易被丢弃。调整时同时核对工作空间、限速和实际跟随误差，不以取消限制解决跟随问题。

程序首先等待 bridge 的左右手柄姿态有效、时间戳持续更新，然后才连接机械臂。
出现 `Waiting for fresh left and right controller poses` 时，Y/B 尚未生效；
先确认 `xr-bridge` 正在监听相同端口、头显应用已打开 Send、两只手柄均已唤醒。
等待日志每 2 秒显示时间戳和左右姿态状态；出现 `Both XR controllers are ready`
并完成机械臂连接后，再松开两侧 Grip、长按 Y/B。启动等待超时不会连接机械臂或发送急停。

按键映射：左/右 Grip 接合对应机械臂；Trigger 松开时对应夹爪闭合，按下时按比例
张开。松开两侧
Grip 后，长按 Y/B 分别使能左/右臂。松开对应侧 Grip 后，长按 X/A 分别让左/右
臂回零；回零开始时另一臂也会立即保持，回零结束后两只手都必须松开再重新接合。
不要同时长按 X+A；在没有完整双臂连杆碰撞检测时，程序会拒绝联合回零。需要两臂
都回零时，先单独完成一臂，松开 X/A，再单独回零另一臂。
按下任意一侧摇杆会对两臂发送电子急停。XR 消息过期时两臂保持，恢复后必须先
松开 Grip 再重新接合。

持续遥操使用较严格的 `--stale-timeout`（默认 0.2 秒）；MOVE_J 回零期间的 CAN
反馈调用可能产生更长调度间隔，因此使用独立的 `--homing-stale-timeout`（默认
1.0 秒）。这只放宽回零期间的 XR 断流判定，任一摇杆急停仍会立即生效。

夹爪命令力上限由 `--gripper-force` 控制（默认 1.0 N）。闭合时若 `0x2A8` 反馈力
达到 `--gripper-force-hold-threshold`（默认 0.8 N），程序会锁存当时的实际开口并
保持；再次按 Trigger 请求张开至少 1 mm 后解除锁存。阈值设为 0 可禁用软件力保持。
易变形物体应同时降低命令力和保持阈值，并从空载低值开始验证；该反馈来自夹爪驱动器，
不是独立力传感器，不能保证物体绝对不变形。

左右 `xyz-min/max` 都在各自机械臂基座坐标系内。默认没有启用双臂碰撞检测；应先
按真实安装收紧两个工作空间。如果两台机械臂基座坐标轴平行，可测量它们在同一坐标
系中的原点，并启用额外的法兰距离保护，例如：

```bash
  --left-base-origin 0,0.30,0 \
  --right-base-origin 0,-0.30,0 \
  --min-flange-separation 0.20
```

该保护只检查两个法兰点之间的距离，不是完整的连杆碰撞检测，不能替代物理急停、
低速测试和合理的工作空间隔离。

运行中出现 `WORKSPACE_LIMIT` 或 `FOLLOW_LIMIT` 不是程序崩溃：前者表示对应手柄
目标超出该臂 `xyz-min/max`，应向内移动或松开 Grip 后重新接合；后者表示目标比
实际末端领先超过 `max-target-lead`（默认 2 cm），超出的手部位移已被丢弃，应
降低手速。如果刚接合就持续出现 `WORKSPACE_LIMIT`，查看 `ENGAGED xyz=...`，按
真实安全范围调整对应的 `--left-xyz-*` 或 `--right-xyz-*`。如果手已经停止仍持续
出现 `FOLLOW_LIMIT`，应松开 Grip，并检查机械臂是否已使能、状态是否正常以及
关节是否受限；不要直接通过增大阈值掩盖未运动的问题。限位提示最多每秒打印一次。

测试覆盖姿态保持、坐标映射、旋转限幅、握持迟滞、断流重新接合、启动拒绝冻结数据和默认速度上限，不替代真机运动及停止验证。

修正逐关节截断造成的方向偏差：关节速度、跟随误差和位置限制统一用同一个比例缩放整组关节增量。若不存在满足限制的等比例步长，则保持并要求重新接合。

日志 `hand_delta` 为 XR 原始坐标位移，`requested_delta` 为工作空间限制后的机械臂基座目标位移（均为米）。当前固定映射为机器人 X=-XR Z、机器人 Y=-XR X、机器人 Z=XR Y，依赖头显跟踪坐标与机器人朝向对齐；站位不会自动校准映射。默认 scale=1.0，手移动 10 cm 对应目标 10 cm，单次接合相对目标最多 20 cm；关节速度上限也会限制实际跟随速度。

可用 `--xr-yaw-deg` 绕机器人竖直轴校准水平映射，例如 90 表示将默认映射后的 +X 转向 +Y；竖直方向保持不变。此参数不会根据站位自动估计，先在 dry-run 查看 `requested_delta` 的方向。改变头显跟踪原点后应重新检查。

`--xr-yaw-deg` 只能旋转、不能镜像。若**只有左右**（或只有前后）相反、另一个水平方向正确，用 `--xr-mirror lr`（或 `--xr-mirror fb`）纠正——例如面朝机械臂操作时常见左右相反；竖直方向不受镜像影响。前后和左右**同时**相反时应使用 `--xr-yaw-deg 180`，不要叠加两个镜像。双臂分别是 `--left-xr-mirror` / `--right-xr-mirror`，两侧可独立设置。

不确定坐标映射时先用只读脚本验证（保持 bridge 运行、戴好头显、握好手柄）：

```bash
python scripts/check_xr_frame.py --hand right --yaw 0 --mirror none
```

按提示依次做「朝机械臂推 / 往自己的右手方向平移 / 向上抬」三个动作，脚本会打印每次
动作在 PICO 坐标系和机器人基座坐标系里的位移分量，并列出所有 `yaw × mirror` 组合的
结果，用来挑出符合直觉的那一组。判定标准：手朝机械臂推 → `机X` 为正、手抬高 →
`机Z` 为正、手往你右边 → `机Y` 为负。脚本只订阅 bridge 发布的 XR 状态，不连接机械臂。

房间坐标系由头显 **recenter** 决定：改变 recenter 时的朝向会整体旋转映射，因此每次
开机应保持固定站位与朝向，改完参数后重新验证一次。端到端复核可用单臂 dry-run 打印的
`hand_delta`（XR 原始位移）与 `requested_delta`（限幅后的机器人基座位移）对照方向。

默认 `--orientation-weight 0.00001` 为平移模式的弱姿态偏好，避免腕部奇异位形让位置运动几乎停住；夹爪朝向可能让步。提高该值会更重视姿态，但可能再次降低平移响应。`--orientation` 模式仍使用更强的姿态任务。

诊断输出：`loop_hz` 为控制循环实际频率；`ik_ms` 为单次逆解耗时；`limit_ratio` 为关节限幅保留的步长比例（接近 0 表示受限严重）；`tracking_error_mm` 为最终请求与反馈末端位置差；`joint_lead_rad` 为命令领先反馈的最大关节角。后者上限仍为 0.05 rad。

PICO 单臂和双臂现在都使用带验证的 Placo IK。求解以“上一周期安全命令”为种子，
实际关节反馈仍由跟随误差边界约束；这能提高解的连续性，但不会允许命令脱离反馈。
每次求解都会检查位置/姿态残差，超限时把本周期笛卡尔步长依次缩到
`0.5 / 0.25 / 0.125`，仍不合格才拒绝该帧。

零空间正则默认锚定**接合瞬间捕获的关节姿势**（`--posture-anchor engage`）。位置任务
权重 1.0、正则项 1e-4，所以它只作用在位置解不唯一的自由度上，把机械臂保持在你接合
时的构型（包括夹爪朝向）。`--posture-anchor seed` 随上一周期命令更新参考，不能纠正累积漂移。默认使用固定接合参考 `engage`；历史对比数字见实验台账。

接合时的姿势也会影响手感：回零后的折叠近零位姿位姿条件数可达 300～700（腕部接近
奇异），此时任何逆解都只能靠腕部大幅调整来跟随，容易看起来混乱。建议操作前先把臂
摆到一个舒展的工作姿态（例如 J2≈45°、J3≈-90°、J5≈30°）再接合。

新增诊断字段：`ik_residual_mm`/`res` 为已接受目标的末端位置残差，
`cond_xyz` 为平移 Jacobian 条件数，`cond_pose` 为使用 0.1 m 特征长度归一化后的
6D 位姿条件数，`ik_shrink`/`shrink` 为实际保留的 IK 目标比例；数值越小表示缩步越
明显。`cart_ratio` 和 `joint_ratio` 分别表示笛卡尔及关节步长保留比例，`limits` 会列出
`WORKSPACE_LIMIT`、`FOLLOW_LIMIT`、`CART_STEP`、`IK_SHRINK`、`JOINT_STEP`、
`JOINT_FOLLOW` 或 `JOINT_BOUND`。条件数用于同一机械臂不同姿态间的趋势判断，不是
绝对距离或角度误差。

#### 限制平移目标积累

平移目标现在由每帧手部增量更新。默认 `--max-target-lead 0.02`，在反馈位于接合边界内时，目标最多领先实际末端 2 cm；超过领先距离、工作空间或固定接合范围的输入会被丢弃，不排队执行。手停住后不会继续消化先前几十厘米的输入。现有关节命令仍受速度和关节跟随误差限制，这不是对实际停止距离的保证。朝向仍使用原来的相对姿态逻辑。

慢速小幅移动按scale映射（默认1:1）；移动过快时会丢弃部分位移。遇到以下提示应调整操作：

- `FOLLOW_LIMIT`：机械臂追不上手，放慢动作，必要时松手重新接合。
- `RELATIVE_LIMIT`：达到本次接合的 20 cm 相对范围，松开握持键、收回手、重新握住。
- `WORKSPACE_LIMIT`：达到工作空间边界，向内移动。
- `JOINT_LIMIT`：有明显位置误差且关节步长被大幅缩小，可能是关节速度/跟随误差/位置限制；此提示不等于已经确认奇异位形。

提示最多每秒输出一次，不改变 B 使能、A 回零或正常退出保持使能的行为。`tracking_error_mm` 现在对应受限后的有效目标误差，不能直接与旧版本的原始目标误差比较。

#### 接合方式与误操作保护

单臂、双臂默认 `--engagement-mode grip`：长按 Y/B 使能左/右臂（单臂用 B）后，
**握住 Grip（≥0.9）才跟随，松开 Grip 立即脱离并保持当前位置**——松手即暂停，
重新握住即从新手部基准继续，不会追补松手期间的动作。Trigger 仍控制夹爪，
摇杆按下仍为急停。朝向跟随需要 `--orientation`。重新接合前保持手柄稳定；
接合时的手柄和机械臂位姿成为新的相对基准。一次接合内朝向基准保持固定，
手腕转回对应原来的末端朝向；受关节限位、奇异姿态和速度限制约束，
不能保证所有姿态都可达。

默认关节限速0.45 rad/s，每周期最大关节步长0.009 rad，朝向步长1.2°（每步限制还需与实际控制频率一起核对）。IK 每个候选最多做四次收敛迭代，最终指令仍通过关节限速和反馈误差检查。

相邻有效手柄数据位置变化超过 `--max-hand-jump-m 0.05`，或旋转变化超过 `--max-hand-jump-deg 30`，触发 `INPUT PAUSED`，保持当前位置并停止夹爪指令。双臂任一侧触发会暂停两侧，分别重新长按 Y/B 恢复；不会自行恢复。grip 模式下数据过期或逆解拒绝需要松开 Grip 后再重新握紧恢复；`INPUT PAUSED` 例外，必须长按使能键。恢复时重新捕获基准，不追赶暂停期间的手柄动作。该检测针对手柄位姿突变，不是完整的碰撞检测。

```bash
robotarm data-acquisition piper-xr -- \
  --arms 2 --xr-bridge --left-channel can_leader1 --right-channel can_follower1 \
  --orientation --engagement-mode direct \
  --max-hand-jump-m 0.05 --max-hand-jump-deg 30 --live
```

如需无需按住 Grip 的连续跟随（使能一次后手柄移动即跟随，松手不停），
使用 `--engagement-mode direct`。该模式下松开 Grip 不会暂停；摇杆电子急停仍然有效。不要把制造断流作为日常暂停手段；需要松手暂停请用默认 `grip` 模式。

#### 退出控制程序后，指定一台机械臂回零

独立回零无需PICO或XR bridge，命令见[硬件回零](#33-使能失能复位独立回零)。用channel选择单臂；启动时在当前角度使能，再以默认10%的MOVE_J回到六关节零角度，夹爪保持。先退出该通道的其他控制程序并清空回零路径。

省略 `--execute` 只打印预览，不连接机械臂。`--timeout` 默认 60 秒，完成要求所有关节距离零角度不超过 2°、运动状态结束并稳定 0.5 秒。关节反馈超过 0.5 秒未更新、控制异常、超时或 Ctrl+C 中断会发送电子急停；正常完成不会失能。可用 `--firmware v189` 指定已知固件驱动。

`home-piper` 的 `--joint-limit-tolerance-deg` 默认 5°，允许回零前关节反馈略超出 SDK 标称限位；例如 J5 上限 +70°、反馈 +73.596° 可以通过启动检查。参数可设为 0～5°，超过 5° 仍拒绝启动。该容差只用于反馈检查，`MOVE_J` 命令仍被限制在 SDK 标称限位内。完成容差 `--tolerance-deg 2` 独立使用真实反馈角度判定，反馈不会裁剪后用于判定完成。启动前应确认实体关节未卡住或顶住机械限位，编码器读数可信；若回零没有进展，程序会停止并急停，不要反复重试。

## 5. 数据合并、转换与审查

### 5.1 原始session合并

多个原始session要先合成一个**原始**目录时，使用`merge-piper-raw`；不转换为LeRobot：

```bash
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/SESSION_A data/raw/piper_raw/SESSION_B \
  --output data/raw/piper_raw/MERGED_SESSION
```

替换占位符后先运行预检查；确认后加`--execute`复制并重新编号。原数据不动，已有输出拒绝覆盖。
完整示例、来源映射和相机兼容限制见[原始数据合并](#51-原始session合并)。

2026-10-19的三个原始session已经合成`data/raw/piper_raw/20261019_merged_122eps`，不要重复覆盖。
要复用这个流程合并别的批次，换输入列表与新的输出名：

```bash
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/20261019_140239_223914 \
  data/raw/piper_raw/20261019_143324_295798 \
  data/raw/piper_raw/20261019_151443_922093 \
  --output data/raw/piper_raw/20261019_merged_copy_v2 \
  --execute
```

原始合并不重编码视频、不改NPZ值；输出含`session.json`、`episodes/`、`merge_manifest.json`。
默认只选complete轨迹，拒绝重复来源/不同任务/不兼容相机；原session保留。
**不要把源session和它的合并副本同时送入转换或训练，否则重复采样甚至造成训练/验证泄漏。**

#### 原始合并检查与来源追溯

| 参数/行为 | 说明 |
|---|---|
| `sessions...` | 任意数量原始session目录，不能传LeRobot目录；不建议用日期通配符误选合并产物 |
| `--output` | 必填的新raw目录；不能与任何输入重合、互为父子目录，不支持覆盖/追加 |
| `--dry-run` / 不加`--execute` | 检查数组、视频头中的尺寸/FPS/帧数、兼容性和编号；不创建文件 |
| `--execute` | 独立复制，SHA256校验，再写合并元数据；视频不重编码，不进行模型推理 |
| `--include-incomplete` | 默认只保留complete且有帧的episode；显式允许未完成条目仍须通过数据完整性检查 |
| `--allow-mixed-tasks` | 默认拒绝不同任务；显式允许后逐episode保留标签，不自动改成同一任务 |

合并前退出采集程序；发现`.in_progress`会拒绝。中途失败保留部分输出和
`MERGE_INCOMPLETE.txt`，原数据不受影响；不要将未完成输出用于训练/转换。
预检查不完整解码每个视频帧；复制校验确保内容未被合并过程改变，不代表数据质量或任务成功。

来源保存在每个`episode.json`的`merge_source`及根目录`merge_manifest.json`，
清单包含输入session完整元数据、原episode路径/编号和文件SHA256。
重复传同一session、同时传原数据及含相同轨迹的本工具合并产物会报错，避免重复采样。
合并目录不能代替备份；质量审查标记需按新的目录/编号重新核对，不会自动复制到训练数据集。

要求FPS、关节单位/名称/动作语义、相机角色/序列号/视频格式/已记录内参等兼容。
**这些检查不能确认相机外参一致**；拆装后的视角变化仍需人工检查。
不同安装批次的setup信息完整保留在来源元数据中，不应把首个session的setup当作所有批次的证明。

### 5.2 原始数据转成LeRobot（训练前必做）

直接将2026-10-19已经合并好的原始目录转换为双相机LeRobot数据集：

```bash
robotarm data-acquisition convert-piper-raw -- \
  data/raw/piper_raw/20261019_merged_122eps \
  --output data/lerobot/piper_cup_20261019_122eps_v1 \
  --repo-id local/piper_cup_20261019_122eps_v1 \
  --image-writer-threads 4 --encoder-threads 4
```

这条命令会实际写入新数据集，**不用加`--execute`，也没有`--dry-run`参数**。不会改原始session。
输出必须是新目录；转换成功后可见`meta/`、`data/`、`videos/`。

通用单session模板（替换`SESSION`、`DATASET`）：

```bash
robotarm data-acquisition convert-piper-raw -- \
  data/raw/piper_raw/SESSION \
  --output data/lerobot/DATASET --repo-id local/DATASET \
  --image-writer-threads 4 --encoder-threads 4
```

也可以跳过原始合并，直接把多个原始session转成一个LeRobot数据集；二选一即可：

```bash
robotarm data-acquisition convert-piper-raw -- \
  data/raw/piper_raw/20261019_140239_223914 \
  data/raw/piper_raw/20261019_143324_295798 \
  data/raw/piper_raw/20261019_151443_922093 \
  --output data/lerobot/piper_cup_20261019_direct_v1 \
  --repo-id local/piper_cup_20261019_direct_v1 \
  --image-writer-threads 4 --encoder-threads 4
```

| 转换参数 | 含义 |
|---|---|
| `--output` / `--repo-id` | 本地输出目录／数据集标识；本命令不会上传Hub |
| 不传`--camera-role` | 保留所有录制的相机；本批即global+wrist |
| `--camera-role global` | 单全局相机消融，另开输出目录，不覆盖双相机数据 |
| `--space joint --frame absolute` | 默认：六关节rad＋夹爪m的绝对目标，匹配当前真机部署 |
| `--space tcp` / `--frame relative` | 改变动作表示；当前在线执行不能直接照搬 |
| `--include-incomplete` | 允许非complete条目，但仍做完整性检查；训练通常不要用 |
| `--image-writer-threads` / `--encoder-threads` | 图像写入／视频编码线程数 |
| `--overwrite` | 会删除已有输出重建；不是增量合并，日常命令不使用 |

已有`--append`接口当前存在相机角色变量初始化顺序问题，不作为可用推荐流程；
需要增加数据时，把原来的源session和新增session一起转换到**新目录**。
`robotarm modeling convert-dataset`是同一转换器的兼容入口，通常统一用`data-acquisition convert-piper-raw`。

### 5.3 浏览、质量审查、读取数量

```bash
robotarm modeling dataset-browser -- \
  --dataset-root data/lerobot/piper_cup_20261019_122eps_v1

# 只读数据集元信息，查看total_episodes、total_frames、fps和features
python -m json.tool data/lerobot/piper_cup_20261019_122eps_v1/meta/info.json
```

浏览器默认 <http://127.0.0.1:8766>。数据动作采用六关节 rad + 夹爪 m 的绝对命令目标。
页面顶部的 **English / 中文** 按钮可即时切换界面语言，并在当前浏览器记住选择。
切换不会重载episode或丢弃未保存的标记、备注与裁剪范围；任务文字、数据集名和用户备注保持原文。
质量标记与数据分开保存，不用 `lerobot-replay` 代替只读预览。
↑/↓切episode，←/→逐帧，空格播放；在页面标记keep/review/reject。
审查文件默认`artifacts/reports/dataset_quality/<数据集名>.json`；`--quality-filter`训练时排除review/reject，
未审查条目默认仍会被保留，不等于已经人工验收。

#### 每条episode独立裁剪并导出

启动命令仍是上面的`dataset-browser`。修改代码后需退出旧服务、重新启动并刷新页面。

1. 在左侧选择episode，在“当前 Episode 保留片段”填写起点和终点，例如`5`到`14`秒。
   也可逐帧定位后点“当前帧设起点”“当前帧后设终点”；后者包含当前这一帧。
2. 勾选“只播放所选片段”预览，绿色曲线区域表示保留范围；自动质量指标仍针对整条原轨迹。
3. 点击“保存标记与裁剪”。每条轨迹独立保存，可给下一条设置不同范围；切换时未保存会提示。
   “恢复整条”并保存可撤销该条裁剪；质量状态的“清除”不会自动撤销裁剪。
4. 填写一个尚不存在的新数据集名称，点“导出已保存的裁剪数据集”。
   导出在源数据集旁边创建新目录，包含保留及未审查条目，排除复查和剔除条目。
   未设置裁剪的条目保留整条；左侧搜索/状态筛选只影响显示，不改变导出范围。
5. 等页面显示完成，再用新目录浏览或训练。导出期间保持服务运行；仅保存标记不会改变训练数据。
   导出进度条每秒刷新：显示本次参与导出的总条数、已完成条数、百分比及正在处理的原 Episode；
   同时显示帧处理进度和视频编码阶段，支持中英文切换。每条视频编码并保存成功后才计为完成一条。
   条数达到 100% 后仍可能在生成统计与来源清单，须等“完成”提示；失败时保留已完成计数，部分输出不能用于训练。

时间范围为**左闭右开**`[起点, 终点)`，按采样帧对齐；20 FPS下`[5,14)`为第100～279帧，
共180帧。双相机、state、action采用同一帧范围，保留原动作值，episode/帧编号与时间戳从零重建，
归一化统计重新计算，视频会重新编码而非无损复制。原始数据集与raw会话不修改。
当前每条episode支持一个连续保留区间，不能把同一条的多段不连续动作拼在一起。

裁剪标记仍存在`artifacts/reports/dataset_quality/<数据集名>.json`的`trim`字段；
导出目录的`meta/trim_provenance.json`保存审查快照及原episode/帧范围映射。
新目录不沿用旧数据集的episode编号审查文件，应重新审核；不要将原数据和裁剪副本混入同一次训练。
原数据带有未应用的裁剪标记时，`--quality-filter`和π0 LoRA审查入口会拒绝静默训练完整轨迹，
提示先导出；**不使用审查入口而直接训练原目录，仍会训练原始完整轨迹**。
裁掉起始动作会改变示教起始状态分布，真机仍需匹配裁剪后示教的初始条件。

导出失败或中断时，新目录可能包含`TRIM_INCOMPLETE.txt`，不可用于训练；
排查错误后换新目录名重试，不覆盖已有结果。导出使用CPU读取/重编码，不做模型推理，不连接机械臂。

只在**同源、同episode顺序**的衍生数据集上迁移审查结果：

```bash
python -m robotarm.modeling.quality_filter \
  --dataset-root data/lerobot/TARGET_DATASET \
  --derive-from artifacts/reports/dataset_quality/SOURCE_DATASET.json
```

含裁剪范围的审查文件不支持此直接迁移，以免帧范围套用到不同时间轴。

元数据修复入口`robotarm modeling normalize-piper-metadata`是旧pick-up数据专用规则，
会修改元数据并要求`--backup-root`；不要拿它自动统一今天新任务的文字。详见[采集手册](#4-采集与遥操)。

补充的 LeRobot 原生数据查看器（只查看，不能用发送实体动作的 replay 替代）：

```bash
lerobot-dataset-viz \
  --repo-id local/piper_cup_20261019_122eps_v1 \
  --root data/lerobot/piper_cup_20261019_122eps_v1 \
  --episode-index 0 --num-workers 0
```

## 6. ACT训练、续训与loss页面

使用新输出目录，以下是操作模板，不是保证最优的配方：

```bash
TASK_DATASET=data/lerobot/piper_cup_20261019_122eps_v1
TASK_RUN=artifacts/checkpoints/act_cup_20261019_chunk50_new_v1
robotarm modeling check-environment -- --dataset-root "$TASK_DATASET"
robotarm modeling train -- "$TASK_DATASET" "$TASK_RUN" act \
  --quality-filter --quality-shuffle-seed 1000 \
  --dataset.eval_split=0.2 --eval_steps=1000 --max_eval_samples=0 \
  --batch_size=16 --steps=40000 --save_freq=2000 --seed=1000 \
  --policy.chunk_size=50 --policy.n_action_steps=20
```

此处用第5节转换后的新数据；若要用旧180条，改`TASK_DATASET`为
`data/lerobot/piper_pick_cup_dual_camera_lf_180eps`，同时更换新训练目录。
`check-environment`会检查依赖及CUDA计算，不连接机械臂。`train`会立即训练，没有`--execute`开关。
修改第三个位置参数`act`可选择该环境支持的`diffusion`/`smolvla`，但不能直接沿用ACT专用参数。

| 参数 | 含义 |
|---|---|
| `chunk_size=50` | 监督/预测50步；20FPS时覆盖2.5秒 |
| `n_action_steps=20` | 运行时消费窗口；对ACT不改变训练监督长度 |
| `quality-shuffle-seed` | 筛选后episode打乱种子，用于固定划分 |
| `eval_split=0.2` | 按任务取传入episode列表尾部20%作为验证 |
| `max_eval_samples=0` | 全量验证，不是关闭验证 |
| `steps` | 总优化步数，resume时也不是新增步数 |
| `save_freq` | checkpoint间隔；最好与验证周期对齐 |

另一个终端运行：

```bash
python scripts/train_monitor.py \
  --run-dir artifacts/checkpoints/act_cup_20261019_chunk50_new_v1 --port 8420 --interval 2
```

打开 <http://127.0.0.1:8420>。关闭监控不会停止训练；目录未创建时监控会等待。
续训命令和优化器状态要求见[训练手册](#续训与留出集包装入口)。

### 续训与留出集包装入口

续训需完整checkpoint：`pretrained_model/`及同级`training_state/`，保持数据及划分不变。
下面是通用模板，先替换`RUN`和`STEP`；不要指向已完成/正在运行的历史目录重复启动：

```bash
lerobot-train \
  --config_path=artifacts/checkpoints/RUN/checkpoints/STEP/pretrained_model/train_config.json \
  --resume=true --steps=90000
```

例如从40000继续到90000，是新增50000步；`steps`是总更新数，不是新增量。
`train-with-holdout`是兼容包装入口，不比显式`train --dataset.eval_split`多一种验证算法：

```bash
robotarm modeling train-with-holdout -- \
  data/lerobot/piper_cup_20261019_122eps_v1 \
  artifacts/checkpoints/act_cup_20261019_holdout_new_v1 \
  --dataset.eval_split=0.2 --max_eval_samples=0 \
  --batch_size=16 --steps=40000 --save_freq=2000 --seed=1000 \
  --policy.chunk_size=50 --policy.n_action_steps=20
```

该包装器不接第三个`act`位置参数；默认policy来自YAML。它的旧默认eval_split是6/31，
不是对任意数据集固定留出6条，所以示例显式指定0.2。
ACT训练总loss包含KL，验证为eval/z=0的L1；公平比较用下一节A/B/C/D，不直接拿两条总loss相减。

### 训练划分、验证口径与权重选择

`--quality-reviews` 可指定审查文件；筛选后用 `--quality-shuffle-seed` 固定 episode 顺序。
LeRobot 按每个任务的列表尾部取 `ceil(n × eval_split)` 个验证 episode；不同种子重复训练是重复随机留出，不是严格 k-fold。
`eval_steps` 按优化更新步数计，不是 epoch；保存间隔与验证间隔对齐，便于挑选有实际验证值的权重。

当前安装的 LeRobot 每 eval_steps 调用 policy.eval()、无梯度验证，再回到训练模式。
日志 eval/eval_loss 是各 batch loss 算术平均。ACT 启用 VAE 时训练总 loss 为
重建 L1 + kl_weight×KL，验证 z=0，仅 L1；都排除 padding。
公平比较训练/验证的 L1 使用下一节报告，而非直接比较训练总 loss 与验证 loss。

无账号时 WandB 默认 offline，日志在 run 的 `wandb/`。事后 `training-report` 默认写到
`<run>/training_report/`，生成曲线、CSV、Markdown，并合并续训日志去重。
验证最优 step 未保存时不能构造一个不存在的 checkpoint；报告只从有对应验证值的已保存项里选候选。
`last` 指向最新保存项；实时页面可将选定项链接为 `checkpoints/best`。
部署需要完整 `pretrained_model/`，完整续训还需要同级 `training_state/`。

## 7. 分析模型

### 训练曲线、Open-loop与潜变量对照

以下使用本机已有LF180/090000模型作示例；换模型时对应替换run、checkpoint和数据集，不能混用划分。

```bash
# 合并所有续训日志；不加载GPU，不改权重
robotarm modeling training-report -- \
  --run-dir artifacts/checkpoints/act_cup_ring_dual_20261013_chunk50_180eps_lf_v1

# Open-loop平台：读取记录观测，不用预测推进真实环境
robotarm modeling trajectory-studio

# A/B/C/D：相同划分，重建与推理L1；快速抽样
robotarm modeling act-latent-report -- \
  --checkpoint artifacts/checkpoints/act_cup_ring_dual_20261013_chunk50_180eps_lf_v1/checkpoints/090000/pretrained_model \
  --device cuda --batch-size 8 --max-frames 128 --repeats 2 --seed 1000 \
  --output artifacts/reports/act_latent/act_090000_sampled.html
```

Open-loop 默认 <http://127.0.0.1:8765>。A/C 使用真实动作编码潜变量；B/D 使用z=0；
四组都不含KL、排除padding。训练总loss、验证loss、物理动作误差和真机成功率不能混用。

| Open-loop / 报告参数 | 用法 |
|---|---|
| 逐帧重规划、输出步k | 每个真实观测重新预测，比较第k步与记录的t+k动作 |
| 连续消费动作块 | 消费一段后重规划；界面输出步须为0 |
| `trajectory-studio -- --port 8767` | 更换平台端口 |
| 潜变量`--max-frames 128` / `0` | 每划分抽样128起点／全量，0会明显更慢 |
| 潜变量`--batch-size 8 --repeats 2 --seed 1000` | 配对比较时保持一致，A/C重复后验采样 |
| `--dataset-root` | 训练数据迁移后覆盖本地位置，仍须为原数据集 |

Open-loop读取真实后续观测，不会用预测动作推进真实环境，也不经过online-policy安全限幅。
缓存写入`artifacts/logs/trajectory_studio/<run_id>/`；判断当前相机是否偏移需要另做实时对照。

#### A/B/C/D复现要求

从 checkpoint 的 `train_config.json` 恢复划分，四组都关闭增强、dropout 和 BN 更新，使用 FP32。
只统计归一化动作空间的 L1、排除 padding，按有效元素加权；与验证日志的逐 batch 算术平均口径不完全相同。
HTML/JSON 同时保存逐维度结果、A/C 重复标准差、episode 和抽样索引。
A/C 使用真实动作编码的随机潜变量；B/D 使用 z=0，不把训练总 loss 中的 KL 混进来。
保持 batch、seed、repeats 和抽样索引一致才可比较 checkpoint。
无 CUDA 可用 `--device cpu --cpu-threads 4 --batch-size 4`；
数据迁移可用 `--dataset-root` 改路径，但必须仍是训练时同一份数据。

### 命令行抽样评估

```bash
robotarm modeling evaluate -- \
  --checkpoint artifacts/checkpoints/act_cup_ring_dual_20261013_chunk50_180eps_lf_v1/checkpoints/090000/pretrained_model \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --episodes 23 --samples 256 --device cuda --seed 1000 \
  --output artifacts/reports/eval/act_lf180_090000_ep23.json
```

`23`只是可替换的episode示例，需核对对应run的验证划分；`--samples`是总抽样上限。
普通`evaluate`不代替π0 LoRA专用重载/评估入口。物理动作包含rad/m，逐维度查看，不把混合单位的整体MAE当成功率。

### 汇总多个ACT checkpoint的HTML报告

先对多个checkpoint生成**相同划分、起点、batch、seed、repeats**的潜变量JSON，再汇总；
下面读取项目中已有的三份报告，不跑新GPU推理：

```bash
robotarm modeling act-training-report -- \
  --run-dir artifacts/checkpoints/act_cup_ring_dual_20261013_chunk50_180eps_lf_v1 \
  --latent-reports \
    artifacts/reports/act_latent/20261015_lf180_040000_sampled.json \
    artifacts/reports/act_latent/20261015_lf180_060000_sampled.json \
    artifacts/reports/act_latent/20261015_lf180_090000_sampled.json \
  --output artifacts/reports/act_training/act_lf180_comparison.html
xdg-open artifacts/reports/act_training/act_lf180_comparison.html
```

报告命令统一放在 `robotarm modeling`：

| 命令 | 产物/用途 |
|---|---|
| `training-report` | 合并续训曲线、CSV、Markdown；只在已保存且有对应验证值的权重中选候选 |
| `act-latent-report` | GPU/CPU离线重建与推理L1，HTML/JSON |
| `act-training-report` | 汇总现有潜变量报告与续训日志，HTML/JSON/CSV |
| `pi0-training-report` | π0训练曲线和最佳/最终checkpoint抽样推理 |
| `pi0-denoising-report` | 同观测、同噪声的5/10次去噪对照 |

详细命令见[训练手册](#6-act训练续训与loss页面)与[π0手册](#8-π0与π05离线训练测试)。
最近一次实测报告索引、参数和结论在[实验台账](BestPractice_and_LessonLearn.md)，不在本页维护。

## 8. π0与π0.5离线训练测试

### 8.1 基础模型推理延迟与量化

使用已有数据帧，不连接CAN、实时相机或控制机械臂；需要本地基础权重和可用GPU：

```bash
# π0：完整动作块延迟，不是单个动作步耗时
robotarm modeling pi-offline-benchmark -- \
  --policy-type pi0 --checkpoint artifacts/pretrained/pi0_base \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --index 0 --wrist-slot right --device cuda --warmup-runs 1 --runs 5

# π0.5
robotarm modeling pi05-offline-benchmark -- \
  --checkpoint artifacts/pretrained/pi05_base \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --index 0 --wrist-slot right --device cuda --warmup-runs 1 --runs 5
```

INT8 weight-only对照：在相应命令后加`--quantization int8-weight-only`，不覆盖原权重；
需匹配torchao/PyTorch环境。`--index`是数据帧、`--runs`是完整预测次数、预热不计入稳态统计。
基础权重的32维动作未直接适配Piper，不能因推理成功就发送真机执行。

### 8.2 π0 LoRA微调与云端环境

云端单独配置Python 3.12环境，只安装训练extra；本地robotarm环境不要为此重新装整套依赖：

```bash
python3.12 -m venv .venv-pi0
source .venv-pi0/bin/activate
python -m pip install -e '.[pi-lora]'
python -m pip check
```

完整上传/离线tokenizer缓存/包校验步骤见[云端包手册](#85-云端包上传校验与启动)。
云端需能使用GPU和BF16；仅有pi0_base文件不保证tokenizer缓存齐全。

```bash
# 默认仅预览20步试跑计划，不训练；确认云端环境后加 --execute
robotarm modeling train-pi0-lora -- \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --output-dir artifacts/checkpoints/pi0_lora_smoke_new_v1 --smoke

# 正式训练模板；先不加execute检查参数，确认后在云端追加 --execute
robotarm modeling train-pi0-lora -- \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --output-dir artifacts/checkpoints/pi0_lora_lf180_new_v1 \
  --batch-size 1 --steps 20000 --rank 16 --alpha 32 --lr 0.0001 \
  --chunk-size 50 --n-action-steps 20 --eval-split 0.2 --seed 1000 \
  --save-freq 1000 --eval-steps 1000 --max-eval-samples 256
```

这是操作模板，不是最优参数声明。batch从可运行的小值起步，看峰值显存再调整；
`--smoke`固定20步并最终保存/验证，正式训练要另开目录。与ACT入口不同，这里用连字符参数。
`--rank`/`--alpha`是LoRA容量/缩放，`--lr`为学习率；`--n-action-steps`不是去噪次数。

#### LoRA预设与数据适配

入口默认 batch=1、rank=16、alpha=32、lr=1e-4、chunk=50、n_action_steps=5。
上面的模板显式覆盖为20；这里不读取ACT在线YAML覆盖。BF16和梯度检查点默认开启。
`--smoke` 固定20步，save/eval=20，验证最多8帧、workers=0；正式默认验证上限256帧，
`--max-eval-samples 0` 为全量。32GB显存不保证任意batch都能运行，先检查峰值再增加。

预设使用绝对关节动作，保留双相机global/wrist原键名；由数据重建7维特征和归一化统计，
模型内部补到32维，额外缺失相机用掩码处理。默认对π0专家q/v投影和state/action相关投影添加LoRA，
不是全参数微调。执行计划保存在run同级的 `<run>.launch.json`，迁回结果时一并保存。

用固定seed打乱入选episode后按每任务尾部切分。审查文件的dataset_root须匹配数据集，
迁移时核对是同一数据集后只修改云端副本；入口不会自动忽略不匹配的剔除记录。
`--quality-reviews` 可显式指定文件，默认查找同名审查文件；没有文件则使用全部episode。
tokenizer需缓存或授权下载 `google/paligemma-3b-pt-224`，必要时用 `hf auth login`，
不要把登录凭据打进训练包。上传pi0_base不等于上传tokenizer。

### 8.3 模型迁回与单帧重加载

带回完整`pretrained_model/`：adapter权重/配置、policy config、processors及统计量、train_config；
继续训练还需同级`training_state/`。本地保留训练时匹配的基础权重，不只复制一个safetensors。

```bash
robotarm modeling check-pi0-lora -- \
  --checkpoint artifacts/checkpoints/pi0_lora_lf180_v1/checkpoints/020000/pretrained_model \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --index 0 --output artifacts/reports/pi0_lora/local_reload.json
```

显式本地base和dataset路径解决云端路径不同；不需要改原adapter。单帧检查只证明重载和输出形状，不证明任务成功。

### 8.4 π0训练与去噪次数HTML报告

```bash
# 训练曲线＋最佳/最终checkpoint抽样离线预测
robotarm modeling pi0-training-report -- \
  --run-dir artifacts/checkpoints/pi0_lora_lf180_v1 \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --samples 16 --repeats 2 --seed 1000 \
  --output artifacts/reports/pi0_lora/pi0_training_review.html

# 配对比较5/10次去噪，同观测、同噪声
robotarm modeling pi0-denoising-report -- \
  --checkpoint artifacts/checkpoints/pi0_lora_lf180_v1/checkpoints/020000/pretrained_model \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --samples 32 --near-start-samples 16 \
  --output artifacts/reports/pi0_lora/pi0_denoising_review.html
xdg-open artifacts/reports/pi0_lora/pi0_denoising_review.html
```

第一条加`--logs-only`只读日志；第二条已有同名JSON时加`--render-only`只重绘HTML。
报告带JSON/CSV明细。Flow loss、动作L1、逐关节deg/夹爪mm、离线耗时分别看；均不是闭环成功率。

去噪对照的 `--samples` 是每个划分的均匀起点数；额外起始样本
`--near-start-samples` 每episode最多一个，要求frame_index≤20且J2/J3均在±3°。
默认每起点使用两个固定噪声种子，排除padding，并审计J2/J3边界。
计时不包含视频解码、实时相机、CAN或预览，不能作为实际控制频率。

### 8.5 云端包上传、校验与启动

先按[打包命令](#12-辅助脚本)生成新的独立快照。包包含代码、LF180完整LeRobot数据、
pi0_base及processor、tokenizer缓存和审查副本，不包含原始session、ACT权重、Conda环境、
登录凭据或原生硬件SDK。首次安装Python依赖仍需联网；模型与tokenizer可离线加载。
本包用于离线训练，保留CLI源码不代表装有CAN、相机、PICO或仿真依赖。

把压缩文件和同名.sha256一起上传，在云端它们所在目录执行；名称替换成实际包名：

```bash
sha256sum -c RobotArm_pi0_lora_cloud_new_v1.tar.gz.sha256
tar -xzf RobotArm_pi0_lora_cloud_new_v1.tar.gz
cd RobotArm_pi0_lora_cloud_new_v1
python3 scripts/cloud/verify_bundle.py
```

校验会读取所有权重和视频，耗时取决于磁盘吞吐。不要向已有工作目录覆盖解压。
然后按[云端环境](#82-π0-lora微调与云端环境)安装独立Python 3.12环境，
使用云平台与GPU配套的CUDA/PyTorch，不运行本机统一硬件安装脚本。
激活环境后，每个新终端执行：

```bash
source scripts/cloud/cloud_env.sh
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print("BF16:", torch.cuda.is_bf16_supported())'
```

环境脚本切回包根目录并设置包内HF缓存、离线模型加载和WandB offline。
审查副本的dataset_root已由打包器改为包内相对路径，原文件保持不变。

```bash
# 默认预览；先检查生效参数
bash scripts/cloud/run_smoke.sh
# 实际20步试跑、保存和验证
bash scripts/cloud/run_smoke.sh --execute
# 重新加载adapter，离线预测一帧
bash scripts/cloud/check_smoke.sh
# 正式训练先预览，再执行
bash scripts/cloud/run_train.sh
bash scripts/cloud/run_train.sh --execute
```

脚本默认正式配置：batch=1、rank=16、alpha=32、lr=1e-4、chunk=50、n_action_steps=5、
seed=1000、eval_split=.2、steps=20000、save/eval=1000、BF16和梯度检查点。
可追加 `--batch-size`、`--n-action-steps`、`--steps`、`--output-dir` 等入口参数，
以预览输出为准；重复运行旧目录会被拒绝。先确认试跑峰值显存、loss和重载正常，再开正式训练。

```bash
# 云端另开终端：先激活训练环境
source scripts/cloud/cloud_env.sh
python scripts/train_monitor.py \
  --run-dir artifacts/checkpoints/pi0_lora_lf180_v1 --port 8420 --interval 2

# 在本机另开终端；替换实际SSH用户名/主机
ssh -N -L 8420:127.0.0.1:8420 USER@CLOUD_HOST
```

本机打开 <http://127.0.0.1:8420>。不要将无认证监控页面直接暴露公网。
带回文件见[迁回与重加载](#83-模型迁回与单帧重加载)；π0 flow-matching loss不能与ACT L1直接比较。
历史云端压缩包是静态快照，2026-10-13包不含后来新增的异步在线代码，部署使用本地当前版本。

## 9. 在线运行：先shadow，再执行

### 9.1 ACT shadow与审批

以下以ACT为例，显式使用20步，覆盖当前YAML里的30步。
确认与训练一致的相机装定、物体布局和起始姿态；先停其他遥操/相机程序。

```bash
TASK_CHECKPOINT=artifacts/checkpoints/act_cup_ring_dual_20261013_chunk50_180eps_lf_v1/checkpoints/090000/pretrained_model
TASK_SHADOW=artifacts/logs/shadow/act_090000_manual.jsonl
TASK_APPROVAL=artifacts/reports/runtime_approvals/act_090000_manual.json
robotarm modeling online-policy -- \
  --checkpoint "$TASK_CHECKPOINT" \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 --n-action-steps 20 \
  --duration 60 --report "$TASK_SHADOW"

# shadow结束并检查后，生成绑定权重和参数的审批
robotarm modeling approve -- --report "$TASK_SHADOW" --output "$TASK_APPROVAL"
```

shadow 读取相机/反馈，但不发送模型运动。检查完整日志中的 `watchdog_failures`、
`planner_underflow`、原始动作及限幅情况，不以“程序运行结束”代替验证通过。
默认日志同名重跑可能覆盖，需留存试次时换`--report`路径。

### 9.2 ACT真机执行

复用上一小节同终端定义的变量；若新开终端，先重新定义这三个变量。确保执行参数与审批一致：

```bash
robotarm modeling online-policy -- \
  --checkpoint "$TASK_CHECKPOINT" \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 --n-action-steps 20 \
  --duration 60 --report artifacts/logs/shadow/act_090000_manual_execute.jsonl \
  --approval "$TASK_APPROVAL" --execute
```

启动保持暂停，按X开始。`approval-required: false`当前会跳过审批文件检查，不表示安全检查或任务已通过。
需要强制审批时在YAML改为true；权重/运行参数改变后重做验证。

完全使用YAML默认参数时，可一条命令做shadow＋审批（不发送模型运动）：

```bash
robotarm modeling validate-runtime -- \
  --checkpoint "$TASK_CHECKPOINT" --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --seconds 60 --min-samples 1000
```

该入口不接受`--n-action-steps`等在线覆盖；它此时用YAML的30步，**不能冒充上面20步命令的验证**。
需要覆盖参数时用显式shadow→approve→execute。

| 输入 | 执行模式行为 |
|---|---|
| X | 开始/暂停一轮模型控制 |
| 长按Y 0.3秒 | 暂停并回零，夹爪保持 |
| Ctrl+C | 回零后退出，**不是急停** |
| Alt+S | 电子急停 |
| 视频窗口F/F11、双击、Esc | 切换全屏／返回窗口；窗口默认置顶、可调整大小 |

在线视频窗口获得焦点时也支持X/长按Y，`S`为电子急停。shadow不执行按键回零/急停。
`--duration`在shadow中为总时长，在execute中为每次按X启动的试次上限，到时暂停。

预测不合理先查输入/模型/时序，不以放宽软件限位或取消watchdog解决。

#### ACT限幅与日志字段

下表是 2026-10-19 配置快照，不代表已验证适合所有模型；修改后以实际启动日志为准。

| 参数 | 当前 ACT 值 | 含义 |
|---|---:|---|
| `--max-joint-step-deg` | 2.5 | 单周期相对上次命令的变化上限 |
| `--max-joint-lead-deg` | 6 | 命令相对反馈的超前上限 |
| `--max-gripper-step-mm` / `--max-gripper-lead-mm` | 8 / 16 | 夹爪步长 / 超前上限 |
| `--max-act-raw-joint-gap-deg` | 12 | 原始预测相对反馈最大关节差，超限暂停 |
| `--max-act-raw-gripper-gap-mm` | 30 | 原始夹爪预测相对反馈差上限 |
| `--max-act-raw-gripper-bound-tolerance-mm` | 1 | 原始夹爪边界容差，实际目标仍为合法开度 |
| `--max-act-clamp-hold-steps` | 10 | 同一限幅动作持续保持的周期上限 |
| `--max-inference-ms` / `--max-camera-age-ms` | 60 / 50 | 推理耗时 / 观测年龄上限，ms |
| `--n-action-steps` | YAML 当前为 30 | 在线消费步数；上述例子显式覆盖为 20，不改变chunk_size和训练loss |
| `--max-chunk-inference-ms` | 按策略覆盖 | 异步整块推理时限，与同步推理门槛区分 |
| `--report` | 显式指定 | 本次JSONL日志路径，用于诊断和审批，避免同名覆盖 |

真实步长/超前限幅时保持当前 ACT 动作，追上后再消费下一条；容差内夹爪端点修正不累计
等待跟随次数。超出原始预测门槛时，不靠多次小步追到异常目标。

| 日志字段 | 含义 |
|---|---|
| state / raw_action / safe_action | 反馈 / 原始预测 / 限幅目标，rad/m |
| act_tracking_action | 容差内夹爪端点修正后的比较目标 |
| action_clamped / act_tracking_clamped | 原始预测被修改 / 修正后仍需等待跟随 |
| act_gripper_boundary_corrected | 是否修正夹爪端点 |
| act_action_held / act_hold_steps | 保持旧动作 / 连续周期数 |
| command_sent / watchdog_failures | 是否真正下发 / 暂停原因 |

报告：`artifacts/logs/shadow/<run>_<step>_<hash12>[_execute].jsonl`；暂停时查完整记录。
healthy 只表示门控通过，任务成功率需现场记录。相机角色按 checkpoint 加载。
ACT 在线为 FP32；Diffusion/SmolVLA 默认异步规划，参数在各自 YAML 段。
审批控制周期门槛在 approve 段，包含超时占比和最长单拍周期数。

### 9.3 π0 LoRA异步shadow与实验执行

```bash
robotarm modeling online-policy -- \
  --checkpoint artifacts/checkpoints/pi0_lora_lf180_v1/checkpoints/020000/pretrained_model \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 \
  --num-inference-steps 5 --n-action-steps 20 \
  --max-chunk-inference-ms 600 --max-plan-age-ms 1250 --replan-lead-steps 12 \
  --duration 60 --report artifacts/logs/shadow/pi0_020000_manual.jsonl
```

先用上面命令只读验证。确认后在同参数命令追加`--execute --experimental-pi0-execute`，
并改用新的execute日志路径；若启用了审批还需提供匹配的`--approval`。
当前属于实验功能，未完成稳定闭环验收，不能因为shadow跑完就跳过故障检查。

π0 需要匹配的LoRA adapter、基础权重和保存的processors，使用异步动作块。
其 `n_action_steps=20` 是观测时刻起1秒窗口，会跳过过期前缀，不是推理完成后还能执行1秒。
真机还要求 `--experimental-pi0-execute`；它不是成功率或安全认证。
π0去噪次数是 `--num-inference-steps`，不是动作帧数或控制频率。
π0.5基础权重的影子测试不允许执行Piper动作。

#### π0运行依赖与队列时序

仅支持本项目7维绝对关节LoRA及匹配基础权重，需CUDA/BF16。本地缺少PEFT时补装，不重装全部依赖：

```bash
python -m pip install --no-deps peft==0.21.2
python -m pip check
```

模型仅由一个后台工作线程访问；控制侧不等待CUDA，但不保证实际循环达到20Hz。
`--replan-lead-steps 12` 表示队列剩余不超过12步时提交最新观测，不是额外执行12步。
`--max-chunk-inference-ms 600` 限制整块计算耗时，
`--max-plan-age-ms 1250` 限制观测年龄；推理超时、观测过期或队列耗尽均暂停，需重新按X，不自动恢复。
π0控制频率必须与训练FPS一致。未覆盖时去噪次数保留checkpoint配置；当前示例的checkpoint为10，
命令显式改为5，修改后必须重新比较离线误差。
窗口管理器不支持置顶/全屏时会提示并继续预览，不改变安全门控。

π0.5基础权重的实时影子测试示例（不允许执行）：

```bash
robotarm modeling online-policy -- \
  --checkpoint artifacts/pretrained/pi05_base --task "Pick up the cup" \
  --channel can_follower1 --device cuda --duration 30 --rate 1 --max-camera-age-ms 150
```

## 10. 仿真与模型可视化

这些命令操作仿真，不发送真机CAN。相机预览`realsense-stream`是本组的真实硬件例外。
GUI需要桌面显示环境；若必须指定X显示，按实际情况加`--display :1`，不要盲目照抄显示编号。

```bash
# 杯子任务查看器、带接触力保护的脚本基线
robotarm simulation viewer -- --view split
robotarm simulation force-pick-cup -- --episodes 5 --render

# 旧方块任务对照，不等同于杯子任务
robotarm simulation scripted-pick-place -- --episodes 5 --seed 0
robotarm simulation force-scripted-pick-place -- --episodes 5 --seed 0

# 采集仿真LeRobot数据，独立于真机数据目录
robotarm simulation collect-lerobot -- \
  --episodes 100 --action-frame absolute --include-wrist --width 320 --height 240 \
  --output data/simulation/lerobot/piper_cup_dual_new_v1 \
  --repo-id local/sim-piper-cup-dual-new-v1

# 用匹配仿真任务/表示的ACT闭环测试，先替换SIM_RUN和STEP
robotarm simulation run-act -- \
  --checkpoint artifacts/checkpoints/SIM_RUN/checkpoints/STEP/pretrained_model \
  --dataset-root data/simulation/lerobot/piper_cup_dual_new_v1 \
  --episodes 10 --device cuda --render
```

`collect-lerobot`只保存qualified轨迹；默认只全局相机，示例显式加腕部。
真实数据用640×480、仿真示例用320×240，不能不检查相机/动作语义就互换checkpoint。

```bash
# 仿真PPO训练/评估；不是给现有真机ACT直接接RL
robotarm simulation train-ppo -- --total-timesteps 5000000 --n-envs 16 --device cuda
robotarm simulation evaluate-ppo -- --episodes 10

# 旧方块环境的VLA实验分支
robotarm simulation small-vla -- collect --episodes 100
robotarm simulation small-vla -- train
robotarm simulation small-vla -- eval
robotarm simulation open-language-vla -- --render --interactive
robotarm simulation hierarchical-vla -- --render --interactive

# 仅在需要安装/修复PhysX GPU运行库时执行，会修改本机运行环境
robotarm simulation setup-physx

# 独立模型查看器；替换文件路径
robotarm simulation vis-mjcf -- --xml_path /PATH/TO/scene.xml
robotarm simulation vis-urdf -- --urdf_path /PATH/TO/model.urdf
```

MJCF用MuJoCo，URDF用Placo/Meshcat；文件参数是下划线，不是`--xml-path`/`--urdf-path`。
仿真环境、恢复采集、成功判据和输出位置详见[仿真手册](#10-仿真与模型可视化)。

### 仿真环境、采集语义与续采

#### `PiperPickCupToContainer-v1`（推荐）

目标是把带把手的白色杯子抓起，搬运到蓝色容器内，松开后稳定直立。环境包含：

- 真实参与碰撞的空心复合杯、把手、五面容器、工作垫和两件干扰物；
- 每回合随机杯子位置/朝向、容器位置、干扰物位置和机械臂初始关节；
- 与实采接近的 640×480 全局相机和腕部相机；
- 杯子位姿、目标位置、TCP 相对位置、双指接触力和阶段状态；
- 接近、抓取、抬升、运输、入箱、松爪、稳定的分阶段稠密奖励；
- 严格成功条件：杯体完全落在容器有效区域、保持直立、夹爪松开、杯子与机械臂均静止；
- 300 个控制步的回合长度，适合后续模仿学习或强化学习。

视觉与物理增强默认开启：三点式冷暖光、2048 阴影、PBR 粗糙度/金属度，以及按材质
区分的接触参数。橡胶垫、塑料容器、陶瓷杯和干扰物使用不同摩擦与恢复系数；每次 reset
还会随机改变环境曝光和白平衡。做确定性视觉回归时可在 `gym.make(...)` 中传入
`domain_randomization=False`；任务位置随机化仍然保留。

杯子任务的 `force-pick-cup` 脚本直接读取 PhysX 计算的左、右指尖接触力。只有双指接触
连续达到阈值才允许抬升，搬运中失去接触会停止后续放置并安全张开夹爪。它不会使用
`attach`、焊接或逐帧修改杯子位姿来伪造抓取。默认阈值为合力 2 N，可用
`--force-threshold` 调整。

默认容器随机范围使用固定末端姿态下的稳定可达域，而不是机械臂理论极限。这样脚本可作为
ACT 专家数据源，避免把边界处的 IK 绕行、擦碰和倾倒写进示范；后续训练出的策略仍可用
更宽的评估分布单独测试泛化能力。

脚本会分别输出环境 `success` 和数据质量门槛 `qualified`。只有流程正常完成且最终成功的
轨迹才是 `qualified=True`；搬运途中失去力反馈后碰巧掉进容器也会被拒绝。环境随机数已
统一接入 episode RNG，可用下面的命令逐条复现异常：

```bash
robotarm simulation force-pick-cup -- --episode-seed 444819 --render --display :1
```

可视化画面会同步显示专家当前阶段、双指接触力、六个关节的实际/目标角度，
以及夹爪实际/目标宽度。夹爪以默认 `2 mm` 的步长渐进闭合；双指接触稳定后，
目标宽度改为实际宽度减去 `2 mm` 预紧量并贯穿搬运过程，释放时再渐进张开。
这避免了旧专家持续输出 `0/70 mm` 二值动作。可分别用
`--gripper-step-mm` 和 `--gripper-preload-mm` 调整。

历史固定回归结果见实验台账。生成 ACT 数据时
必须只写入 `qualified=True` 的轨迹，并对被拒轨迹自动补采，不能用最终 `success` 单独筛选。

#### ACT仿真数据与质量门槛

首次采集和运行ACT见本节前面的命令。`scripts/regenerate_sim_act_dataset.sh` 会删除并原路径重建其指定的旧100条数据，不是普通续采入口；除非明确要丢弃该数据且已有备份，否则不要执行。

如果达到 `--max-trials` 时尚未凑满 100 条，已保存的 episode 会完整 finalize，之后可
更换 seed 并从当前总数继续补采（`--episodes` 表示目标总数，不是新增数量）：

```bash
robotarm simulation collect-lerobot -- \
  --episodes 100 --max-trials 200 --resume --seed 20260906 \
  --action-frame absolute --include-wrist --width 320 --height 240 \
  --output data/simulation/lerobot/piper_cup_dual_new_v1 \
  --repo-id local/sim-piper-cup-dual-new-v1
```

仿真 LeRobot 数据统一放在 `data/simulation/lerobot/`，真实数据仍放在
`data/lerobot/`。数据集额外包含 `simulation_provenance.json`，明确记录
`source=simulation`、环境、控制器、seed、合格/拒绝轨迹及表示方式。

- `observation.state`：绝对 `[J1..J6, gripper_width]`，单位为 rad/m；
- `action[t]`：默认记录专家实际执行后的下一帧 `observation.state[t+1]`，单位为
  rad/m；仍是 `pd_joint_pos` 可执行的绝对关节/夹爪目标，但不会混入 IK 内部目标
  切换产生的虚假跳变；
- 只有显式传入 `--action-frame relative` 时，才记录 `state[t+1] - state[t]`；
- 默认只记录全局相机；当前真机 ACT 常用双相机，需显式`--include-wrist`并核对模型输入；
- 仅提交 `qualified=True` 的 episode，失败轨迹清空后自动补采。
- 每次 reset 后先用零位移动作等待至少 20 帧，并要求连续 8 帧关节变化不超过
  `0.003 rad` 后才开始记录；每条轨迹的最大单帧关节增量写入 provenance。
- 仿真从与专家相同 IK 分支上的高位安全 Home 开始，第一段接近动作限制为
  `10 mm/帧`，避免旧 Home 在最初几帧触发 J4/J6 构型切换；这些起始帧保留用于
  训练，但应当是连续渐变而不是被裁掉。
- 默认拒绝单帧关节变化超过 `0.4 rad`、夹爪变化超过 `10 mm` 或长度超过 650 帧
  的轨迹；机械臂驱动力上限由 100 降为 30，使 20 Hz 轨迹更接近真机速度。

当前固定抓取放置任务先采 100 条作为 ACT 基线；建议按 episode 切分为 80 条训练、20 条
验证，不要逐帧随机切分，以免同一轨迹同时进入训练集和验证集。

续采必须保持原数据相机与动作配置一致；上例对应本节的双相机初次采集命令。

`run-act` 默认从数据集的 `simulation_provenance.json` 自动识别 absolute/relative
动作语义，分别使用 `pd_joint_pos` / `pd_joint_delta_pos` 控制器闭环执行；默认分别把
单帧关节与夹爪变化限制在 `0.10 rad` 和 `5 mm`，并输出成功率 JSON。

这里的仿真力与真机反馈含义不同：仿真值是接触求解器输出；真机 pyAgxArm 从 CAN
`0x2A8` 的夹爪反馈帧读取 `force` 字段（单位 N）。该值属于驱动器提供的夹持力反馈，
不是独立六维腕力传感器的测量值。

环境 ID：`PiperPickCupToContainer-v1`。默认查看器已经切换到这个环境。代码不依赖
联网下载；这是为了保证新机器和批量训练节点能直接复现。若后续要增加照片级素材，可使用
ManiSkill 官方支持的 YCB 资源（例如 `ycb:025_mug`），但需先单独下载 YCB 资产包并
记录其许可证与版本。

#### `PiperPickCubeToBox-v1`（兼容环境）

原有红方块入蓝箱任务继续保留。现有 scripted、force-scripted、PPO 和 VLA 示例代码
仍默认面向这个环境，避免已有 checkpoint 和实验脚本失效；迁移训练管线时应显式传入新
环境 ID，并重新采集演示/训练模型，不能把旧 checkpoint 直接当作杯子任务模型。

`small-vla` 的默认数据路径为 `data/simulation/piper_vla_demos.h5`。仿真训练
结果默认写入 `artifacts/simulation/`。

## 11. 全部CLI功能清单

以下与当前注册入口逐一对应，共46个（data-acquisition 11、modeling 20、simulation 15）。
每个Python入口的完整参数通常可用`robotarm <组> <命令> -- --help`查看；
`train`/`train-with-holdout`是shell包装，按第6节的位置参数格式使用。

### data-acquisition：硬件、遥操、原始数据

| 完整入口 | 功能／使用位置 |
|---|---|
| `robotarm data-acquisition read-status` | 只读机械臂反馈，第3节 |
| `robotarm data-acquisition enable-control` | enable/disable/reset，第3节，执行会改变硬件状态 |
| `robotarm data-acquisition home-piper` | 独立回零，第3节，执行会运动 |
| `robotarm data-acquisition restore-piper-role` | 恢复Follower角色，第3节，可能失去保持力 |
| `robotarm data-acquisition collect-leader-follower` | 自定义软件主从采集／只读／纯控制，第4节 |
| `robotarm data-acquisition collect-piper` | 手柄遥操采集、手柄监控、相机检查，第4节 |
| `robotarm data-acquisition xr-bridge` | PICO后台桥接，第4节 |
| `robotarm data-acquisition piper-xr` | 单/双臂PICO统一入口，第4节 |
| `robotarm data-acquisition piper-bimanual-xr` | 独立双臂PICO入口，参数见`-- --help`及PICO手册 |
| `robotarm data-acquisition merge-piper-raw` | raw→raw独立复制、编号与来源映射，第5节 |
| `robotarm data-acquisition convert-piper-raw` | raw→LeRobot训练格式，第5节 |

### modeling：训练、评估与部署

| 完整入口 | 功能／使用位置 |
|---|---|
| `robotarm modeling check-environment` | 依赖、数据与CUDA检查，第6节 |
| `robotarm modeling train` | LeRobot训练包装，第6节 |
| `robotarm modeling train-with-holdout` | 固定留出比例的兼容包装，第6节 |
| `robotarm modeling train-pi0-lora` | 独立π0 LoRA训练计划／执行，第8节 |
| `robotarm modeling check-pi0-lora` | LoRA重新加载＋单帧离线预测，第8节 |
| `robotarm modeling convert-dataset` | 与convert-piper-raw相同的兼容入口，第5节 |
| `robotarm modeling dataset-browser` | 数据浏览、人工审查、逐episode裁剪与新数据集导出，第5节 |
| `robotarm modeling normalize-piper-metadata` | 旧pick-up元数据修复工具，非通用任务重标记，修改前需备份 |
| `robotarm modeling trajectory-studio` | Open-loop验证平台，第7节 |
| `robotarm modeling evaluate` | checkpoint指定episode抽样误差，第7节 |
| `robotarm modeling training-report` | 训练日志汇总、CSV/曲线/Markdown，第7节 |
| `robotarm modeling act-latent-report` | ACT A/B/C/D离线L1，第7节 |
| `robotarm modeling act-training-report` | ACT续训日志＋已有潜变量报告汇总，第7节 |
| `robotarm modeling pi0-training-report` | π0训练日志＋抽样动作误差，第8节 |
| `robotarm modeling pi0-denoising-report` | π0 5/10次去噪配对报告，第8节 |
| `robotarm modeling pi-offline-benchmark` | π0/π0.5基础模型离线延迟，第8节 |
| `robotarm modeling pi05-offline-benchmark` | π0.5兼容延迟入口，第8节 |
| `robotarm modeling online-policy` | 实时shadow／显式执行，第9节 |
| `robotarm modeling approve` | 从shadow报告生成绑定审批，第9节，不发送动作 |
| `robotarm modeling validate-runtime` | 自动shadow＋审批，第9节，会占用CAN/相机但不发送模型动作 |

### simulation：仿真与可视化

| 完整入口 | 功能／使用位置 |
|---|---|
| `robotarm simulation realsense-stream` | 真实相机预览，不是仿真，第3节 |
| `robotarm simulation viewer` | ManiSkill场景/相机查看器，第10节 |
| `robotarm simulation scripted-pick-place` | 旧方块任务脚本基线，第10节 |
| `robotarm simulation force-scripted-pick-place` | 旧方块任务接触式脚本基线，第10节 |
| `robotarm simulation force-pick-cup` | 杯子任务力保护基线，第10节 |
| `robotarm simulation collect-lerobot` | 保存合格仿真示范为LeRobot，第10节 |
| `robotarm simulation run-act` | ACT闭环仿真，第10节 |
| `robotarm simulation train-ppo` | 仿真PPO训练，第10节 |
| `robotarm simulation evaluate-ppo` | 仿真PPO评估，第10节 |
| `robotarm simulation small-vla` | collect/train/eval实验分支，第10节 |
| `robotarm simulation open-language-vla` | 开放语言实验分支，第10节 |
| `robotarm simulation hierarchical-vla` | 分层策略实验分支，第10节 |
| `robotarm simulation setup-physx` | 安装/修复GPU运行库，第10节，改变本机环境 |
| `robotarm simulation vis-mjcf` | MuJoCo查看MJCF，第10节 |
| `robotarm simulation vis-urdf` | Placo/Meshcat查看URDF，第10节 |

WeGo采集和旧replay-compare已不是当前入口；不要照历史实验记录调用已移除命令。
仿真PPO/VLA保留作实验，不代表已接入真实Piper的强化学习闭环。

## 12. 辅助脚本

这些功能未都注册成CLI，但仍可使用：

| 脚本／模块 | 用法与副作用 |
|---|---|
| `scripts/create_unified_env.sh` | 本机统一环境安装/更新，会改变依赖，不是日常启动命令 |
| `scripts/train_monitor.py` | `--run-dir RUN --port 8420 --interval 2`，实时loss网页 |
| `scripts/check_xr_frame.py` | `--hand right --yaw 0 --mirror none`，只读坐标诊断 |
| `scripts/xr_monitor.py` | `--yaw 0 --mirror lr`，XR状态/坐标观察；`--no-plot`关闭图窗 |
| `scripts/build_pi0_cloud_bundle.py` | 打包训练代码/数据/权重/tokenizer，不上传；见下方限制 |
| `scripts/cloud/run_smoke.sh` / `run_train.sh` | 云端试跑/正式训练包装，默认计划，追加`--execute`执行 |
| `scripts/cloud/check_smoke.sh` | 试跑权重重新加载，离线预测一帧 |
| `scripts/cloud/verify_bundle.py` | 在解压的云端包内验证逐文件校验和 |
| `robotarm.modeling.quality_filter` | `python -m ... --dataset-root ... --derive-from ...`，迁移同源审查结果 |
| `robotarm.modeling.reports.best_practice_figures` | 从已有JSON重画台账PNG/SVG，不加载模型 |
| `scripts/analyze_pi0_lora.py` / `compare_pi0_denoising.py` / `analyze_act_resume.py` | 旧兼容壳，优先用第7/8节统一报告命令 |

```bash
python scripts/check_xr_frame.py --hand right --yaw 0 --mirror none
python scripts/xr_monitor.py --yaw 0 --mirror lr --no-plot
python -m robotarm.modeling.reports.best_practice_figures
```

云端打包器**当前固定LF180数据集和pi0_base**，不是任意数据集打包器；不要以为改输出名就换了训练数据。
替换下面tokenizer缓存仓库目录（其中应有`refs/`、`snapshots/`），使用不存在的新输出名：

```bash
python scripts/build_pi0_cloud_bundle.py \
  --bundle-dir artifacts/cloud_exports/RobotArm_pi0_lora_cloud_new_v1 \
  --tokenizer-repo /PATH/TO/models--google--paligemma-3b-pt-224
```

会创建独立目录、`.tar.gz`和`.sha256`，占用较多磁盘；训练配置及权重迁回说明见[云端手册](#85-云端包上传校验与启动)。
旧日期压缩包是静态快照，不会自动包含今天修改的代码。

## 13. 常见问题

| 现象 | 先检查什么 |
|---|---|
| `robotarm: command not found` | 激活robotarm环境、确认editable安装；不要直接重建已有可用环境 |
| CAN名称不存在 | `ip -details -brief link show type can`，确认角色后替换channel；名字不证明主从 |
| 相机打不开／No device／帧超时 | 退出另一个占用程序、核对serial与USB连接，先单路`--color-only`检查 |
| 双相机采集只录到一路 | 手柄默认wrist为空，显式传腕部序列号，查看session.json相机角色 |
| 从臂已使能而拒绝启动 | 确认现场状态后加`--allow-enabled-start`，不要为省提示反复使能/失能 |
| 输入参数不识别 | 顶层转发使用`--`；训练`--batch_size`与在线`--n-action-steps`命名不同 |
| 合并/转换输出已存在 | 改新输出名或使用已有结果；`--overwrite`可能删除结果，不是“继续” |
| 数据仍是raw无法训练 | 先运行第5.2节转换，训练输入应有meta/info.json、data、videos |
| 转换后数量与目录名不一致 | 看meta/info.json，complete/质量筛选可能改变数量，不按名字推断 |
| Pi0需要run shadow first | 先真实shadow检查；实验执行还需专用开关及可能的审批，不只补一个标志就算验证 |
| clamp/raw gap/bound/underflow触发 | 查看本次JSONL、预测/反馈/限幅/时间戳，不盲目放宽门控 |
| 配置改了却没生效 | 重启进程，检查CLI是否覆盖YAML、policies是否覆盖通用值，看启动日志 |
| 报告能打开但图片/数据缺失 | artifact可能没迁移；保存HTML和JSON/CSV及引用资源，台账图在docs/best_practice_assets |
| pip check报NumPy/cmeel-boost | 先看[环境说明](#11-环境版本与验收)，不要为消除提示随意升级NumPy |

常用服务：数据浏览8766，Open-loop8765，实时loss8420，XR bridge60062；端口占用时改各自参数。
云端监控默认本机监听，使用SSH端口转发，不为方便把无认证页面直接暴露公网。

## 14. 测试与文件管理

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s tests -p 'test_online_policy_safety.py' -v
git diff --check
```

| 目录 | 职责 |
|---|---|
| `src/robotarm/` | 产品代码，CLI与可复用离线分析；不存本次运行结果 |
| `src/robotarm/config/` | 当前默认值；不代替checkpoint中的历史参数 |
| `scripts/` | 环境/云端编排、诊断、兼容入口；不再复制报告实现 |
| `tests/` | 可重复的无硬件回归测试，不是实验结果目录 |
| `docs/best_practice_assets/` | 文档图片与台账图表，保留资源路径 |
| `data/` | 原始会话与LeRobot数据集 |
| `artifacts/` | 权重、日志、报告、云端快照 |
| `archive/` | 历史资料/旧快照，不是当前运行入口 |

`ROBOTARM_DATA_ROOT`、`ROBOTARM_ARTIFACTS_ROOT`、`ROBOTARM_CONFIG_DIR` 可重定向根目录。
不要按文件年龄批量删除测试、checkpoint、质量审查或原始数据；分类和清理规则见[项目维护](#14-测试与文件管理)。

### 目录与边界

```text
src/robotarm/
  core/                 相机、机器人接口、运动学
  teleop/               手柄、主从、XR控制
  record/               录制、转换与数据格式
  modeling/             训练/验证/执行入口
    reports/            离线报告实现和共享HTML组件
    trajectory_studio/  Open-loop平台的后端和前端
  simulation/           仿真（不是废弃的真机代码）
scripts/                环境、云端编排、只读诊断、旧命令兼容壳
tests/                  单元/回归测试；硬件使用mock
docs/best_practice_assets/  台账图表和文档资源（保留，不是旧结果缓存）
data/                   原始数据和转换数据
artifacts/              实验产物，不进入源码包
archive/                历史资料和快照，不作为当前入口
```

`src/robotarm/modeling/dataset_browser.html` 和
`trajectory_studio/frontend/*` 是运行所需前端，不是可以删除的历史报告。
`src/robotarm/simulation/viewer.py` 是仿真入口，也不等于以前清理掉的viewer.html。

### 哪些文件保留，哪些可以整理

| 类型 | 处理原则 |
|---|---|
| 原始session、LeRobot meta/data/videos | 保留；不按日期或名字推断无用 |
| checkpoint/pretrained_model | 部署需完整config、权重、processors及其统计；π0还需基础权重/tokenizer |
| checkpoint/training_state | 续训需要；部署不需要不等于可以自动删除 |
| WandB离线文件、train_config、入选episode列表 | 保留，训练曲线/划分的原始证据 |
| dataset_quality、runtime_approvals | 不是缓存；保存人工审查和审批依据 |
| HTML/JSON/CSV实验报告 | 至少保留台账引用的一组及复现方法；HTML与明细不是重复证据 |
| trajectory_studio派生视频/缓存 | 确认无引用且原数据仍在后才逐个清理；不要直接删整个logs |
| `__pycache__`、`.pytest_cache` | 可重建；不用为整洁反复删除，已被.gitignore排除 |
| `.cache/huggingface` | 可能含离线tokenizer/数据缓存与凭据，不按普通临时文件批量清理 |
| cloud_exports压缩包及解压目录 | 是独立快照；确认已上传/备份再按明确目标处理，不自动同步 |
| tests中的旧bug回归 | 保留；功能仍存在就不是过时实验 |

有疑问的产物先列清单、核对引用，再移动到明确的归档目录。不要运行针对工作区、
数据根或环境根目录的递归删除；本次整理没有删除数据、权重或实验报告。
`archive/`、`artifacts/`被Git忽略，本机有文件不等于已备份到仓库。

### 新增功能与报告

1. 可复用逻辑放`src/robotarm/`，在所属组的`WORKFLOWS`登记入口；CLI保持`-- --help`可用。
2. 离线报告优先放`modeling/reports/`，公用表格/图表使用`reports/html.py`。
   不让一个实验脚本导入另一个实验脚本的内部实现。
3. 生成文件写入`artifacts/reports/<类别>/`，不要写回源码目录。
   `approve`历史默认输出仍保留兼容；使用时显式传`--output artifacts/reports/runtime_approvals/<名称>.json`。
4. 报告名带run、checkpoint或日期；重跑需保留证据时使用新文件名。
   在线默认JSONL同一checkpoint可能被后续运行覆盖，需保留试次时显式传`--report`。
5. README保留完整的常用操作命令、全量CLI清单及参数说明，不只给跳转入口；长说明也集中在README对应小节，不另建重复操作手册。
   实验的实测数值、训练参数、失败和未验证假设写入Best Practice。新增入口时运行`test_readme_workflows`核对覆盖。
6. 新增回归测试并运行相关测试，收尾再跑全部测试；不以测试通过代替硬件验收。

### 兼容入口

| 旧命令 | 首选命令 |
|---|---|
| `python scripts/analyze_pi0_lora.py ...` | `robotarm modeling pi0-training-report -- ...` |
| `python scripts/compare_pi0_denoising.py ...` | `robotarm modeling pi0-denoising-report -- ...` |
| `python scripts/analyze_act_resume.py ...` | `robotarm modeling act-training-report -- ...` |

旧脚本只转发，不复制算法；历史报告里的复现命令仍可用。
`training_report.read_history`保留给实时监控读取单文件；事后汇总用`read_run_history`合并续训、去重。
