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
| SmolVLA直接训练、参数、续训 | [SmolVLA训练](#smolvla直接微调) |
| Open-loop、A/B/C/D、HTML报告 | [离线分析](#7-分析模型) |
| π0微调、迁回、延迟测试 | [π0工作流](#8-π0与π05离线训练测试) |
| 真机执行、审批、按键 | [在线运行](#9-在线运行act直接执行与可选shadow) |
| HG-DAgger接管采集、退出后重开 | [HG-DAgger](#hg-dagger人工接管与纠正数据) |
| HIL双终端运行、暂停退出与续训 | [HIL-SERL运行与续训](#hil-serl当前010000模型的运行与续训) |
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
ACT原始关节差软告警/严重遥测阈值为12°/15°、原始夹爪差告警为30mm；π0 的原始关节差硬门槛为15°。
这些是配置快照，**不是新推荐值**。
本次未修改这些值。2026-10-28当前ACT真机抖动测试采用显式
`--n-action-steps 1 --temporal-ensemble-coeff 0.01`；这两个参数必须配套，不能只写其中一个。
不要将当前YAML默认值当成已验证的真机运行参数。

`approval-required: false` 当前跳过审批文件检查，不代表验证通过；仍需执行开关和运行时门控。
开启审批请按[部署流程](#9-在线运行act直接执行与可选shadow)操作，不为绕过失败日志而放宽门槛。

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

#### 固定系统 CAN 名称（重启、重新插拔后保持）

2026-10-22 用户确认的适配器对应关系如下；内核分配的`can0/can1`编号可能随插拔变化，
规则按唯一序列号匹配，不按编号或USB插口匹配。更换适配器后需重新核对序列号，不能直接复用。

| 当时接口 | USB-CAN 序列号 | 固定名称 |
|---|---|---|
| `can0`（从臂） | `004300414148570D20343133` | `can_follower1` |
| `can1`（主臂） | `003200324648570A20303731` | `can_leader1` |

先核对当前硬件标识：

```bash
udevadm info --query=property --path=/sys/class/net/can0
udevadm info --query=property --path=/sys/class/net/can1
```

从项目根目录安装规则（需要管理员权限；已有同名自定义规则时先检查内容，勿直接覆盖）：

```bash
sudo install -m 0644 scripts/system/10-robotarm-can-follower1.link /etc/systemd/network/10-robotarm-can-follower1.link
sudo install -m 0644 scripts/system/10-robotarm-can-leader1.link /etc/systemd/network/10-robotarm-can-leader1.link
sudo udevadm control --reload
```

安装规则不会立即修改当前接口。**退出所有机械臂控制/采集程序，确认机械臂安全静止并有可靠支撑后**，
若接口与上表仍一致，可立即改名并按1Mbps重新启用；下面的`down`会中断CAN通信，不是机械臂安全停止命令：

```bash
sudo ip link set can0 down
sudo ip link set can1 down
sudo ip link set can0 name can_follower1
sudo ip link set can1 name can_leader1
sudo ip link set can_follower1 up type can bitrate 1000000
sudo ip link set can_leader1 up type can bitrate 1000000
ip -details -brief link show type can
```

也可在机械臂安全停机后重新插拔适配器或重启PC，让规则自动命名，再运行上面的两条`up`命令。
`.link`只负责固定名称，不负责开机自动启用CAN，也不改变机械臂主从角色或使能状态。
接口已是目标名称时不重复执行`can0/can1`改名命令；不要交换这两块适配器所连接的机械臂。

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

# 一条命令复位主臂和从臂的软件急停；执行前同时托稳两臂
robotarm data-acquisition reset-both-arms -- \
  --confirm-drop --execute

# 六关节回零，夹爪保持开度
robotarm data-acquisition home-piper -- \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --speed-percent 10 --timeout 60 --execute

# 需要退出主臂角色时，切回Follower角色；不发送关节运动/使能命令
robotarm data-acquisition restore-piper-role -- \
  --interface socketcan --channel can_leader1 --firmware v189 --confirm-drop --execute
```

角色切换后按提示重新上电。`--speed-percent`是独立回零速度，不能当作模型/遥操的执行速度。
`reset-both-arms`默认复位`can_follower1`和`can_leader1`，先确认两路CAN都有反馈，再连续发送两条复位命令并等待新的正常状态。不加`--execute`只做双臂通信预检。该命令不能解除被按下的实体急停按钮。

## 4. 采集与遥操

### 主从臂：当前自定义实现

```bash
robotarm data-acquisition collect-leader-follower -- \
  --execute --allow-enabled-start --parallel-home \
  --leader-channel can_leader1 --follower-channel can_follower1 \
  --global-camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --setup-id leader_follower_v1 \
  --task "Pick up the KS11 FLVV component and place it in the target black area with hole"
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
终端状态行每0.5秒刷新，持续显示本次会话已保存条数；录制中还会显示正在采第几条及当前帧数。
保存、丢弃或产生未完整episode后会立即打印三类累计计数，丢弃条目不会计入已保存数量。
双相机窗口预览最多刷新10Hz，不改变控制或录制频率。若跟随仍卡顿，跟随中按`s`查看
`[timing]`：`actual`应接近目标频率，`overruns`应很少；`record p95/max`偏高可在采集命令中
加`--no-preview`复测，`move_js p95/max`偏高则检查从臂CAN接口和总线负载。
对比纯控制模式与正式录制模式，可进一步区分相机/编码负载和控制链路问题。
`--setup-id`标记装定批次；`--task`必须写实际示教任务，例如仅抓取与抓取后放入孔中不要误用同一个描述。
新会话默认写入`data/raw/piper_raw/<时间戳>/`。相机预览、对齐、跟随、回零及未录制退出
都不会创建时间戳目录；第一次按`r`开始 episode 时才创建`session.json`和`episodes/`。

#### 主从对齐、参数与动作记录

软件桥接读取零力主臂，再用 MOVE_JS 控制从臂；不要让另一程序或固件直连同时控制从臂。
终端单键，不用回车。

| 键 | 功能 |
|---|---|
| l | 请求主臂零力拖动 |
| a | 固定按键时的主臂姿态，从臂按对齐速度推进，完成后保持 |
| f | 开始/暂停跟随；对齐中取消对齐 |
| s | 打印两臂模式、角度、目标、误差和反馈年龄 |
| r / d | 开始/保存 episode / 丢弃 episode |
| h | 非录制时双臂回零并闭合夹爪；回零前若在 f 跟随，完成后自动恢复 |
| q | 保持位置退出 |
| x | 从臂急停，主臂尝试使能保持后退出 |

一条 episode：l → a → 等待对齐完成 → f → r → 完成任务 → r。
`a` 对齐默认使用 20% MOVE_J、关节目标推进上限 15°/s；与 `f` 跟随的 `--speed` 独立。
例如初始最大关节差 60°，默认目标推进约 4 秒（原 5°/s 为 12 秒），实际到位可能更慢，
并仍要求所有关节误差不超过 2°、稳定 0.5 秒才完成。需要恢复原慢速时加
`--alignment-speed 10 --alignment-rate-deg-s 5`。对齐目标仍冻结在按 `a` 时的主臂姿态，
中途按 `f` 取消；单周期限幅、跟随误差和反馈超时保护仍生效。
启动或保存后可按 h，完成后保持使能且不自动录下一条；按 h 前若在 f 跟随模式，程序会用
回零后的实测位置重新建立 MOVE_JS 保持目标并自动恢复跟随；原本暂停则继续暂停。主臂用固件回零指令，
速度由控制器决定；从臂用 10% MOVE_J。部分固件在主臂回零期间不发布关节帧，程序会在
短时等待后退出固件回零态、恢复普通反馈，并用角色切换后的新六关节反馈确认零位；若固件
回零尚未完成，则自动用有反馈验证的 10% MOVE_J 补完，不会仅凭等待时间或
`motion_status` 判定成功。主臂夹爪若 1.5 秒内完全没有闭合进展，会保持实测开度、打印警告
并继续恢复跟随，不再使已经回零的双臂急停；重新进入跟随后操作一次主臂夹爪即可同步。
默认顺序模式回零中退出只急停当前臂。
确认两条回零轨迹互不相交且周围无人员、杯子和障碍物后，可在启动命令增加
`--parallel-home`：按 h 后主臂固件回零和从臂 10% MOVE_J 同时开始，各自独立验证到位；
任意一臂失败或中断都会电子急停两臂。此选项不提供双臂碰撞检测。

| 参数 | 含义 |
|---|---|
| `--joint-signs` / `--joint-offsets-deg` | 关节映射方向 / 零点偏移，默认同向、零偏移 |
| `--control-rate` | 默认跟随 50 Hz，录制另为 20 FPS |
| `--alignment-speed` | `a` 对齐的 MOVE_J 速度百分比，默认 20，上限 30；不改变回零速度 |
| `--alignment-rate-deg-s` | `a` 对齐的关节目标推进上限，默认 15°/s，上限 30°/s；仍受每周期最大步长限制 |
| `--max-step-joint-deg` | 每周期关节目标最大变化，当前默认 3° |
| `--max-following-error-deg` | 从臂反馈与目标的误差上限 |
| `--max-gripper-following-error-mm` | 夹爪张开命令领先反馈的停机阈值，默认20 mm；闭合方向超过该值视为物体接触并继续 |
| `--gripper-force` | 从臂夹爪命令力，主从数采默认2.0 N；易变形物体可单次降为1.0 N，程序上限5 N |
| `--joint-limit-margin-deg` | 命令距标称限位裕量，至少 0.5° |
| `--stale-timeout` | 从臂反馈超时，默认 0.2 秒 |
| `--leader-init-timeout` | 等待主臂完整初值，默认 30 秒 |
| `--home-timeout` / `--home-gripper-speed-mm-s` | 回零各阶段超时 / 夹爪闭合速度，默认 60 秒 / 10 mm/s |
| `--parallel-home` | 按 h 时双臂并行回零；默认关闭，任一侧异常会急停两臂 |

主臂已在零力模式时按提示轻移三个关节对并操作夹爪。主臂静止可保持最后输入，
从臂始终检查反馈新鲜度。a 对齐使用 10% MOVE_J、目标最大 5°/s；跟随前误差须在 2° 内
（配置阈值更小时取更小值），按 s 看详情。

记录 state 为从臂六关节 rad + 夹爪 m，action 为实际下发的限速/限幅后目标。抓到实体后，
从臂夹爪反馈会停在物体宽度，而action继续保留主臂要求的较小闭合宽度；这是正确的接触示教，
程序会提示“闭合受阻”并继续跟随和录制。只有张开目标比反馈大超过
`--max-gripper-following-error-mm`才按夹爪未响应停止。
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
  --channel can_follower1 --firmware v189 --scale 0.8

# 双臂只读；确认后加 --live
robotarm data-acquisition piper-xr -- \
  --arms 2 --xr-bridge --left-channel can_follower1 --right-channel can_leader1 \
  --firmware v189 --scale 0.8
```

单臂长按右 Grip+B 使能、Grip接合/松开保持、长按右 Grip+A 回零、右摇杆按下急停。
双臂长按同侧 Grip+Y/B 使能、同侧 Grip+X/A 回零，任一摇杆按下急停两臂；也兼容把该物理键映射到 Trigger 的控制器配置。
`--scale`调位移比例，方向与安全工作空间配置见[PICO手册](#pico控制参数与排查)。
独立双臂入口`robotarm data-acquisition piper-bimanual-xr`也保留；优先用上面的`--arms 2`统一入口。

### PICO控制参数与排查

统一入口 `piper-xr` 使用 `--arms 1`（默认）控制单臂，`--arms 2` 控制双臂。
单臂使用右手柄和 `--channel`；双臂使用左/右手柄与各自的
`--left-channel` / `--right-channel`。双臂模式读取 `xr.yaml` 的
`piper-bimanual-xr` 配置，单臂模式读取 `piper-xr` 配置。
当前摆放对应左手柄 → `can_follower1`、右手柄 → `can_leader1`；CAN 中的主从角色名不等于物理左右。
双臂夹爪最大开度默认 100mm，与单臂一致；Trigger 按到底请求最大张开。旧命令若显式指定相反通道
或 `--gripper-max-mm 70`，会覆盖新默认值，需要同步修改并重启遥操进程。

保持XRoboToolkit PC Service和bridge运行，启动命令见上节。无`--live`仍需读取CAN反馈，但不使能、不发送运动目标；确认后在同参数命令加`--live`，长按B才使能。

`--firmware v189` 跳过逐通道固件探测（S-V1.8-9 / S-V1.9-0 都用 v189 驱动），
启动更快，也不再打印探测线程的 `Bad file descriptor` 回溯。`--scale 0.8` 表示
手走 10 cm、末端走 8 cm；当前默认 `2.0` 表示慢速、小幅输入按 2:1 映射，机械臂
移动距离约为此前默认值的两倍。首次真机可显式使用较低比例核对方向和工作空间。

长按右 Grip+B 约 1 秒；看到 `Piper enabled. Joint motor enables: [True, True, True, True, True, True]` 后，松开 B 即以当时手柄位置建立新基准并进入跟随，不必额外松开再握 Grip。不加 `--live` 时组合键只打印提示，不操作硬件。短按无动作，`--record` 已禁用。Grip 达到 0.9 接合，低于 0.7 释放；默认以位置为优先、尽量保持接合时的法兰朝向（奇异位形附近允许朝向让步）。添加 `--orientation` 可让朝向随右手柄旋转。右 Trigger 仍只控制夹爪；长按右 Grip+A 回零并保持使能，右摇杆按下急停。命令修饰键阈值默认 0.8，可用 `--command-modifier-threshold` 调整。

PICO 默认配置独立存放在 `src/robotarm/config/xr.yaml`，不再读取共享文件的 `piper-xr` 段。当前单臂默认控制频率 100 Hz、双臂 50 Hz，平移目标限速 0.12 m/s，单关节命令限速 0.45 rad/s（约 25.8°/秒）。使用 `--translation-speed`（m/s）和 `--joint-speed`（rad/s）调整；同时保留每步限幅，取更严格者。控制周期延误时不补发大步长。不要复制旧示例中的每步限幅到新的频率上，否则会改变速度上限。目标推进基于限幅后关节命令的正解，减少目标积累。


| PICO参数 | 含义 |
|---|---|
| `--translation-speed` / `--joint-speed` | 平移m/s、单关节rad/s上限 |
| `--max-step-translation` / `--max-step-joint` | 单周期m/rad上限，与速度×周期取较小值 |
| `--max-target-lead` / `--max-relative-translation` | 相对反馈超前／单次接合范围，单位m |
| `--xyz-min` / `--xyz-max` | 基座坐标系工作空间；双臂分别用left-/right-前缀 |
| `--posture-anchor engage` / `seed` | 固定接合姿态参考／随上一周期种子变化 |
| `--engage-settle-seconds` / `--engage-deadband-m` | 接合后重建参考的时间／首条运动命令所需的主动手部位移；默认0.25秒、2毫米 |
| `--verbose-status` | 临时恢复高频按键、机械臂状态、IK、目标和关节诊断；默认关闭以避免终端刷屏 |

XR 超过 0.2 秒不更新时脱离；恢复后必须松手再接合。松手/断流/逆解拒绝时发送当前关节位置作为保持目标，正常退出（包括回零期间 Ctrl+C）时发送当前位置保持命令，保持使能，不发送软件急停；运行异常或右摇杆主动急停仍发送软件急停。停止效果仍需在真实控制器上验证。回零期间也监测 XR 断流、退出信号和右摇杆急停。

离线验证：

```bash
python -m unittest discover -s tests -p test_pico_control.py -v
```

#### 双手柄控制双臂

通过统一入口 `piper-xr -- --arms 2` 启动双臂控制：默认左手柄控制 `can_leader1`，右手柄控制 `can_follower1`。
旧入口 `piper-bimanual-xr` 仍可使用，行为相同。
升级代码后必须重启 XR bridge，因为旧 bridge 消息不包含左手柄字段。
双臂默认使用 `--tracking-frame head`：每次 Grip 接合时冻结头显水平朝向，将操作者
前方/右方/上方映射到机械臂 `+X/-Y/+Z`。PICO 重定位或操作者站位变化不再要求手工修改
世界坐标 yaw；接合后转头也不会改变本次映射。两侧 mirror 默认均为 `none`，仍可分别覆盖。

确认两个手柄方向和两台机械臂反馈后再增加 `--live`。首次真机建议缩小映射比例、
一次只接合一只手；方向和镜像按实测填写（`--left/right-xr-yaw-deg`、
`--left/right-xr-mirror`）：

```bash
robotarm data-acquisition piper-xr -- \
  --arms 2 --xr-bridge --left-channel can_follower1 --right-channel can_leader1 --firmware v189 \
  --tracking-frame head --left-xr-mirror none --right-xr-mirror none \
  --scale 2.0 \
  --live
```

降低映射比例可减少目标位移；降低速度上限也会使快速手部输入更容易被丢弃。调整时同时核对工作空间、限速和实际跟随误差，不以取消限制解决跟随问题。

程序首先等待 bridge 的左右手柄姿态有效、时间戳持续更新，然后才连接机械臂。
出现 `Waiting for fresh left and right controller poses` 时，Grip/Trigger+Y/B 尚未生效；
先确认 `xr-bridge` 正在监听相同端口、头显应用已打开 Send、两只手柄均已唤醒。
等待日志每 2 秒显示时间戳和左右姿态状态；出现 `Both XR controllers are ready`
并完成机械臂连接后，再长按同侧 Grip+Y/B。启动等待超时不会连接机械臂或发送急停。

按键映射：左/右 Grip 接合对应机械臂；Trigger 松开时对应夹爪闭合，按下时按比例
张开。长按同侧 Grip+Y/B 分别使能左/右臂。长按同侧 Grip+X/A 分别让左/右
臂回零；回零开始时另一臂也会立即保持，回零结束后两只手都必须松开再重新接合。
同时长按两侧 Grip+X/A 会执行顺序回零：右臂保持时先回左臂，再由左臂保持、回右臂，
不会让两臂同时运动。该顺序仍不等于完整的双臂连杆碰撞检测，执行前必须清空两条回零路径。
组合键长按期间
暂停对应夹爪更新，松开面键后 Trigger 恢复控制夹爪。
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

日志 `hand_delta` 为 XR 原始坐标位移，`requested_delta` 为工作空间限制后的机械臂基座目标位移（均为米）。双臂默认在每次接合时用头显 yaw 自动建立操作者坐标：向前对应机器人 `+X`、向右对应 `-Y`、向上对应 `+Z`；俯仰和侧倾不会混入平移映射。`--tracking-frame world` 可恢复固定世界坐标映射。默认 scale=2.0，手移动 10 cm 对应目标20 cm，单次接合相对目标最多20 cm；关节速度上限也会限制实际跟随速度。

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

单臂、双臂默认 `--engagement-mode grip`：长按同侧 Grip+Y/B 使能左/右臂（单臂用右 Grip+B）后，
**握住 Grip（≥0.9）才跟随，松开 Grip 立即脱离并保持当前位置**——松手即暂停，
重新握住即从新手部基准继续，不会追补松手期间的动作。Trigger 仍控制夹爪，
摇杆按下仍为急停。朝向跟随需要 `--orientation`。重新接合前保持手柄稳定；
接合时的手柄和机械臂位姿成为新的相对基准。一次接合内朝向基准保持固定，
手腕转回对应原来的末端朝向；受关节限位、奇异姿态和速度限制约束，
不能保证所有姿态都可达。

默认关节限速0.45 rad/s，每周期最大关节步长0.009 rad，朝向步长1.2°（每步限制还需与实际控制频率一起核对）。IK 每个候选最多做四次收敛迭代，最终指令仍通过关节限速和反馈误差检查。

相邻有效手柄数据位置变化超过 `--max-hand-jump-m 0.05`，或旋转变化超过 `--max-hand-jump-deg 30`，默认触发 `INPUT REBASED`：机械臂保持当前位置、丢弃跳变并在下一帧自动建立新手柄基准，不再要求重新按 Y/B。双臂任一侧跳变会同时重建两侧基准。需要旧的严格锁存行为时使用 `--hand-jump-action pause`，此时打印 `INPUT PAUSED` 并要求重新长按对应 Grip+Y/B。数据过期或逆解拒绝仍按各自保护逻辑处理。该检测针对手柄位姿突变，不是完整的碰撞检测。

```bash
robotarm data-acquisition piper-xr -- \
  --arms 2 --xr-bridge --left-channel can_follower1 --right-channel can_leader1 \
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

推荐流程为：**raw sessions → merged raw dataset → 直接审核/标注/裁剪 → 一次转换为 LeRobot**。
无需先生成未清洗的 LeRobot，也无需在标注后重建一次同样的视频。

| 阶段 | 输入 → 输出 | 是否重编码视频 | 主要产物 |
|---|---|---:|---|
| 采集 | 机械臂/相机 → raw session | 否 | `session.json`、`episodes/*/frames.npz`、原始MP4 |
| 合并 | 多个raw session → merged raw | 否 | 连续编号的raw episode、`merge_manifest.json` |
| 清洗 | merged raw → 审查标记 | 否 | `annotations/reviews.json` |
| 最终转换 | 已清洗merged raw → LeRobot | **是，仅一次** | 连续编号的训练数据、视频及来源清单 |

raw 是可继续合并和重新清洗的主数据；LeRobot 是完成清洗后供 ACT、VLA 等训练使用的派生数据。

### 5.1 原始session合并

多个原始session要先合成一个**原始**目录时，使用`merge-piper-raw`；不转换为LeRobot：

```bash
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/SESSION_A data/raw/piper_raw/SESSION_B \
  --output data/raw/piper_raw/MERGED_SESSION
```

替换占位符后先运行预检查；确认后加`--execute`复制并重新编号。原数据不动，已有输出拒绝覆盖。

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

#### 在 merged raw 上清洗和标注，再转换一次

```bash
# 合并与校验：先不加 --execute 检查，确认后再复制
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/SESSION_A data/raw/piper_raw/SESSION_B \
  --output data/raw/piper_raw/MERGED_SESSION --execute

# 直接打开 merged raw；也支持单个未合并的 raw session
robotarm modeling dataset-browser -- \
  --dataset-root data/raw/piper_raw/MERGED_SESSION
```

页面会显示“Raw 审核”，直接播放原 MP4、读取 NPZ 中的 state/action；支持原有
keep/review/reject、备注、姿态标签及逐 episode 首尾裁剪。操作仅更新
`MERGED_SESSION/annotations/reviews.json`，不修改原视频、NPZ 或 episode 元数据。
裁剪范围按 `[start_frame, end_frame)` 保存；保存标注本身不编码视频。

清洗后点击“将清洗结果转换为 LeRobot”，填写新名称，即在 `data/lerobot/` 下生成最终训练集。
可用 `--export-root` 指定另一输出父目录。只转换 keep/未审查条目及其保留帧段，跳过
review/reject；也可按标签导出。注意未审查条目默认可导出，并不代表人工验收通过。
CLI 同样自动读取 raw 目录内的标注，无需先导出一个裁剪后的 raw 副本：

```bash
robotarm data-acquisition convert-piper-raw -- \
  data/raw/piper_raw/MERGED_SESSION \
  --output data/lerobot/MERGED_SESSION_clean_v1
```

用 `--reviews PATH` 可指定单个 raw session 的自定义标注文件。导出只在最后一次转换时
编码最终视频并重建连续 episode/frame 编号；`meta/piper_conversion.json` 保留来源、原帧范围、
标签和备注，训练质量筛选及重新打开导出集会继承已重编号标注，不重复套用旧裁剪范围。

后续新增采集时，可将已标注的 merged raw 与新 raw session 合并为**另一个新目录**。
默认位置的 raw 标注会自动按原始 episode 身份迁移；已有 keep/reject、备注、标签和裁剪不丢失，
新条目保持未审查。合并仍拒绝重复来源，标注绑定原始来源与 NPZ 内容；检测到 episode
顺序或内容改变时拒绝误用标注。自定义 `--reviews` 文件应先保存回上述默认位置再参与自动迁移。

如果需要保留每一代不可变快照，继续使用新的 `--output`：

```bash
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/MERGED_SESSION \
  data/raw/piper_raw/NEW_SESSION \
  --output data/raw/piper_raw/MERGED_SESSION_V2 \
  --execute

robotarm modeling dataset-browser -- \
  --dataset-root data/raw/piper_raw/MERGED_SESSION_V2
```

如果长期只想保留一份滚动MASTER，使用事务式`--append-to`。第一次不加`--execute`只预检：

```bash
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/NEW_SESSION \
  --append-to data/raw/piper_raw/MASTER

# 确认新增episode数量、最终总数和复制空间后执行
robotarm data-acquisition merge-piper-raw -- \
  data/raw/piper_raw/NEW_SESSION \
  --append-to data/raw/piper_raw/MASTER \
  --execute
```

该模式不重新复制MASTER已有视频/NPZ，只把新增episode复制到暂存区、逐文件校验，再发布连续编号、
来源清单和审查映射。原有keep/reject、备注、标签和裁剪保持不变，新episode继承其源session已有
标注或保持未审查。执行期间`APPEND_INCOMPLETE.json`会阻止浏览、转换和保存标注；成功后自动删除。
输入`NEW_SESSION`不会自动删除，确认MASTER能正常打开且数量正确后，可自行移走或删除输入目录。

异常会在当前进程内自动回滚。如果断电或强制结束留下恢复标记，先预览再执行回滚；不要手工删除
恢复标记或`.raw_append_*`暂存目录：

```bash
robotarm data-acquisition merge-piper-raw -- \
  --append-to data/raw/piper_raw/MASTER \
  --recover-append

robotarm data-acquisition merge-piper-raw -- \
  --append-to data/raw/piper_raw/MASTER \
  --recover-append --execute
```

`--append-to`要求MASTER中的episode已经连续编号且均为complete；同一MASTER不要并行执行两个追加任务。
如果希望每次都保留可直接回退的完整数据版本，使用前面的`--output MERGED_SESSION_V2`模式。

这条流程不提供旧 LeRobot 标注自动反向迁移到 raw 的功能；已有训练集及其审查文件仍可沿用旧入口。

#### 原始合并检查与来源追溯

| 参数/行为 | 说明 |
|---|---|
| `sessions...` | 任意数量原始session目录，不能传LeRobot目录；不建议用日期通配符误选合并产物 |
| `--output` | 必填的新raw目录；不能与任何输入重合、互为父子目录，不支持覆盖/追加 |
| `--append-to` | 事务式扩展一个连续、完整的raw MASTER；旧payload不复制，只增加新episode |
| `--dry-run` / 不加`--execute` | 检查数组、视频头中的尺寸/FPS/帧数、兼容性和编号；不创建文件 |
| `--execute` | 独立复制，SHA256校验，再写合并元数据；视频不重编码，不进行模型推理 |
| `--recover-append` | 与`--append-to`合用，预览或回滚断电/强制结束留下的未完成事务 |
| `--include-incomplete` | 默认只保留complete且有帧的episode；显式允许未完成条目仍须通过数据完整性检查 |
| `--allow-mixed-tasks` | 默认拒绝不同任务；显式允许后逐episode保留标签，不自动改成同一任务 |

合并前退出采集程序；发现`.in_progress`会拒绝。中途失败保留部分输出和
`MERGE_INCOMPLETE.txt`，原数据不受影响；不要将未完成输出用于训练/转换。
预检查不完整解码每个视频帧；复制校验确保内容未被合并过程改变，不代表数据质量或任务成功。

来源保存在每个`episode.json`的`merge_source`及根目录`merge_manifest.json`，
清单包含输入session完整元数据、原episode路径/编号和文件SHA256。
重复传同一session、同时传原数据及含相同轨迹的本工具合并产物会报错，避免重复采样。
合并目录不能代替备份。默认位置的raw审查标记会随再次合并重新映射，并在最终转换后按新的
LeRobot episode编号继承；若源NPZ内容或episode身份发生变化，工具会拒绝套用旧标记。

要求FPS、关节单位/名称/动作语义、相机角色/序列号/视频格式/已记录内参等兼容。
**这些检查不能确认相机外参一致**；拆装后的视角变化仍需人工检查。
不同安装批次的setup信息完整保留在来源元数据中，不应把首个session的setup当作所有批次的证明。

### 5.2 清洗完成后最终转换为LeRobot（训练前必做）

确认merged raw中的keep/review/reject、标签和裁剪范围后，将它一次转换为双相机LeRobot数据集。
转换器自动读取`annotations/reviews.json`：保留keep和未审查episode，排除review/reject，并只编码
各episode保留的帧段。建议重要训练集先把所有episode明确标为keep或reject，避免未审查数据混入。

例如，将2026-10-19已经合并并清洗好的原始目录转换为最终训练集：

```bash
robotarm data-acquisition convert-piper-raw -- \
  data/raw/piper_raw/20261019_merged_122eps \
  --output data/lerobot/piper_cup_20261019_122eps_v1 \
  --repo-id local/piper_cup_20261019_122eps_v1 \
  --image-writer-threads 4 --encoder-threads 4
```

这条命令会实际写入新数据集，**不用加`--execute`，也没有`--dry-run`参数**。不会改原始session。
输出必须是新目录；转换成功后可见`meta/`、`data/`、`videos/`。

通用模板（替换`MERGED_SESSION`、`DATASET`）：

```bash
robotarm data-acquisition convert-piper-raw -- \
  data/raw/piper_raw/MERGED_SESSION \
  --output data/lerobot/DATASET --repo-id local/DATASET \
  --image-writer-threads 4 --encoder-threads 4
```

转换成功后，`meta/piper_conversion.json`记录raw episode身份、原始帧范围、审查快照和编号映射；
重新用Dataset Browser打开该LeRobot目录时，备注和标签会继承，已经应用的raw裁剪不会重复执行。

兼容旧用法：也可以跳过raw合并，直接把多个raw session转换成一个LeRobot数据集，但不适合需要
跨批次统一清洗、反复加入新采集数据的工作流。新数据优先按5.1节先合并raw、审核，再统一转换：

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

已有项目如果只保留了LeRobot工作集，需要向其中增加新raw数据，可继续使用事务式
`extend-dataset`。它先复制到临时目录，
完成追加和回读校验后再发布；能继承Dataset Browser标记，也能处理已经裁剪导出、缺少
`piper_raw_session.json`的数据集。`new`模式保留旧目录并创建新数据集：

```bash
robotarm modeling extend-dataset -- \
  --mode new \
  --base-dataset data/lerobot/BASE_DATASET \
  --new-raw data/raw/piper_raw/NEW_SESSION \
  --output data/lerobot/NEW_DATASET

# 上述命令默认只预检；确认后原样重跑并加：
#   --execute
```

`append`模式不传`--output`，成功后原子替换原目录，并保留一个
`*_backup_before_append_时间`完整备份。新episode默认标记为`review`和
`cup_orientation=sideways`，审核通过前不会被`--quality-filter`选入训练；可用
`--new-status keep`或`--new-label upright|uncertain|none`显式修改。
底层`convert-piper-raw --append`会原地修改目标，不提供事务和审查标记迁移，日常不要直接调用。
`robotarm modeling convert-dataset`是同一转换器的兼容入口，通常统一用`data-acquisition convert-piper-raw`。

2026-10-27新增的59条侧倒杯数据，统一放回未裁剪的122条数据集后再审核、裁剪。推荐保留旧目录，
生成181条未裁剪工作集：

```bash
# 第一次不加 --execute，只做兼容性、数量和标记迁移预检
./scripts/extend_lerobot_dataset.py \
  --mode new \
  --base-dataset data/lerobot/piper_pick_cup_20261019_direct_122eps \
  --new-raw data/raw/piper_raw/20261027_150825_390106 \
  --output data/lerobot/piper_pick_cup_20261019_direct_181eps_untrimmed_20261027

# 确认预检显示 122 + 59 = 181 后，原样重跑并在末尾增加：
#   --execute
```

脚本继承原122条的keep/review/reject、备注、姿态标签和未应用裁剪范围；新59条编号为122～180，
默认写入`review + cup_orientation=sideways`。完成后打开181条工作集，逐条把新数据改为keep/reject并
设置裁剪范围，再从页面导出最终训练集。不要把122条原集、181条工作集及其裁剪副本同时送入一次训练。

### 5.3 浏览、质量审查、读取数量

支持直接打开 raw/merged raw（推荐，见上文）或已有 LeRobot 数据集，自动识别格式。

```bash
robotarm modeling dataset-browser -- \
  --dataset-root data/lerobot/piper_cup_20261019_122eps_v1

# 只读数据集元信息，查看total_episodes、total_frames、fps和features
python -m json.tool data/lerobot/piper_cup_20261019_122eps_v1/meta/info.json
```

浏览器默认 <http://127.0.0.1:8766>。数据动作采用六关节 rad + 夹爪 m 的绝对命令目标。
页面顶部并排显示 **中文 / English**，有蓝色底色的一项是当前语言；点击另一项可即时切换，
并在当前浏览器记住选择。
切换不会重载episode或丢弃未保存的标记、备注与裁剪范围；任务文字、数据集名和用户备注保持原文。
质量标记与数据分开保存，不用 `lerobot-replay` 代替只读预览。
↑/↓切episode，←/→逐帧，空格播放；在页面标记keep/review/reject。
LeRobot 审查文件默认`artifacts/reports/dataset_quality/<数据集名>.json`，raw 审查文件默认在源目录
`annotations/reviews.json`；`--quality-filter`训练时排除review/reject，
未审查条目默认仍会被保留，不等于已经人工验收。

每条episode还可保存结构化的“任务结果（HIL）”标签：`success`（成功）或
`failure`（失败）。标签写入同一份审查JSON的`episodes.<编号>.labels`，不会修改源LeRobot数据；
左侧可按标签筛选，导出区也可只导出某个标签或未标记条目。旧审查文件中的
`cup_orientation`会为兼容性保留，但平台不再显示或提供`upright/sideways`姿态标记功能。
尚未人工保存过的episode默认预填`keep + success`，并显示`DEFAULT / 未人工复核`；点击保存后
写入`review_origin=manual`并切换为“已人工复核”。默认值可以参与预览和导出，但不能当作人工验收结论。

#### 每条episode独立裁剪并导出

启动命令仍是上面的`dataset-browser`。修改代码后需退出旧服务、重新启动并刷新页面。

1. 在左侧选择episode，在“当前 Episode 保留片段”填写起点和终点，例如`5`到`14`秒。
   也可逐帧定位后点“当前帧设起点”“当前帧后设终点”；后者包含当前这一帧。
2. 勾选“只播放所选片段”预览，绿色曲线区域表示保留范围；自动质量指标仍针对整条原轨迹。
3. 可选填“任务结果（HIL）”，然后点击“保存标记、标签与裁剪”。每条轨迹独立保存，可给下一条设置不同标签和范围；切换时未保存会提示。
   “恢复整条”并保存可撤销该条裁剪；质量状态的“清除”不会自动撤销裁剪。
4. 导出区选择“全部可导出条目”或指定标签，填写一个尚不存在的新数据集名称，点“导出已保存的裁剪数据集”。
   导出在源数据集旁边创建新目录，包含保留及未审查条目，排除复查和剔除条目。
   未设置裁剪的条目保留整条；左侧状态/标签筛选只影响显示，导出范围只由导出区标签选择决定。
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
新目录不能直接套用旧数据集的episode编号审查文件。没有新审核文件时，审核页面及`--quality-filter`
会自动读取新目录内的`meta/trim_provenance.json`，按新旧编号映射继承状态和备注，并移除已应用的裁剪范围。
这也兼容此前已导出的数据集，无需重新导出或逐条重新标记；`--quality-shuffle-seed 1000`仍正常生效。
重新审核并保存后，以新的审核文件为准；显式指定但不存在的`--quality-reviews`不会静默回退。
导出未完成、清单不匹配或普通数据集缺审核文件仍会报错。不要将原数据和裁剪副本混入同一次训练。
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

若同一个实际任务在历史合并过程中产生了多个语言标签，可发布一个统一 task 的新数据集。
该命令默认只预览；加`--execute`后保留源目录，复制可变元数据并硬链接视频，不重新编码视频：

```bash
robotarm modeling relabel-dataset-task -- \
  --dataset-root data/lerobot/piper_pick_cup_direct_223eps_trimmed_20261028 \
  --output-root data/lerobot/piper_pick_cup_direct_223eps_trimmed_single_task_20261029 \
  --task "Pick up the cup in different poses"

# 核对预览中的episode、帧数和旧标签分布后，原样重跑并追加：
#   --execute
```

新目录会同步更新`tasks.parquet`、episode任务、帧级`task_index`、统计量和`total_tasks`，
并写入`meta/task_relabel.json`记录旧标签分布。此操作只统一语言条件，不改变动作轨迹；
如果旧数据实际包含不同的后续行为，应先按目标行为裁剪，不能仅靠改文字把它们视为同一任务。

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

### ResNet18 layer3高分辨率特征实验

仓库为固定版本`lerobot==0.6.1`提供了一个可迁移补丁，使ACT可以显式选择ResNet的
`layer3`（stride 16）或默认`layer4`（stride 32）特征。旧配置不含新字段时仍默认
`layer4`，因此旧checkpoint的读取和结构不变。`layer3`是不同模型结构，必须使用新输出目录
从头训练，不能从现有`layer4` checkpoint续训。

当前环境首次应用和之后校验：

```bash
bash scripts/apply_lerobot_act_feature_layer_patch.sh \
  --python "$(command -v python)" --apply
bash scripts/apply_lerobot_act_feature_layer_patch.sh \
  --python "$(command -v python)" --check
```

以当前FLVV数据做对照实验时，只改变特征层，其余训练设置保持一致：

```bash
TASK_DATASET=data/lerobot/merged_20261113_pick_place_flvv
TASK_RUN=artifacts/checkpoints/20261115_act_flvv_110eps_chunk30_layer3_v1
robotarm modeling train -- "$TASK_DATASET" "$TASK_RUN" act \
  --quality-filter --quality-shuffle-seed 1000 \
  --dataset.eval_split=0.2 --eval_steps=1000 --max_eval_samples=0 \
  --batch_size=16 --steps=100000 --save_freq=2000 --seed=1000 \
  --policy.chunk_size=30 --policy.n_action_steps=20 \
  --policy.vision_feature_layer=layer3
```

不要同时设置`--policy.replace_final_stride_with_dilation=true`。对480×640输入，ResNet18的
`layer4`每路相机产生15×20=300个视觉token，`layer3`产生30×40=1200个；双相机总视觉token
从600增至2400，显存占用和Transformer计算量会显著上升。首次先跑少量step确认batch size和
推理延迟；若OOM，优先下调`--batch_size`，不要修改图像或训练划分后再与旧基线直接比较。

新建本机完整环境时，`scripts/create_unified_env.sh`会自动应用补丁。云端只训练ACT时，上传或
克隆完整仓库及数据后使用轻量训练环境脚本（默认CUDA 12.8 PyTorch wheel）：

```bash
cd /path/to/RobotArm
CONDA_EXE="$(command -v conda)" \
  ROBOTARM_ACT_ENV_NAME=robotarm-act \
  bash scripts/create_act_training_env.sh
conda activate robotarm-act
bash scripts/apply_lerobot_act_feature_layer_patch.sh --check
robotarm modeling check-environment -- \
  --dataset-root data/lerobot/merged_20261113_pick_place_flvv
```

云服务器需由`nvidia-smi`确认驱动支持默认CUDA 12.8 wheel。若必须使用另一个官方CUDA wheel源，
应同时设置匹配的`PYTORCH_INDEX_URL`和`TORCHAO_VERSION`，不能只替换其中一项；Torch/torchvision
版本仍由脚本锁定。数据和完整checkpoint迁移时保持相对路径；训练配置会记录云端绝对路径，但
权重加载仍应传实际存在的本地目录。

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

打开 <http://127.0.0.1:8420>。页面右上角并排显示 `中文`／`English`，有底色的一项是当前语言；
选择保存在当前浏览器；关闭监控不会停止训练，目录未创建时监控会等待。
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

### SmolVLA直接微调

在已有GPU环境中，从项目根目录执行。只补训练依赖，不需要重新创建GPU环境：

```bash
python -m pip install -e '.[smolvla-training]'
hf download lerobot/smolvla_base --local-dir artifacts/pretrained/smolvla_base
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

第二个模型是视觉语言主干和tokenizer，默认进入Hugging Face缓存；首次使用需要联网。
`--base-checkpoint`指向含`config.json`、`model.safetensors`和前后处理器文件的整个目录。
SmolVLA启动器会把基础模型输入配置适配为Piper的7维state/action和`global + wrist`双相机，
保留相机原名及顺序，状态/动作按本数据集统计量归一化，无需改数据集列名。

先检查计划，不加`--execute`不会加载模型或训练；计划会打印实际语言指令和选中的episode数量：

```bash
robotarm modeling train-smolvla -- \
  --dataset-root data/lerobot/pick_up_flvv_20261115_curated \
  --base-checkpoint artifacts/pretrained/smolvla_base \
  --output-dir artifacts/checkpoints/smolvla_flvv_expert_v1 \
  --training-mode expert --batch-size 4 --steps 30000 \
  --lr 0.0001 --warmup-steps 1000 --decay-lr 0.0000025 \
  --chunk-size 50 --n-action-steps 20 \
  --eval-split 0.1 --eval-steps 2000 --max-eval-samples 256 \
  --save-freq 2000 --log-freq 100 --num-workers 4 --seed 1000
```

将上述命令的输出目录改为`artifacts/checkpoints/smolvla_flvv_smoke_v1`，并追加
`--smoke --execute`执行20次更新、保存及小样本验证。通过后，用原来的新输出目录并追加
`--execute`正式训练。smoke会将warmup缩到最多2步、每步记录loss，避免20步全处于warmup；
smoke和正式训练应使用不同输出目录，启动器拒绝覆盖已有目录及launch manifest。

| 参数 | 起始值与含义 | 调整方法 |
|---|---|---|
| `--dataset-root` | 最终LeRobot数据目录 | 需7维绝对关节state/action、双相机及非空语言任务；无需再次转换已清洗数据 |
| `--training-mode` | `expert`，冻结VLM并训练动作专家和状态投影等动作侧模块 | `lora`仅训练低秩适配器，是显存不足时的备选 |
| `--batch-size` | `4`，每次更新的样本数 | OOM降到2或1；显存充足可试8或16，更改后重新比较验证表现 |
| `--steps` | `30000`，总更新次数 | 用验证曲线和离线/任务表现选checkpoint；与batch共同决定看过多少样本 |
| `--lr` | `1e-4`，峰值学习率 | loss不稳定可试`5e-5`；与batch变化一起记录 |
| `--warmup-steps` | `1000`，学习率从低值升到峰值的步数 | 必须小于总步数；短实验相应缩小 |
| `--decay-lr` | `2.5e-6`，余弦衰减的学习率下限 | 先保持；衰减周期随实际`steps`设置 |
| `--chunk-size` | `50`，监督并预测50步动作 | 20Hz下覆盖2.5秒，先保持 |
| `--n-action-steps` | `20`，推理消费窗口 | 20Hz下约1秒，主要影响部署；不改变训练监督长度，不能超过chunk |
| `--eval-split` | `0.1`，按episode划分留出集 | 固定seed；不随机拆相邻帧，防止训练/验证相互泄漏 |
| `--eval-steps` / `--save-freq` | `2000` / `2000`，验证和保存间隔 | 建议对齐，便于比较checkpoint |
| `--max-eval-samples` | `256`，每次验证样本上限 | `0`表示全量验证；更稳定但耗时更多 |
| `--log-freq` | `100`，每多少步记录loss | 短实验可设10 |
| `--num-workers` | `4`，数据加载进程数 | 视频解码/多进程问题先设0排查 |
| `--seed` | `1000`，episode打乱和训练随机种子 | 比较参数时固定 |
| `--rank` / `--alpha` | `16` / `32`，LoRA秩和缩放参数 | 仅`lora`模式生效；先保持 |
| `--quality-reviews` | 可选审核JSON | 自动沿用默认位置及裁剪导出审核；排除review/reject，拒绝未应用的裁剪 |

这些是第一轮基线，不保证最优。验证loss衡量留出数据上的拟合，不等同于机械臂任务成功率。
267条数据共119431帧；batch 4、30000步约采样12万个训练样本，约为留出后数据量的1.1倍，
不足以仅凭步数断言已经收敛。先看验证是否继续改善，再决定延长训练。

训练默认使用AMP、离线WandB日志，不上传模型。另一个终端运行已有loss监控：

```bash
python scripts/train_monitor.py \
  --run-dir artifacts/checkpoints/smolvla_flvv_expert_v1 --port 8420
```

checkpoint在`RUN/checkpoints/步数/pretrained_model/`，完整续训还需要同级`training_state/`。
中断后从已保存点恢复，使用LeRobot原生命令（不要重新传基础模型路径）：

```bash
lerobot-train \
  --config_path=artifacts/checkpoints/smolvla_flvv_expert_v1/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true --steps=30000
```

`steps`仍为目标总步数。续训会恢复原优化器和调度器；如果超过原计划的30000步，原余弦调度
已到下限，不会自动重新warmup。需要重新设计调度时应另做实验并记录配置。

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

先按[打包命令](#12-辅助脚本)生成新的独立快照。默认包包含代码、LF180完整LeRobot数据、
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

## 9. 在线运行：ACT直接执行与可选shadow

### 9.1 ACT shadow与审批（可选）

以下以ACT为例，显式使用单步消费和时间集成，覆盖当前YAML里的30步。
确认与训练一致的相机装定、物体布局和起始姿态；先停其他遥操/相机程序。

```bash
TASK_CHECKPOINT=artifacts/checkpoints/act_cup_ring_dual_20261013_chunk50_180eps_lf_v1/checkpoints/090000/pretrained_model
TASK_SHADOW=artifacts/logs/shadow/act_090000_manual.jsonl
TASK_APPROVAL=artifacts/reports/runtime_approvals/act_090000_manual.json
robotarm modeling online-policy -- \
  --checkpoint "$TASK_CHECKPOINT" \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01 \
  --warn-inference-ms 60 --max-inference-ms 500 \
  --warn-act-raw-joint-gap-deg 12 --max-act-raw-joint-gap-deg 15 \
  --max-act-raw-gap-consecutive-steps 3 \
  --max-act-subgoal-no-progress-steps 10 \
  --duration 60 --report "$TASK_SHADOW"

# shadow结束并检查后，生成绑定权重和参数的审批
robotarm modeling approve -- --report "$TASK_SHADOW" --output "$TASK_APPROVAL"
```

shadow 读取相机/反馈，但不发送模型运动。检查完整日志中的 `watchdog_failures`、
`planner_underflow`、原始动作及限幅情况，不以“程序运行结束”代替验证通过。
默认日志同名重跑可能覆盖，需留存试次时换`--report`路径。

### 9.2 ACT真机执行

当前 `online-policy.approval-required: false`，可以跳过 shadow 和审批，直接启动这个
ACT checkpoint 的执行入口。程序会先使能从臂、将六关节回到零位（运动途中保持夹爪开度，
到位后闭合夹爪并确认反馈）、
预热模型并准备 JS 控制；打印 `ACT ready at home; press X` 后仍保持暂停，按 X 才发送
模型动作。抓取中或抓取后单击 R，会暂停当前试次、丢弃剩余预测动作，并在当前位置保持
机械臂，先张开夹爪至最多 80 mm（已抓住的物体会在当前位置释放），确认夹爪反馈后再
回零；到零位后闭合夹爪至示教起始开度，等待重新按 X。启动回零、长按 Y 回零及
Ctrl+C 退出前回零也会在到位后闭合夹爪。新试次使用新的相机画面和关节
状态。复位期间按 Ctrl+C 会发送电子急停；夹爪张开、闭合或回零超时也会电子急停。

```bash
robotarm modeling online-policy -- \
  --checkpoint artifacts/checkpoints/act_cup_20261028_chunk30/checkpoints/100000/pretrained_model \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01 \
  --warn-inference-ms 60 --max-inference-ms 500 \
  --warn-act-raw-joint-gap-deg 12 --max-act-raw-joint-gap-deg 15 \
  --max-act-raw-gap-consecutive-steps 3 \
  --max-act-subgoal-no-progress-steps 10 \
  --duration 60 --report artifacts/logs/shadow/act_100000_direct_execute.jsonl \
  --execute
```

跳过 shadow 只跳过执行前的审批流程；运行时的原始动作差值、限位、相机时效等
看门狗仍然生效。如果回零后首个预测动作越过关节物理边界或包含非有限值，按 X 后会立即
暂停；仅原始关节/夹爪差较大时按下述遥测告警与ACT动作消费背压处理。
推理耗时使用两级门槛：`--warn-inference-ms 60`是软告警，超过后终端打印
`WARNING: ... continuing execution`，并写入日志的`performance_warnings`，但仍发送本次经过
安全限幅的动作；`--max-inference-ms 500`才是硬停门槛，超过后写入`watchdog_failures`并暂停。
20Hz下一个周期为50ms，例如229ms的偶发尖峰约跨过4至5个周期；等待推理期间控制器保持
上一目标，推理完成后继续执行。关节越界、原始动作异常、相机过期等门控不受软告警影响，
仍会立即暂停。若需修改硬停值，应根据完整真机遥测评估，不应将其取消。
原始关节差只作为预测遥测：超过`--warn-act-raw-joint-gap-deg 12`会告警，超过
`--max-act-raw-joint-gap-deg 15`及其连续拍数会记录严重差距，但不会仅凭“移动中的原始预测
离反馈越来越远”暂停。程序仍按200°/s²加速度、每拍2.5°步长和相对反馈6°超前量限幅；一旦
触发最终超前/步长限幅，就按当前速度的制动距离生成一次性的有界子目标并冻结；实际命令仍
始终限制在反馈前方6°以内。硬限幅期间暂停ACT动作消费（包括时间集成序列），让真机追赶；
最终硬限幅一解除就恢复下一次推理，不要求Piper精确停在固定点，因此不会被约2°的JS跟随
残差卡住。只有仍处于硬限幅状态且连续`--max-act-subgoal-no-progress-steps 10`拍没有达到
最小反馈进展才暂停；20Hz下约为0.5秒。目标越过关节/夹爪物理边界或包含NaN/Inf仍立即暂停。
当前抖动优化的推荐组合是`--n-action-steps 1 --temporal-ensemble-coeff 0.01`：每个控制周期
只消费一步，同时对重叠chunk的ACT预测做时间集成，减小chunk内部和边界处的动作跳变。
启用`--temporal-ensemble-coeff`时程序要求`n_action_steps=1`，否则拒绝启动。它只改变在线推理
和动作消费方式，不改变checkpoint的`chunk_size`、训练监督长度或训练loss。每次用不同
`--report`路径保存日志，不要放宽安全限值来掩盖问题。

若将 `online-policy.approval-required` 改为 `true`，可复用上一小节同终端定义的
变量走审批流程；若新开终端，先重新定义这三个变量，并确保执行参数与审批一致：

```bash
robotarm modeling online-policy -- \
  --checkpoint "$TASK_CHECKPOINT" \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01 \
  --warn-inference-ms 60 --max-inference-ms 500 \
  --warn-act-raw-joint-gap-deg 12 --max-act-raw-joint-gap-deg 15 \
  --max-act-raw-gap-consecutive-steps 3 \
  --max-act-subgoal-no-progress-steps 10 \
  --duration 60 --report artifacts/logs/shadow/act_090000_manual_execute.jsonl \
  --approval "$TASK_APPROVAL" --execute
```

启动先自动回零并保持暂停，按 X 开始。需要强制审批时在 YAML 改为 true；
权重或运行参数改变后重新验证。

完全使用YAML默认参数时，可一条命令做shadow＋审批（不发送模型运动）：

```bash
robotarm modeling validate-runtime -- \
  --checkpoint "$TASK_CHECKPOINT" --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --seconds 60 --min-samples 1000
```

该入口不接受`--n-action-steps`或`--temporal-ensemble-coeff`等在线覆盖；它此时用YAML的30步，
**不能冒充上面单步＋时间集成命令的验证**。
需要覆盖参数时用显式shadow→approve→execute。

| 输入 | 执行模式行为 |
|---|---|
| V | 开始／结束一次评估录制；不改变模型控制状态，同一程序可录制多段 |
| X | 开始/暂停一轮模型控制 |
| I | 仅在`--hg-dagger`下：从臂立即保持、主臂进入零力模式；主臂有效移动后开始专家控制和标记；再次按I交还ACT并恢复主臂同步，未移动则取消接管；完整episode由X→X保存 |
| D | 仅在`--hg-dagger`下：放弃正在进行的trial，删除该trial的临时视频与NPZ并暂停；R/Y回零后才能开始下一条 |
| 1 / 0 | 仅在`--hg-label-outcome`下：为刚结束的ACT+人工接管trial标记成功／失败；这是历史replay标签，不会启动RL |
| R | ACT 当前试次暂停，当前位置张开夹爪，回零后闭合；再按 X 开始新试次 |
| 长按Y 0.3秒 | 暂停并回零，运动途中保持夹爪开度，到位后闭合 |
| Ctrl+C | 通常回零并闭合夹爪后退出，两臂保持使能；回零过程中按下则电子急停 |
| Alt+S | 电子急停 |
| 视频窗口F/F11、双击、Esc | 切换全屏／返回窗口；窗口默认置顶、可调整大小 |

在线视频窗口获得焦点时也支持V、X、R、长按Y，`S`为电子急停。shadow不执行按键回零/急停，
但仍可用V录制模型输入和预测结果。
`--duration`在shadow中为总时长，在execute中为每次按X启动的试次上限，到时暂停。

按V后才会创建录制目录，默认位于
`artifacts/policy_recordings/<启动时间>/recording_<编号>/`；再次按V立即停止采样，MP4在后台封装，
不会等待编码完成后才恢复控制。一次`online-policy`进程中可以反复按V，生成多个编号递增的录制目录。
录制跨越暂停、X启动/停止以及R/Y回零；R/Y的阻塞运动阶段仍继续记录相机和关节反馈。输出同时包含：

| 文件 | 内容 |
|---|---|
| `combined.mp4` | global与wrist横向拼接在同一画面，并叠加录制时间、trial和执行/暂停状态 |
| `global.mp4`、`wrist.mp4` | 两路无叠字的独立原始相机视频 |
| `frames.npz` | 与视频采样对齐的7维state、安全action、原始模型action、相机时间戳/帧号、trial和按键事件 |
| `joints.csv` | 便于直接查看的关节rad/deg、夹爪m/mm、安全action和原始action |
| `recording.json` | 帧数、时长、丢帧、文件映射和完整/异常状态 |
| 上级`session.json` | checkpoint、task、策略类型、相机序列号和录制参数 |

默认录制FPS为控制频率与相机FPS的较小值。可用
`--recording-root PATH --recording-fps 20 --recording-encoder-queue-size 120`覆盖。
需要系统已安装`ffmpeg`；仅在首次按V时才检查并启动编码器。编码队列满、磁盘不足或写盘失败只会终止
当前录制并标为`incomplete`，不会自行继续发送或停止机械臂命令；机械臂安全仍由原有watchdog控制。
若退出程序前没有再次按V，当前片段会保存为`incomplete_recording_<编号>`，避免把非显式结束的数据冒充完整录像。

#### HG-DAgger人工接管与纠正数据

HG-DAgger使用人工专家直接控制门控：按`X`后，程序从模型运行开始保存完整trial，并把同一组经过安全限幅的ACT目标
同时发给主臂和从臂；正常ACT执行及回零期间不再用从臂反馈驱动主臂的相对跟随层。专家发现将要失败时按`I`
进入`ARMED`待运动状态，此时模型动作队列被清空、从臂保持接管位置、主臂切换为零力模式；检测到主臂
相对接管点的有效移动后才切到`EXPERT`控制并逐帧标记专家区间。完成恢复后再次按`I`交还模型；
同时主臂退出零力并从当前实际位姿无跳变地恢复跟随。若主臂尚未移动，再按`I`只取消接管，完整trial仍继续。接管前后的模型动作保留在raw中供
HIL-SERL/RECAP使用，但HG监督转换只导出`expert_mask=1`的连续区间，不会把失败模型动作当成
训练action。当前实现使用
`can_leader1`主臂，接管时记录“从臂姿态－映射后主臂姿态”的离合偏置，因此不要求两臂绝对
对齐，也不会因控制源切换而跳到主臂绝对姿态。

下面以当前267条BC数据训练的16万步ACT模型为例。该命令只运行ACT与HG-DAgger，
不启动SAC learner，也不执行RL策略：

```bash
robotarm modeling online-policy -- \
  --checkpoint artifacts/checkpoints/20261115_act_flvv_267eps_layer4_chunk50_bs16_160k_v1/checkpoints/160000/pretrained_model \
  --interface socketcan --channel can_follower1 --firmware v189 \
  --camera-serial 347522071736 --wrist-camera-serial 260322272554 \
  --device cuda --rate 20 --camera-fps 20 \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01 \
  --warn-inference-ms 60 --max-inference-ms 500 \
  --warn-act-raw-joint-gap-deg 12 --max-act-raw-joint-gap-deg 15 \
  --max-act-raw-gap-consecutive-steps 3 \
  --max-act-subgoal-no-progress-steps 10 \
  --hg-dagger --hg-leader-interface socketcan --hg-leader-channel can_leader1 \
  --hg-task "Pick up the KS11 FLVV component and place it in the target black area with hole" \
  --hg-setup-id flvv_act160k_hg_dagger_v1 \
  --duration 150 --report artifacts/logs/shadow/act_flvv_160000_hg_dagger.jsonl \
  --execute
```

也可以用同一模型对应的包装脚本；下列命令与上面一样仍是独立的DAgger采集：

```bash
scripts/run_act_policy.sh --execute -- \
  --hg-dagger --hg-leader-interface socketcan --hg-leader-channel can_leader1 \
  --hg-task "Pick up the KS11 FLVV component and place it in the target black area with hole" \
  --hg-setup-id flvv_act160k_hg_dagger_v1
```

如果要测试 HG 微调后效果较好的 **010000** 检查点，并继续收集 HG 纠正数据，显式指定该模型：

```bash
scripts/run_act_policy.sh --execute \
  --checkpoint artifacts/checkpoints/flvv_hg_then_hil_round1/act/checkpoints/010000/pretrained_model -- \
  --hg-dagger --hg-leader-interface socketcan --hg-leader-channel can_leader1 \
  --hg-task "Pick up the KS11 FLVV component and place it in the target black area with hole" \
  --hg-setup-id flvv_act010000_hg_dagger_v1
```

上面两种 HG 命令都不更新 HIL 权重。HG 退出后直接重跑对应命令即可，启动时重新双臂回零、
等待 `X`，并创建新的 raw session；旧数据保留。不要用新 session 的 episode 编号判断是否覆盖了旧数据。
退出前按 `X` 保存当前完整 trial；若直接 `Ctrl+C`，正在运行的 trial 记为 incomplete，
下次从新 trial 开始，不从中间动作续接。正常一次 `Ctrl+C` 会双臂回零并保持使能后退出；
回零中再次按 `Ctrl+C` 则急停。

启动顺序：程序先连接主臂，再并发启动主臂固件回零和从臂低速MOVE_J回零；主臂固件回零速度由控制器决定。只有两臂关节都到位、从臂夹爪闭合后才进行模型预热。
从臂关节回零到位后立即闭合自身夹爪，不再等待主臂完成回零；R/Y及退出回零也采用这个时序。
主臂处于Leader角色而无普通夹爪反馈时，程序使用从臂实测夹爪宽度作为主臂初始保持目标，不再要求人工开合主臂夹爪。主臂当前六关节位置仍必须读到，以免未知位姿下切换角色或盲发回零目标。
看到`dual-arm policy control ready; press X to start`后才按`X`；运行模型后同一组ACT安全目标会同时发给两臂。
按`R`、长按`Y`或正常按一次`Ctrl+C`退出时，两臂也会同时回零；回零中的第二次`Ctrl+C`或急停则立即同时停止两臂。
将要失败时按`I`，先确认主臂夹爪电机失能，再将主臂切入零力；切换后不向主臂夹爪发送位置驱动命令，由人工操作夹爪，退出接管时恢复夹爪使能。接管等待期间从臂保持；移动主臂触发专家标记并完成纠正；再次按`I`让模型从当前最新观测重新
规划。退出接管时保留两臂当前关节与夹爪的相对偏置，主臂从自身当前位置执行模型目标的变化量，避免立即追到从臂的绝对位置；角色切换完成后重新读取观测再执行模型。同一个X trial可以接管多次，各段有独立`intervention_id`。任务结束后再按`X`，整个
X→X trial保存为一个complete raw episode；`R`、`Y`、急停、watchdog或退出中断的trial标为
incomplete，默认转换不会用于训练。
若当前trial不应保留，直接按`D`：程序删除本条HG临时目录并暂停，不保存complete或incomplete
episode；随后必须按`R`或长按`Y`回零，才能按`X`开始下一条。`D`只丢弃HG训练trial，按`V`
单独开启的评估录像仍按`V`保存或结束。
`--duration`只累计模型控制时间，人工接管期间暂停该计时，不会在恢复动作中途因试次超时切断专家控制。

每帧raw保存`control_source`（0=novice、1=armed、2=expert）、`expert_mask`、
`intervention_id`和`command_sent`。默认以关节相对移动`0.75°`或夹爪相对移动`1.5mm`为候选，连续2个控制周期达到阈值后触发；
用`--hg-start-joint-motion-deg`、`--hg-start-gripper-motion-mm`和`--hg-start-motion-frames`
调整。离合基准仍取按`I`时的主从臂姿态；按键后的等待帧标为armed，HG导出时自动排除。

HG编码器在允许trial启动前预热，按`X`时只启动episode时间线；按`I`和主臂有效移动时均不再
创建ffmpeg进程，因此编码器启动不会隔在第一条专家命令和对应视频帧之间。预热会提前创建临时session，若整个
进程没有产生episode，正常退出时会清理。完整接管数据默认输出到
`data/raw/piper_raw/hg_dagger_<时间戳>/episodes/episode_<编号>/`，因此不会与使用纯时间戳
目录名的普通专家数采混淆；转换后的LeRobot目录继承同一`hg_dagger_`前缀。episode包含`global.mp4`、`wrist.mp4`、
`frames.npz`和`episode.json`。`action`始终是该帧实际下发的安全限幅目标，不论来源是novice还是
expert；`novice_action`、`expert_requested_action`、`leader_state`和跟随误差作为诊断字段保留。
raw审核页面的曲线用紫色背景显示专家区间。转换器识别`training_export=expert_mask_only`后，会把
每个连续专家区间导出成标准ACT episode，并把novice/armed帧留在raw中。HG-DAgger不会关闭相机、
关节边界、反馈进展或急停看门狗。

#### HIL-SERL数据准备与后续在线学习

本节以同一个16万步ACT为基线；“ACT+RL残差”准备、训练和真机入口见[HIL-SERL操作说明](HIL_SERL.md)。
上面的`--hg-dagger`命令采集的是ACT试次；它可以为后续RL提供历史replay seed，不能冒充
由SAC actor在线产生的新数据。真正的HIL-SERL训练必须另开session，由RL actor控制并在训练中
持续产生online replay。

HIL-SERL不能把完整HG trial直接当作“全专家示范”：完整trial中的novice帧属于online replay，
只有`expert_mask=1`的transition应额外进入human/offline buffer。普通leader-follower专家示范则整条
都是human demo。项目现已实现这两类raw到LeRobot RL transition数据的无损准备，输出保留完整episode，
并新增`next.reward`、`next.done`、`complementary_info.is_intervention`、`control_source`、
`intervention_id`和`command_sent`。下面的ACT残差基线示例显式使用7维绝对关节动作；raw源不修改。

先打开raw审核页，逐条完成两项标记：质量状态选“保留”，`任务结果（HIL）`选“成功”或“失败”。
HIL准备入口只接受显式保留且有结果的完整trial；有裁剪、未标结果、待复查、剔除或incomplete的条目
不会被静默当成RL数据。

```bash
# 1. 审核由16万步ACT产生的历史HG试次，并逐条标成功/失败
robotarm modeling dataset-browser -- \
  --dataset-root data/raw/piper_raw/HG_DAGGER_ACT160K_SESSION

# 2. 预检：HG完整trial只作为历史replay seed，保留ACT与专家的全过程
robotarm modeling prepare-hil-serl -- \
  data/raw/piper_raw/HG_DAGGER_ACT160K_SESSION \
  --buffer-role replay --space joint --frame absolute \
  --output data/lerobot/hil_serl_act160k_replay_seed --dry-run

# 3. 数量确认后，去掉--dry-run才会真正转换
robotarm modeling prepare-hil-serl -- \
  data/raw/piper_raw/HG_DAGGER_ACT160K_SESSION \
  --buffer-role replay --space joint --frame absolute \
  --output data/lerobot/hil_serl_act160k_replay_seed
```

已有BC离线示范使用`data/lerobot/pick_up_flvv_20261115_curated`直接加载，不需要再次
从RAW转换或重编码视频。`HG_DAGGER_ACT160K_SESSION`替换为上一步实际生成的
`hg_dagger_<时间戳>`目录。

成功trial默认只有最后1帧`reward=1`，其余帧为0；失败trial全部为0；每条trial最后一帧
`done=true`。这适合稀疏奖励RL。只有在采集时已经让机械臂在成功状态稳定保持了一段时间、准备训练
视觉reward classifier时，才使用例如`--success-tail-frames 10`把最后10帧标为正样本；不要为了凑
正样本把尚未成功的帧误标为1。输出`meta/hil_serl_conversion.json`记录来源、结果、奖励帧数和接管统计。

在线阶段的正确数据流是：SAC actor控制Piper并把每个`(state, action, reward, next_state, done)`写入
online replay；按接管键后仍采用当前的主从臂离合与“主臂实际移动才开始接管”逻辑，实际执行的专家
transition同时写入online replay和human/offline buffer；learner按约1:1混合两个buffer更新critic和
actor，再周期性把新权重发回actor。人工接管频率应随成功率提高逐渐下降。

当前仓库已完成上述审核、reward/done生成、完整replay导出以及LeRobot 0.6.1 `ReplayBuffer`读回验证；
Piper残差SAC actor现通过`robotarm modeling hil-serl -- actor`复用原控制链，训练通过同入口的
`learner`子命令运行，两者使用本机原子文件同步。LeRobot自带环境目前针对SO系列总线接口，
不能绕过Piper的关节限位、反馈watchdog、急停和安全回零直接启动。可选依赖安装：

```bash
python -m pip install -e ".[hil-serl]"
```

在完成Piper Gym适配、固定安全工作空间、reset策略和shadow验证前，不要直接运行
`python -m lerobot.rl.actor`向真机发送探索动作。

#### HIL-SERL当前010000模型的运行与续训

2026-11-17：旧 `hil_act010000` 使用无时间集成的 ACT 首动作参考，与效果较好的原 ACT 执行链不同。
旧 learner 已正常保存并停止，原权重、RAW 和 replay 保留用于分析。旧 v1 run 不再允许续训或执行残差；
不要把旧缓存／残差权重复制到新 v2 run。当前先验证 ACT 基线，不启动在线训练。

**第一步：原 ACT 基线评估。** 此命令只从旧 run 读取 ACT 检查点位置，直接走原来的
`select_action`、`n_action_steps=1` 和 `temporal_ensemble_coeff=0.01`，不加载 SAC 权重，
不创建 HIL 特征、不向 learner 提交试次。使用同样的物体位置、相机和流程核对独立成功率：

```bash
robotarm modeling hil-serl -- actor \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1/hil_act010000 \
  --mode baseline --execute -- \
  --hg-leader-interface socketcan \
  --hg-leader-channel can_leader1 \
  --hg-setup-id flvv_act010000_baseline_v2
```

`--mode baseline` 也是 actor 的新默认值。启动双臂回零，`X` 开始／保存，`I` 接管／交还，
`D` 丢弃，`1/0` 只记录 RAW 人工结果；本模式不会提交在线 replay。正常 `Ctrl+C` 双臂回零、
保持使能后退出。退出后重跑同一命令即可；未完成的 trial 不能从中间续接。
数据使用 `hg_dagger_` 前缀，避免把原 ACT 基线试次冒充残差策略的新在线数据。

**第二步：基线确认后，才重新准备 v2 特征。** 以下默认只预览，追加 `--execute` 才实际转换，
不连接真机；必须使用新目录。时间集成需要按帧顺序计算，不能用 batch 同时编码不同时间步。
这一步及后续训练耗时较长，由操作者自行执行：

```bash
robotarm modeling hil-serl -- prepare \
  --dataset data/lerobot/flvv_hil_seed_round1 \
  --act-checkpoint artifacts/checkpoints/flvv_hg_then_hil_round1/act/checkpoints/010000/pretrained_model \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1/hil_act010000_v2 \
  --temporal-ensemble-coeff 0.01 \
  --device cuda
```

v2 缓存按 episode、接管／交还边界重置 ACT 时间集成；真实执行在安全限幅等待时冻结 ACT 时间轴，
丢弃慢帧时撤销该帧对时间集成的更新。新 RAW 还保存实际 ACT 参考，供异常提交后的准确重建。
现有历史 RAW 可保留后续转换使用；本命令只读取指定的 seed 数据集，不自动纳入新采集的 RAW。

**第三步：新 run 的 learner 与确定性残差评估。** 完成第二步实际转换后，终端 A 首次启动：

```bash
robotarm modeling hil-serl -- learner \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1/hil_act010000_v2 \
  --device cuda --steps 100000 --offline-steps 1000
```

learner 不连接真机；预热预算用完后等待标注试次。`Ctrl+C` 保存并退出，恢复同一新 run 时加
`--resume`。`--steps` 是累计总步数上限，恢复时不会重新预热 1000 步。
终端 B 显式选择 `--mode residual`，使用确定性残差均值做对比：

```bash
robotarm modeling hil-serl -- actor \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1/hil_act010000_v2 \
  --mode residual --execute -- \
  --hg-leader-interface socketcan \
  --hg-leader-channel can_leader1 \
  --hg-setup-id flvv_hil_act010000_v2
```

只在需要在线探索时，把 `--mode residual` 改为 `--mode explore`，显式启用随机残差。
`--deterministic` 旧参数仍兼容，但不能与 `--mode explore` 同用。先比较原 ACT 基线和确定性残差，
按独立成功率、接管次数、接管时长判断效果，不把人工救回的成功率当成模型独立成功率。

| 当前操作 | 后续行为 |
|---|---|
| `X` 结束当前 trial | 保存完整 RAW，暂停并等待 `1` 成功／`0` 失败 |
| `1`／`0` | 残差／探索模式提交结果给 learner；基线模式只保存 RAW 标注 |
| 同一进程继续 | 人工复位物体、`R` 回零，再按 `X` 开始下一条 |
| 暂停并退出 actor | 先 `X` → `1/0`，再正常一次 `Ctrl+C` |
| 再次启动 actor | 重跑对应模式命令，不加 `--resume`；新试次开始前同步已发布残差权重 |
| 直接在运行中退出／watchdog 中断 | 本条为 incomplete，不提交 learner；下一次从新 trial 开始 |
| `D` | 丢弃当前 trial；复位并回零后开始下一条 |

仅观测到命令总年龄超限时，残差模式会丢弃该帧，从新观测重新推理；连续第 3 帧超限仍停止试次，
需人工按 X。相机取帧过期、反馈／模式／关节异常和推理超过硬上限仍立即停止。
多次 I 接管时，夹爪到达开合边界后更新离合偏置，避免重复接管后只能在物体宽度附近开合。

如果残差试次已经 `X` 保存完整 RAW，但标结果／提交时退出，可使用离线补交；
`SESSION`／episode 替换为真实目录，`success/failure` 按实际结果选择：

```bash
robotarm modeling hil-serl -- recover \
  --run-dir artifacts/checkpoints/flvv_hg_then_hil_round1/hil_act010000_v2 \
  --raw-episode data/raw/piper_raw/SESSION/episodes/episode_000000 \
  --outcome failure
```

默认只预检，追加 `--execute` 才补交，不连接真机。只接受同一配置、完整非空且拥有对应 ACT 参考的
v2 RAW；重复补交不会重复入库，incomplete 不会补成完整试次。旧 v1 RAW 仍可使用旧 run 离线恢复，
但不会因此重新启动已停止的旧 learner。

采集后不要只用纠正片段单独训练；将它们聚合到原BC数据，并先人工审核。下面直接从raw创建
新的聚合数据集，第一次去掉`--execute`做预检；`SESSION`替换为本次时间戳目录：

```bash
robotarm modeling extend-dataset -- \
  --mode new \
  --base-dataset data/lerobot/piper_pick_cup_direct_223eps_trimmed_single_task_20261029 \
  --new-raw data/raw/piper_raw/SESSION \
  --output data/lerobot/piper_pick_cup_223eps_hg_round1 \
  --new-status review --new-label none \
  --execute
```

本入口实现的是HG-DAgger的人工门控、专家恢复采集和数据聚合闭环；尚未加入论文中基于多个
novice策略分歧自动学习的doubt阈值。当前安全性仍依赖人在模型进入不可恢复状态前及时按`I`。

预测不合理先查输入/模型/时序，不以放宽软件限位或取消watchdog解决。

#### ACT限幅与日志字段

下表是 2026-10-29 配置快照，不代表已验证适合所有模型；修改后以实际启动日志为准。

| 参数 | 当前 ACT 值 | 含义 |
|---|---:|---|
| `--max-joint-step-deg` | 2.5 | 单周期相对上次命令的变化上限 |
| `--max-joint-lead-deg` | 6 | 命令相对反馈的超前上限 |
| `--max-gripper-step-mm` / `--max-gripper-lead-mm` | 8 / 16 | 夹爪步长 / 超前上限 |
| `--warn-act-raw-joint-gap-deg` | 12 | 原始关节差软告警；继续执行限幅后的安全目标 |
| `--max-act-raw-joint-gap-deg` | 15 | 原始关节严重差距遥测阈值，不单独触发停机 |
| `--max-act-raw-gap-consecutive-steps` | 3 | 连续严重差距的告警/日志计数，不单独触发停机 |
| `--max-act-raw-gripper-gap-mm` | 30 | 原始夹爪差距告警阈值；执行仍由背压目标限幅 |
| `--max-act-raw-gripper-bound-tolerance-mm` | 1 | 原始夹爪边界容差，实际目标仍为合法开度 |
| `--max-act-raw-joint-bound-tolerance-deg` | 1 | ACT 原始关节预测允许的小幅越界；执行目标仍硬限幅到安全范围，超过容差才停机 |
| `--max-act-subgoal-no-progress-steps` | 10 | 硬限幅背压期间连续无反馈进展的停机拍数；旧参数名仍兼容 |
| `--act-subgoal-min-progress-deg` / `--act-subgoal-min-gripper-progress-mm` | 0.05 / 0.1 | 清零无进展计数所需的最小误差下降量 |
| `--warn-inference-ms` / `--max-inference-ms` | 60 / 500 | ACT 推理耗时软告警 / 硬停上限，ms；告警不打断执行 |
| `--max-camera-age-ms` | 50 | 观测年龄硬停上限，ms |
| `--n-action-steps` | YAML 当前为 30 | 在线消费步数；当前ACT真机命令显式覆盖为1 |
| `--temporal-ensemble-coeff` | CLI显式为0.01 | 启用ACT时间集成以平滑重叠chunk；要求`n_action_steps=1`，不改变训练loss |
| `--max-chunk-inference-ms` | 按策略覆盖 | 异步整块推理时限，与同步推理门槛区分 |
| `--report` | 显式指定 | 本次JSONL日志路径，用于诊断和审批，避免同名覆盖 |

关节或夹爪张开方向触发步长/超前限幅时，发送端锁定有界目标并暂停ACT动作消费；对应硬限幅解除后立即恢复
时间集成推理，不等待精确位置收敛。原始关节差超过12°/15°仅告警和记录；只有硬限幅仍在
生效且反馈连续无进展，或原始动作越过物理边界/非有限值，才触发安全暂停。
夹爪闭合时，物体可能使宽度无法继续减小；仅闭合宽度被限幅不再冻结 ACT，避免抓住后等待
夹爪到达物体内部的目标宽度、进而误报无进展。闭合指令仍执行原步长、超前、宽度及力设置，
关节限幅或张开无进展仍会暂停。这一规则不等于传感器已确认抓取成功，任务结果仍需人工判定。

| 日志字段 | 含义 |
|---|---|
| state / raw_action / safe_action | 反馈 / 原始预测 / 限幅目标，rad/m |
| act_tracking_action / act_control_action | 最新预测修正值 / 本拍控制器输入（锁存后为固定目标） |
| action_clamped / act_tracking_clamped | 原始预测被修改 / 平滑后目标仍被最终硬限幅 |
| act_backpressure_clamped | 是否需要冻结ACT等待关节或夹爪张开跟随；仅闭合夹爪限幅时为false |
| act_gripper_boundary_corrected | 是否修正夹爪端点 |
| act_subgoal_active / created / reached | 背压目标是否活动 / 本拍创建 / 本拍解除（兼容字段名） |
| act_policy_clock_frozen / act_backpressure_* | ACT时间序列是否暂停消费 / 背压创建、解除及无进展拍数 |
| act_subgoal_*_error / *_progress | 固定目标误差和本拍反馈进展，单位见字段名 |
| act_subgoal_no_progress_steps / act_hold_steps | 连续无进展拍数；后者为兼容字段 |
| act_raw_action_warnings / act_raw_gap_hard_steps | 原始差距告警 / 连续超过严重遥测阈值的拍数 |
| performance_warnings | 非致命性能告警（例如推理超过软阈值） |
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

上面命令是可选的只读分析模式，用于测量模型延迟、队列时序和原始动作，不发送运动命令。
Pi0真机执行不要求先运行shadow，也不检查shadow审批文件；在同参数命令追加
`--execute --experimental-pi0-execute`并改用新的execute日志路径即可。
当前属于实验功能，运行时安全看门狗仍然生效。

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

#### `PiperLiftCup-v1`：当前无把手杯抓起任务

当前实采任务的仿真见[抓杯仿真启动说明](simulation_lift_cup.md)。使用 Piper、棕色桌垫、
60/48/90 mm 无把手杯，支持直立/横躺/混合姿态，以稳定抓起离桌为成功条件。
`robotarm simulation lift-cup -- --cup-pose upright --seed 0 --render` 可运行可视化演示，
`--record --output 新目录` 可导出视频和评估结果。

#### `PiperPickCupToContainer-v1`（入箱任务）

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

以下与当前注册入口逐一对应，共48个（data-acquisition 11、modeling 21、simulation 16）。
每个Python入口的完整参数通常可用`robotarm <组> <命令> -- --help`查看；
`train`/`train-with-holdout`是shell包装，按第6节的位置参数格式使用。

### data-acquisition：硬件、遥操、原始数据

| 完整入口 | 功能／使用位置 |
|---|---|
| `robotarm data-acquisition read-status` | 只读机械臂反馈，第3节 |
| `robotarm data-acquisition enable-control` | enable/disable/reset，第3节，执行会改变硬件状态 |
| `robotarm data-acquisition reset-both-arms` | 一次复位主从两臂软件急停，第3节，可能失去保持力 |
| `robotarm data-acquisition home-piper` | 独立回零，第3节，执行会运动 |
| `robotarm data-acquisition restore-piper-role` | 恢复Follower角色，第3节，可能失去保持力 |
| `robotarm data-acquisition collect-leader-follower` | 自定义软件主从采集／只读／纯控制，第4节 |
| `robotarm data-acquisition collect-piper` | 手柄遥操采集、手柄监控、相机检查，第4节 |
| `robotarm data-acquisition xr-bridge` | PICO后台桥接，第4节 |
| `robotarm data-acquisition piper-xr` | 单/双臂PICO统一入口，第4节 |
| `robotarm data-acquisition piper-bimanual-xr` | 独立双臂PICO入口，参数见`-- --help`及PICO手册 |
| `robotarm data-acquisition merge-piper-raw` | raw→新raw独立合并，或事务追加到滚动MASTER，第5节 |
| `robotarm data-acquisition convert-piper-raw` | raw→LeRobot训练格式，第5节 |

### modeling：训练、评估与部署

| 完整入口 | 功能／使用位置 |
|---|---|
| `robotarm modeling check-environment` | 依赖、数据与CUDA检查，第6节 |
| `robotarm modeling train` | LeRobot训练包装，第6节 |
| `robotarm modeling train-with-holdout` | 固定留出比例的兼容包装，第6节 |
| `robotarm modeling train-pi0-lora` | 独立π0 LoRA训练计划／执行，第8节 |
| `robotarm modeling train-smolvla` | SmolVLA动作专家／LoRA训练计划与执行，见[SmolVLA训练](#smolvla直接微调) |
| `robotarm modeling train-smolvla-lora` | 上述SmolVLA训练器的兼容别名 |
| `robotarm modeling check-pi0-lora` | LoRA重新加载＋单帧离线预测，第8节 |
| `robotarm modeling convert-dataset` | 与convert-piper-raw相同的兼容入口，第5节 |
| `robotarm modeling dataset-browser` | 数据浏览、人工审查、逐episode裁剪与新数据集导出，第5节 |
| `robotarm modeling extend-dataset` | 事务式追加raw数据、继承审查标记或生成新合并集，第5节 |
| `robotarm modeling prepare-hil-serl` | 从Piper raw试次生成带奖励、完成和干预字段的HIL-SERL数据，第10节 |
| `robotarm modeling hil-serl` | ACT残差SAC特征准备、learner训练与续训、真机actor、完整RAW补交；见[运行与续训](#hil-serl当前010000模型的运行与续训) |
| `robotarm modeling relabel-dataset-task` | 保留源目录，发布统一语言task的新数据集，第5节 |
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
| `robotarm simulation lift-cup` | 当前无把手杯抓起任务、录像与评估，第10节 |
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
| `scripts/create_act_training_env.sh` | 新建本机/云端ACT训练专用Conda环境，并自动应用固定版本补丁 |
| `scripts/apply_lerobot_act_feature_layer_patch.sh` | 对`lerobot==0.6.1`应用/校验ACT layer3/layer4补丁 |
| `scripts/train_monitor.py` | `--run-dir RUN --port 8420 --interval 2`，实时loss网页 |
| `scripts/check_xr_frame.py` | `--hand right --yaw 0 --mirror none`，只读坐标诊断 |
| `scripts/xr_monitor.py` | `--yaw 0 --mirror lr`，XR状态/坐标观察；`--no-plot`关闭图窗 |
| `scripts/build_pi0_cloud_bundle.py` | 打包训练代码/数据/权重/tokenizer，不上传；见下方限制 |
| `scripts/cloud/run_smoke.sh` / `run_train.sh` | 云端试跑/正式训练包装，默认计划，追加`--execute`执行 |
| `scripts/cloud/check_smoke.sh` | 试跑权重重新加载，离线预测一帧 |
| `scripts/cloud/verify_bundle.py` | 在解压的云端包内验证逐文件校验和 |
| `scripts/extend_lerobot_dataset.py` | `extend-dataset`直接脚本入口；默认只预检，`--execute`后写入 |
| `robotarm.modeling.quality_filter` | `python -m ... --dataset-root ... --derive-from ...`，迁移同源审查结果 |
| `robotarm.modeling.reports.best_practice_figures` | 从已有JSON重画台账PNG/SVG，不加载模型 |
| `scripts/analyze_pi0_lora.py` / `compare_pi0_denoising.py` / `analyze_act_resume.py` | 旧兼容壳，优先用第7/8节统一报告命令 |

```bash
python scripts/check_xr_frame.py --hand right --yaw 0 --mirror none
python scripts/xr_monitor.py --yaw 0 --mirror lr --no-plot
python -m robotarm.modeling.reports.best_practice_figures
```

云端打包器默认选择LF180；若要训练其他数据，必须显式传`--dataset-root`，只改输出包名不会切换数据。
`--run-prefix`同时设置试跑、正式训练与监控输出名。基础权重仍为本地`pi0_base`。
例如将原始122条抓杯数据的**已裁剪导出**（115条）打包；替换tokenizer缓存仓库目录
（其中应有`refs/`、`snapshots/`），使用不存在的新输出名：

```bash
python scripts/build_pi0_cloud_bundle.py \
  --dataset-root data/lerobot/piper_pick_cup_20261019_direct_122eps_trimmed_20261022 \
  --run-prefix pi0_lora_cup122_trimmed \
  --batch-size 4 \
  --bundle-dir artifacts/cloud_exports/RobotArm_pi0_lora_cloud_cup122_trimmed_bs4_20261024 \
  --tokenizer-repo /PATH/TO/models--google--paligemma-3b-pt-224
```

会创建独立目录、`.tar.gz`和`.sha256`，占用较多磁盘。每个包根目录的`CLOUD_README.md`
记录该包实际数据集、启动与监控命令；训练配置及权重迁回说明见[云端手册](#85-云端包上传校验与启动)。
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
| Pi0 shadow | 可选的延迟、队列和动作分析模式；真机执行不依赖shadow审批，但仍需`--execute --experimental-pi0-execute` |
| clamp/raw gap/bound/underflow触发 | 查看本次JSONL、预测/反馈/限幅/时间戳，不盲目放宽门控 |
| 配置改了却没生效 | 重启进程，检查CLI是否覆盖YAML、policies是否覆盖通用值，看启动日志 |
| HIL／ACT退出后已有shell提示符，但输入不显示 | 在该终端盲输`stty sane`并回车恢复回显；`robotarm`外层现在会在Ctrl+C后等待子进程真正完成回零与清理，再恢复回显和整行输入，也修复启动前遗留的无回显状态；重新运行命令使用新代码 |
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
