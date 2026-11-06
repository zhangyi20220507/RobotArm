# 当前抓杯任务：ManiSkill 仿真

环境 ID：`PiperLiftCup-v1`。参考本地 2026-10-19 抓杯数据中的任务
`Pick up the cup in different poses` 和全局相机画面，使用项目已有 Piper URDF，
棕色桌垫与无把手白色空心杯。杯口直径 60 mm、杯底直径 48 mm、高 90 mm，
是用户确认的首版建模尺寸，并非相机标定测量结果。

杯子使用刚性、带锥度的复合碰撞体；没有纸杯形变、压瘪模型。材质、质量、摩擦、
相机外参尚未实测标定，因此这是任务原型，不是已完成标定的数字孪生。

## 启动

```bash
conda activate robotarm

# 自动抓取直立杯；运行结束输出成功判定
robotarm simulation lift-cup -- --cup-pose upright --seed 0 --render

# 横躺杯的实验脚本，自动根据杯轴调整夹爪方向；目前仍可能抓取失败
robotarm simulation lift-cup -- --cup-pose side --seed 0 --render

# 不打开窗口，保存双相机/第三人称视频、首帧、末帧和 JSON 结果
# 输出目录必须是新目录，以免覆盖已有实验
robotarm simulation lift-cup -- --cup-pose upright --seed 0 \
  --record --output artifacts/simulation/my_cup_demo

# CPU 物理仿真，不渲染；测试连续 10 个 seed，随机直立/横躺
robotarm simulation lift-cup -- --cup-pose mixed --episodes 10 --seed 0 \
  --output artifacts/simulation/my_cup_evaluation

# 只查看场景；空格暂停，r 重置，1/2/3/4 切换相机，q 退出
robotarm simulation viewer -- --env-id PiperLiftCup-v1 --zero-action --view split
```

VNC 下可添加 `--display :1`。可视化和录像需要可用的 Vulkan/GPU 渲染环境；
不带 `--record`、`--render` 时无需 GPU。尺寸可用
`--cup-top-diameter-mm`、`--cup-bottom-diameter-mm`、`--cup-height-mm` 调整。

## 环境与判定

环境默认随机直立/横躺、平面位置和朝向。演示命令默认直立，方便复现。
成功要求整个杯子（包括倾斜时的最低碰撞点）离桌垫至少 70 mm，双指存在方向相反的
接触力，并且杯子和手臂保持稳定，连续满足 10 个控制步（20 Hz 下为 0.5 s）。
仅接触、单指推起、掉落途中经过目标高度都不会算成功。

物理步频为 500 Hz，控制步频为 20 Hz，回合上限 700 步。轻杯版本沿用同一 Piper
模型，并单独使用更柔和的夹爪驱动参数；演示以带关节限位的逆解生成关节位置目标。
`evaluate()` 中包含 `success`、
`is_grasped`、`is_lifted`、`is_stable`、`cup_clearance`、`hold_count`、`fail`。
`cup_clearance` 单位为米，跌落桌面以下会触发 `fail`。
支持 `state`、`rgb` 等 ManiSkill 观测模式；相机键为
`observation.images.wrist` 和 `observation.images.global`。

```python
import gymnasium as gym
import robotarm.simulation.environment  # 注册自定义环境

env = gym.make(
    "PiperLiftCup-v1",
    cup_pose="mixed",  # upright / side / mixed
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    reward_mode="normalized_dense",
    sim_backend="cpu",
    render_backend="none",  # 图像观测需要改为 gpu
)
obs, info = env.reset(seed=0)
# action: XYZ增量(3) + 旋转增量(3) + 两个夹爪关节位置(2)
# 控制器动作约定以 env.action_space 为准；关节控制也可选 pd_joint_pos。
env.close()
```

演示是读取仿真物体状态的脚本基线，用于验证环境、碰撞和抓取，并非视觉策略或
已训练的 ACT / π0。随机姿态的失败也会如实写入 `results.json`，不要将所有演示
自动当作合格训练数据。旧 `force-pick-cup`、`collect-lerobot`、`run-act` 仍对应
`PiperPickCupToContainer-v1`，不应直接作为本任务的策略评测入口。

当前验证：直立杯已录制物理接触抓起并稳定保持的成功示例，见
`artifacts/simulation/lift_cup_no_pause_v2/demo.mp4` 与同目录的 `results.json`。
抓取前的对齐现在会检测停滞，避免达到关节限位后等待整个步数预算耗尽；初始静置
为 0.25 s。结果中的 `stage_steps` 记录各阶段控制步数，除以 20 即为仿真秒数。
修正后 seed 0/1/2 均成功，分别为 242/212/259 步；seed 0 相比原版 373 步，
完整动作仿真时间从 18.65 s 降至 12.1 s。
横躺姿态能正常生成、复位、仿真和评估，但当前脚本尚未实现可靠抓起；已观察到
`no_bilateral_contact` 和 `force_lost_during_lift`，仍需优化低位进爪轨迹。
这部分不能当作已经验证成功的专家策略。

环境按照 [ManiSkill 自定义任务接口](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html)
注册，可继续用于模仿学习或强化学习。
