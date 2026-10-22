# 测试说明

[返回 README](../docs/README.md)

这些文件是可重复的回归测试，不是历史训练产物。没有因“对应旧问题”而删除测试。
运行时使用`robotarm`环境；硬件路径通过mock隔离，不应连接CAN、相机或启动机械臂。
真实GPU离线评估是独立分析命令，结果写入`artifacts/reports/`，不塞进单元测试。

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
# 定向排查
python -m unittest discover -s tests -p 'test_online_policy_*.py' -v
python -m unittest discover -s tests -p 'test_leader_follower*.py' -v
python -m unittest discover -s tests -p 'test_gamepad*.py' -v
python -m unittest discover -s tests -p 'test_pi0*.py' -v
python -m unittest discover -s tests -p 'test_training_report.py' -v
```

| 组 | 主要覆盖 |
|---|---|
| `test_leader_follower*`、`test_home*`、`test_piper_home`、`test_restore_piper_role` | 主从启动、对齐、回零、角色、异常清理 |
| `test_gamepad*` | DLS/Placo接口、输入、目标、步长与拒绝行为 |
| `test_pico*` | XR坐标、单/双臂、安全恢复及控制模式 |
| `test_camera_preview`、`test_recorder*`、`test_convert*` | 相机窗口、预览节流、录制与转换 |
| `test_merge_raw` | 原始session复制、连续编号、来源追踪、兼容检查、重复/覆盖保护；使用微型NPZ/视频夹具 |
| `test_online_policy*`、`test_pi0_async`、`test_approve*`、`test_validate_runtime` | 预测边界、端点修正、异步过期/断供、审批 |
| `test_act_latent_report`、`test_pi0_*report`、`test_training_report`、`test_report_entrypoints` | padding、单位、配对、续训合并、报告兼容入口 |
| `test_pi0_lora`、`test_pi0_cloud_bundle`、`test_pi05*` | LoRA计划/重载约束、无凭据打包、基础模型适配 |
| `test_quality_filter`、`test_train_monitor`、`test_trajectory_studio` | 划分与审查、监控不修改run、Open-loop适配 |
| `test_dataset_trim` | 独立时间范围、标记保存/撤销、路径保护、失败状态；真实导出进度计数、视频保存失败不计完成、新任务归零；微型双相机LeRobot真实导出、视频/动作对齐及末端padding |
| `test_dataset_browser_i18n` | 中英文标签与提示、语言记忆、存储不可用回退；导出进度条与语言切换；切换不丢失未保存备注/裁剪和播放状态 |
| `test_readme_workflows` | README全量CLI覆盖、导航链接、shell语法与常用参数静态校验；不会执行示例命令 |

测试使用临时目录保存夹具。不要把真实checkpoint或视频复制到tests。
断言需要升级时说明行为变化，不通过删除失败断言来“让测试通过”。
测试日志里可能出现模拟急停、超时或参数错误；看最终`OK`/失败用例，而不是仅看输出关键词。
保持平铺结构以兼容当前unittest发现命令与跨用例夹具；暂不为了目录整齐移动全部测试。
