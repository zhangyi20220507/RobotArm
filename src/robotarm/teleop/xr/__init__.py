"""XR/PICO 遥操作包。

- bridge：订阅持久 XR 桥接进程的 ZeroMQ 客户端
- bridge_host：XR 桥接发布进程
- geometry：PICO 跟踪坐标系 → Piper 基坐标系的变换与限位
- piper_ik：基于 Placo 的本地逆解
- runtime：PICO 数采/遥操作主循环
"""
