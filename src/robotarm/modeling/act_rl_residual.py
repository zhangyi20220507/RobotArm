"""Bound a learned absolute joint target around the current ACT target."""

from __future__ import annotations

import math


def blend_act_rl(
    act_action,
    rl_action,
    *,
    gain: float,
    max_joint_delta_deg: float,
    max_gripper_delta_mm: float,
):
    """Return ACT plus a bounded RL correction in Piper's 7D action units.

    The caller must still pass this target through online_policy's feedback,
    acceleration, step, lead, physical-bound and emergency-stop gates.
    """
    import numpy as np

    if not all(math.isfinite(v) for v in (gain, max_joint_delta_deg, max_gripper_delta_mm)):
        raise ValueError("residual parameters must be finite")
    if not 0 <= gain <= 1 or max_joint_delta_deg <= 0 or max_gripper_delta_mm <= 0:
        raise ValueError("invalid residual gain or limits")
    base = np.asarray(act_action, dtype=np.float64)
    learned = np.asarray(rl_action, dtype=np.float64)
    if base.shape != (7,) or learned.shape != (7,):
        raise ValueError("ACT and RL actions must both have 7 dimensions")
    if not np.isfinite(base).all() or not np.isfinite(learned).all():
        raise ValueError("ACT or RL action is non-finite")
    limit = np.asarray(
        [math.radians(max_joint_delta_deg)] * 6 + [max_gripper_delta_mm / 1000.0],
        dtype=np.float64,
    )
    correction = np.clip((learned - base) * gain, -limit, limit)
    return base + correction
