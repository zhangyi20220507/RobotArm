"""XR-only clutch state, independent of hardware and recording."""
from dataclasses import dataclass


@dataclass
class Clutch:
    engage_threshold: float
    release_threshold: float
    engaged: bool = False
    blocked: bool = True
    mode: str = "grip"
    fault_latched: bool = False

    def pause(self):
        self.blocked = True
        self.fault_latched = True

    def resume(self):
        self.blocked = False
        self.fault_latched = False

    def update(self, grip: float, valid: bool) -> bool:
        if self.fault_latched:
            self.engaged = False
            return False
        if self.mode == "direct":
            if not valid:
                self.blocked = True
            self.engaged = not self.blocked
            return valid and self.engaged
        self.engaged = grip >= (self.release_threshold if self.engaged else self.engage_threshold)
        if not valid:
            self.blocked = True
        elif grip < self.release_threshold:
            self.blocked = False
        return valid and self.engaged and not self.blocked


def limit_joint_step(previous, requested, feedback, max_step, max_lead, lower, upper):
    """Keep joint ratios while intersecting step, following and joint bounds.

    If the previous command is no longer feasible, let the caller hold/reclutch
    rather than independently clipping joints and changing Cartesian direction.
    """
    import numpy as np

    delta = np.asarray(requested) - previous
    lo = np.maximum(lower, np.asarray(feedback) - max_lead)
    hi = np.minimum(upper, np.asarray(feedback) + max_lead)
    alpha_lo, alpha_hi = 0.0, 1.0
    peak = float(np.max(np.abs(delta)))
    if peak > max_step:
        alpha_hi = max_step / peak
    for base, step, minimum, maximum in zip(previous, delta, lo, hi):
        if abs(step) < 1e-12:
            if base < minimum - 1e-9 or base > maximum + 1e-9:
                raise RuntimeError('previous joint command outside feasible bounds; release and re-engage')
            continue
        a, b = (minimum - base) / step, (maximum - base) / step
        alpha_lo = max(alpha_lo, min(a, b))
        alpha_hi = min(alpha_hi, max(a, b))
    if alpha_hi < alpha_lo - 1e-9:
        raise RuntimeError('no direction-preserving joint step within limits; release and re-engage')
    return np.asarray(previous) + max(0.0, alpha_hi) * delta


def joint_limit_diagnostics(
    previous, requested, feedback, command, max_step, max_lead, lower, upper
):
    """Return the retained joint-step ratio and likely active limit sources."""
    import numpy as np

    previous = np.asarray(previous)
    requested = np.asarray(requested)
    feedback = np.asarray(feedback)
    command = np.asarray(command)
    delta = requested - previous
    requested_norm = float(np.linalg.norm(delta))
    if requested_norm <= 1.0e-12:
        return 1.0, []
    sent_norm = float(np.linalg.norm(command - previous))
    ratio = min(1.0, sent_norm / requested_norm)
    if ratio >= 1.0 - 1.0e-6:
        return ratio, []

    sources = []
    if float(np.max(np.abs(delta))) > max_step + 1.0e-9:
        sources.append("JOINT_STEP")
    if np.any(requested < feedback - max_lead - 1.0e-9) or np.any(
        requested > feedback + max_lead + 1.0e-9
    ):
        sources.append("JOINT_FOLLOW")
    if np.any(requested < np.asarray(lower) - 1.0e-9) or np.any(
        requested > np.asarray(upper) + 1.0e-9
    ):
        sources.append("JOINT_BOUND")
    if not sources:
        sources.append("JOINT_FEASIBILITY")
    return ratio, sources


def add_input_options(parser, default_engagement="direct"):
    parser.add_argument("--engagement-mode", choices=("direct", "grip"), default=default_engagement,
                        help="grip: hold Grip to follow, release to hold position; "
                             "direct: follow after Y/B enable without holding Grip")
    parser.add_argument("--max-hand-jump-m", type=float, default=0.05,
                        help="pause on a controller position jump between received poses (meters)")
    parser.add_argument("--max-hand-jump-deg", type=float, default=30.0,
                        help="pause on a controller rotation jump between received poses (degrees)")


def validate_input_options(args):
    import math
    if not math.isfinite(args.max_hand_jump_m) or not 0 < args.max_hand_jump_m <= 0.2:
        raise ValueError("max-hand-jump-m must be in (0, 0.2]")
    if not math.isfinite(args.max_hand_jump_deg) or not 0 < args.max_hand_jump_deg <= 90:
        raise ValueError("max-hand-jump-deg must be in (0, 90]")


class PoseJumpGuard:
    """Compare received poses on SO(3), including quaternion sign equivalence."""
    def __init__(self, max_translation, max_rotation_degrees):
        self.max_translation = max_translation
        self.max_rotation_degrees = max_rotation_degrees
        self.previous = None

    def update(self, pose):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from .geometry import valid_xr_pose
        pose = np.asarray(pose, dtype=float)
        if not valid_xr_pose(pose):
            self.previous = None
            return "invalid controller pose"
        previous, self.previous = self.previous, pose.copy()
        if previous is None:
            return None
        distance = float(np.linalg.norm(pose[:3] - previous[:3]))
        angle = float(np.degrees((Rotation.from_quat(pose[3:]) *
                                 Rotation.from_quat(previous[3:]).inv()).magnitude()))
        if distance > self.max_translation or angle > self.max_rotation_degrees:
            return f"controller jump: {distance * 1000:.1f} mm / {angle:.1f} deg"
        return None
