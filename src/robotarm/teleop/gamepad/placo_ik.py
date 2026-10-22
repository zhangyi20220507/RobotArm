"""Optional gamepad adapter; never changes the XR solver or talks to hardware."""

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ...core.kinematics.dls_ik import IKError
from ..xr.piper_ik import PiperIK


class GamepadPlacoIK:
    def __init__(self, dt: float, tcp_offset: float):
        self.ik = PiperIK(dt, orientation_enabled=False)
        self.offset = np.array([0.0, 0.0, tcp_offset])
        # This private model instance uses link6's operational frame at the
        # TCP. Joint placements and the on-disk/PICO model are unchanged.
        frame = self.ik.robot.model.frames[self.ik.robot.model.getFrameId("link6")]
        frame.placement.translation += frame.placement.rotation @ self.offset
        self.ik.robot.update_kinematics()
        self.last_solve_ms = 0.0
        self.posture_anchor = None

    def reset_anchor(self):
        """Capture a new reference on the next command after pause/mode change."""
        self.posture_anchor = None

    def __call__(self, fk, target, seed, bounds, *, active_axes, **_kwargs):
        started = time.perf_counter()
        try:
            return self._solve(fk, target, seed, bounds, active_axes)
        except (RuntimeError, ValueError) as exc:
            raise IKError(f"Placo: {exc}") from exc
        finally:
            self.last_solve_ms = (time.perf_counter() - started) * 1000.0

    def _solve(self, fk, target, seed, bounds, active_axes):
        ik = self.ik
        for index, (low, high) in enumerate(bounds):
            ik.robot.set_joint_limits(f"joint{index + 1}", low, high)
        ik.set_state(np.asarray(seed))
        if self.posture_anchor is None:
            self.posture_anchor = np.asarray(seed).copy()
        # Fixed engagement posture, as in XR: do not ratchet the null-space
        # reference toward each successive feedback sample.
        ik.joints_task.set_joints({f"joint{i + 1}": float(q) for i, q in enumerate(self.posture_anchor)})
        orient = any(axis >= 3 for axis in active_axes)
        # Explicit rotation holds the complete target attitude; translation
        # keeps only a soft attitude preference, as in the PICO position mode.
        ik.orientation_task.configure("gamepad_orientation", "soft", 1.0 if orient else 1e-5)
        desired_rotation = Rotation.from_euler("xyz", target[3:]).as_matrix()
        ik.orientation_task.R_world_frame = desired_rotation
        for _ in range(12):
            # The frame Jacobian now includes the TCP lever arm. Validate with
            # independent SDK FK as well, catching model/frame disagreement.
            ik.ee_task.target_world = np.asarray(target[:3])
            ik.solver.solve(True)
            ik.robot.update_kinematics()
            result = np.asarray(ik.robot.state.q[7:13]).copy()
            if not np.all(np.isfinite(result)):
                raise IKError("Placo returned non-finite joints")
            if any(q < lo - 1e-6 or q > hi + 1e-6 for q, (lo, hi) in zip(result, bounds)):
                raise IKError("Placo exceeded joint limits")
            achieved = fk(result.tolist())
            error = float(np.linalg.norm(np.asarray(achieved[:3]) - target[:3]))
            angle = (Rotation.from_euler("xyz", achieved[3:]) * Rotation.from_matrix(desired_rotation).inv()).magnitude()
            if error <= 0.0008 and (not orient or angle <= math.radians(0.05)):
                return result.tolist()
        raise IKError(f"Placo residual {error * 1000:.2f} mm / {math.degrees(angle):.2f} deg")
