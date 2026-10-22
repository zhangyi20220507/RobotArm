import unittest

import numpy as np

from robotarm.modeling.pi05_piper_adapter import (
    ActionContract, ArmFeedback, decode_and_validate, map_camera_frames,
)


CONFIG = {"joint_limits": {f"joint{i}": (-1.0, 1.0) for i in range(1, 7)}}
ZERO = np.zeros(7)


class Pi05PiperAdapterTests(unittest.TestCase):
    def test_camera_mapping_and_freshness(self) -> None:
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        result = map_camera_frames({"global": (image, 9.95)}, now=10.0)
        np.testing.assert_allclose(
            result["observation.images.base_0_rgb"][:, 0, 0],
            [30 / 255, 20 / 255, 10 / 255],
        )
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            map_camera_frames(
                {"global": (image, 9.95)}, require_wrists=True, now=10.0
            )
        with self.assertRaisesRegex(RuntimeError, "stale"):
            map_camera_frames({"global": (image, 9.0)}, now=10.0)

    def test_absolute_and_delta_can_targets(self) -> None:
        kwargs = dict(
            feedback={"left": ArmFeedback(ZERO, 9.95)},
            reference={"left": ZERO}, previous={"left": ZERO},
            configs={"left": CONFIG}, now=10.0,
        )
        action = np.array([0.001, 0, 0, 0, 0, 0, 0.0001])
        target = decode_and_validate(
            action, ActionContract(("left",), "absolute", True), **kwargs
        )["left"]
        self.assertEqual(
            target.can_calls(),
            (("move_js", [0.001, 0, 0, 0, 0, 0]), ("move_gripper_m", 0.0001)),
        )
        reference = np.array([0.1, 0, 0, 0, 0, 0, 0])
        kwargs.update(
            feedback={"left": ArmFeedback(reference, 9.95)},
            reference={"left": reference}, previous={"left": reference},
        )
        target = decode_and_validate(
            action,
            ActionContract(("left",), "delta_from_observation", True),
            **kwargs,
        )["left"]
        self.assertAlmostEqual(target.joints_rad[0], 0.101)

    def test_rejects_base_dimension_stale_and_unsafe_plan(self) -> None:
        kwargs = dict(
            feedback={"left": ArmFeedback(ZERO, 9.95)},
            reference={"left": ZERO}, previous={"left": ZERO},
            configs={"left": CONFIG}, now=10.0,
        )
        contract = ActionContract(("left",), "absolute", True)
        with self.assertRaisesRegex(RuntimeError, "uncalibrated"):
            decode_and_validate(
                ZERO, ActionContract(("left",), "absolute"), **kwargs
            )
        with self.assertRaisesRegex(RuntimeError, "7D"):
            decode_and_validate(np.zeros(32), contract, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "step"):
            decode_and_validate(
                np.array([0.1, 0, 0, 0, 0, 0, 0]), contract, **kwargs
            )
        kwargs["feedback"] = {"left": ArmFeedback(ZERO, 9.0)}
        with self.assertRaisesRegex(RuntimeError, "stale"):
            decode_and_validate(ZERO, contract, **kwargs)

    def test_dual_arm_step_is_all_or_nothing(self) -> None:
        kwargs = dict(
            feedback={
                "left": ArmFeedback(ZERO, 9.95),
                "right": ArmFeedback(ZERO, 9.95),
            },
            reference={"left": ZERO, "right": ZERO},
            previous={"left": ZERO, "right": ZERO},
            configs={"left": CONFIG, "right": CONFIG}, now=10.0,
        )
        contract = ActionContract(
            ("left", "right"), "delta_from_observation", True
        )
        self.assertEqual(
            set(decode_and_validate(np.zeros(14), contract, **kwargs)),
            {"left", "right"},
        )
        bad = np.zeros(14)
        bad[7] = 0.1
        with self.assertRaisesRegex(RuntimeError, "right J1"):
            decode_and_validate(bad, contract, **kwargs)


if __name__ == "__main__":
    unittest.main()
