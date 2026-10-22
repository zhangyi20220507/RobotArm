"""手柄末端(DLS)模式摇杆映射回归测试。

左摇杆 = X/Y 平移，右摇杆上下 = Z，右摇杆左右 = Yaw，十字键 = Roll/Pitch；
轴之间互相独立——姿态输入不再把平移清零（旧版本会让左摇杆看起来没反应）。
"""

import unittest
from types import SimpleNamespace

from robotarm.teleop.gamepad.device import DPAD_LEFT, DPAD_RIGHT, DPAD_UP, DPAD_DOWN
from robotarm.teleop.gamepad.inputs import cartesian_inputs, joint_inputs

FULL = 32767
DEADZONE = 0.18


def pad(left_x=0, left_y=0, right_x=0, right_y=0, buttons=0):
    return SimpleNamespace(
        buttons=buttons, left_x=left_x, left_y=left_y,
        right_x=right_x, right_y=right_y,
    )


def axes(inp):
    return dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), inp))


class CartesianMappingTests(unittest.TestCase):
    def test_left_stick_drives_translation_only(self):
        self.assertAlmostEqual(axes(cartesian_inputs(pad(left_y=FULL), DEADZONE))["x"], 1.0)
        self.assertAlmostEqual(axes(cartesian_inputs(pad(left_y=-FULL), DEADZONE))["x"], -1.0)
        self.assertAlmostEqual(axes(cartesian_inputs(pad(left_x=FULL), DEADZONE))["y"], -1.0)
        self.assertAlmostEqual(axes(cartesian_inputs(pad(left_x=-FULL), DEADZONE))["y"], 1.0)

    def test_right_stick_vertical_is_z(self):
        self.assertAlmostEqual(axes(cartesian_inputs(pad(right_y=FULL), DEADZONE))["z"], 1.0)
        self.assertAlmostEqual(axes(cartesian_inputs(pad(right_y=-FULL), DEADZONE))["z"], -1.0)

    def test_right_stick_horizontal_is_yaw(self):
        self.assertAlmostEqual(axes(cartesian_inputs(pad(right_x=FULL), DEADZONE))["yaw"], 1.0)

    def test_shared_right_stick_prioritizes_z(self):
        result = axes(cartesian_inputs(pad(right_x=FULL, right_y=FULL // 4), DEADZONE))
        self.assertEqual(result["yaw"], 0.0)
        self.assertGreater(result["z"], 0.0)
        result = axes(cartesian_inputs(pad(right_y=FULL, right_x=FULL // 4), DEADZONE))
        self.assertAlmostEqual(result["z"], 1.0)
        self.assertEqual(result["yaw"], 0.0)

    def test_left_translation_cannot_swallow_right_z(self):
        result = cartesian_inputs(pad(left_y=FULL, right_x=FULL, right_y=FULL // 2), DEADZONE)
        # Runtime's translation-priority stage must leave Z intact.
        if any(result[:3]):
            result[3:] = [0., 0., 0.]
        self.assertGreater(result[2], 0.)

    def test_orientation_does_not_suppress_translation(self):
        """旧版本在这里会把 x/y 清零，导致左摇杆像坏了。"""
        result = axes(cartesian_inputs(pad(left_y=FULL, right_x=FULL), DEADZONE))
        self.assertAlmostEqual(result["x"], 1.0)
        self.assertAlmostEqual(result["yaw"], 1.0)
        result = axes(cartesian_inputs(pad(left_y=FULL, buttons=DPAD_UP), DEADZONE))
        self.assertAlmostEqual(result["x"], 1.0)
        self.assertAlmostEqual(result["pitch"], 1.0)
        result = axes(cartesian_inputs(pad(left_x=FULL, right_y=FULL), DEADZONE))
        self.assertAlmostEqual(result["y"], -1.0)
        self.assertAlmostEqual(result["z"], 1.0)

    def test_dpad_roll_pitch(self):
        self.assertAlmostEqual(axes(cartesian_inputs(pad(buttons=DPAD_LEFT), DEADZONE))["roll"], -1.0)
        self.assertAlmostEqual(axes(cartesian_inputs(pad(buttons=DPAD_RIGHT), DEADZONE))["roll"], 1.0)
        self.assertAlmostEqual(axes(cartesian_inputs(pad(buttons=DPAD_DOWN), DEADZONE))["pitch"], -1.0)

    def test_deadzone_silences_drift(self):
        result = axes(cartesian_inputs(pad(left_x=5000, left_y=5000, right_x=5000), DEADZONE))
        self.assertEqual(result, {"x": 0.0, "y": 0.0, "z": 0.0,
                                  "roll": 0.0, "pitch": 0.0, "yaw": 0.0})

    def test_zero_input_is_neutral(self):
        self.assertEqual(axes(cartesian_inputs(pad(), DEADZONE)),
                         {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0})


class JointMappingTests(unittest.TestCase):
    def test_joint_inputs_unchanged(self):
        result = joint_inputs(pad(left_x=FULL, left_y=FULL, right_x=FULL,
                                  right_y=FULL, buttons=DPAD_RIGHT | DPAD_UP), DEADZONE)
        self.assertEqual([round(v, 3) for v in result], [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
