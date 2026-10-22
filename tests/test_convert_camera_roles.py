"""Camera-role selection for piper_raw -> LeRobot conversion."""

from __future__ import annotations

import unittest

from robotarm.record.convert import select_camera_roles


class SelectCameraRolesTests(unittest.TestCase):
    def test_no_request_keeps_every_recorded_role(self):
        self.assertEqual(
            select_camera_roles(["global", "wrist"], None), ["global", "wrist"]
        )
        self.assertEqual(
            select_camera_roles(["global", "wrist"], []), ["global", "wrist"]
        )

    def test_subset_keeps_recorded_order(self):
        # The written feature order must not depend on flag order, so a
        # `--camera-role wrist --camera-role global` still yields global first.
        self.assertEqual(
            select_camera_roles(["global", "wrist"], ["wrist", "global"]),
            ["global", "wrist"],
        )
        self.assertEqual(
            select_camera_roles(["global", "wrist"], ["global"]), ["global"]
        )

    def test_single_camera_session_is_unchanged(self):
        self.assertEqual(select_camera_roles(["global"], None), ["global"])
        self.assertEqual(select_camera_roles(["global"], ["global"]), ["global"])

    def test_unrecorded_role_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not recorded.*available: global"):
            select_camera_roles(["global"], ["wrist"])

    def test_duplicate_role_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate --camera-role: global"):
            select_camera_roles(["global", "wrist"], ["global", "global"])

    def test_selection_does_not_mutate_input(self):
        recorded = ["global", "wrist"]
        select_camera_roles(recorded, ["global"])
        self.assertEqual(recorded, ["global", "wrist"])


if __name__ == "__main__":
    unittest.main()
