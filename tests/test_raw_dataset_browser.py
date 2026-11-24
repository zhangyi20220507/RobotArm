"""Raw-first curation: source binding, remerge, and one-pass video conversion."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from robotarm.modeling.raw_dataset_browser import RawCollectionStore, RawDatasetStore
from robotarm.modeling.dataset_trim import export_plan, exported_reviews
from robotarm.record.convert import convert_episode, discover_episodes, main as convert_main, write_json
from robotarm.record.merge import build_plan, execute_plan
from robotarm.record.raw_reviews import episode_identity, load_raw_reviews


class RawCurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, name, episodes=2):
        root = self.root / name
        write_json(root / "session.json", {
            "format_version": "piper_raw_v1", "record_fps": 20,
            "task": "Pick component", "state_names": [f"joint{i}" for i in range(6)] + ["gripper"],
            "state_units": ["rad"] * 6 + ["m"],
            "action_semantics": "commanded MOVE_JS joint targets and gripper target",
            "cameras": {role: {"serial": role, "stream": {"width": 32, "height": 24}}
                        for role in ["global", "wrist"]},
        })
        for index in range(episodes):
            episode = root / "episodes" / f"episode_{index:06d}"
            write_json(episode / "episode.json", {
                "episode_index": index, "frames": 6, "status": "complete",
                "task": "Pick component", "camera_roles": ["global", "wrist"],
            })
            values = np.arange(6, dtype=np.float32) + index * 10
            arrays = {"timestamp": np.arange(6) / 20,
                      "observation_state": np.repeat(values[:, None], 7, axis=1),
                      "action": np.repeat((values + .5)[:, None], 7, axis=1)}
            for key, width in {"observation_tcp_pose": 6, "teleop_delta": 6,
                               "camera_host_timestamp": 2, "camera_sensor_timestamp": 2,
                               "camera_frame_number": 2, "camera_frame_age": 2,
                               "camera_skew": None}.items():
                arrays[key] = np.zeros((6, width) if width else (6,))
            np.savez(episode / "frames.npz", **arrays)
            for role in ["global", "wrist"]:
                writer = cv2.VideoWriter(str(episode / f"{role}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (32, 24))
                self.assertTrue(writer.isOpened())
                try:
                    for frame in range(6):
                        writer.write(np.full((24, 32, 3), frame * 30, dtype=np.uint8))
                finally:
                    writer.release()
        return root

    def test_raw_preview_and_reviews_do_not_require_lerobot_or_modify_payloads(self):
        source = self.source("raw")
        before = {str(path.relative_to(source)): path.read_bytes()
                  for path in source.rglob("*") if path.is_file()}
        store = RawDatasetStore(source, export_root=self.root / "lerobot")
        payload = store.episode_payload(0)
        self.assertEqual(payload["action"][2][0], 2.5)
        self.assertEqual(payload["videos"][0]["from"], 0)
        self.assertTrue(store.video_path(0, 0).is_file())
        store.save_review(0, "keep", "valid", {"start_s": .05, "end_s": .2},
                          {"cup_orientation": "upright"})
        restored = RawDatasetStore(source, export_root=self.root / "lerobot")
        self.assertEqual(restored.reviews()["0"]["trim"]["start_frame"], 1)
        self.assertTrue(restored.meta_payload()["raw_reviews"])
        for relative, content in before.items():
            self.assertEqual((source / relative).read_bytes(), content)
        self.assertFalse((source / "meta/info.json").exists())

    def test_collection_reviews_are_saved_to_their_original_raw_sessions(self):
        bc = self.source("bc", episodes=2)
        hg = self.source("hg", episodes=1)
        store = RawCollectionStore([bc, hg])
        self.assertEqual(store.meta_payload()["episodes"][2]["source_session"], "hg")
        self.assertEqual(store.video_path(2, 0), hg / "episodes/episode_000000/global.mp4")
        store.save_review(2, "keep", "corrected", labels={"task_outcome": "failure"})
        self.assertEqual(RawDatasetStore(hg).reviews()["0"]["labels"]["task_outcome"], "failure")
        self.assertFalse((bc / "annotations/reviews.json").exists())
        self.assertEqual(store.reviews()["2"]["status"], "keep")

    def test_stale_review_identity_is_rejected(self):
        source = self.source("raw")
        store = RawDatasetStore(source)
        store.save_review(0, "keep", "")
        path = source / "episodes/episode_000000/frames.npz"
        with np.load(path) as data:
            values = dict(data)
        values["action"] = values["action"] + .1
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "identity/order/content"):
            store.reviews()

    def test_source_changed_before_first_review_cannot_be_annotated(self):
        source = self.source("raw", episodes=1)
        store = RawDatasetStore(source)
        path = source / "episodes/episode_000000/frames.npz"
        with np.load(path) as data:
            values = dict(data)
        values["action"] = values["action"] + .1
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "identity/order/content"):
            store.save_review(0, "keep", "")
        self.assertFalse(store.reviews_path.exists())

    def test_cli_honors_default_reviews_before_creating_output(self):
        source = self.source("raw", episodes=1)
        RawDatasetStore(source).save_review(0, "reject", "bad")
        output = self.root / "clean"
        with self.assertRaisesRegex(ValueError, "没有可导出"):
            convert_main([str(source), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_explicit_missing_reviews_and_incomplete_merge_are_rejected(self):
        source = self.source("raw", episodes=1)
        with self.assertRaises(FileNotFoundError):
            convert_main([str(source), "--reviews", str(self.root / "missing.json")])
        (source / "MERGE_INCOMPLETE.txt").write_text("unfinished")
        with self.assertRaisesRegex(ValueError, "merge is incomplete"):
            discover_episodes(source, False)

    def test_reviews_survive_remerge_and_renumbering(self):
        first, second, third = [self.source(name, episodes=1) for name in ["a", "b", "c"]]
        store = RawDatasetStore(first)
        store.save_review(0, "keep", "source a", {"start_s": .05, "end_s": .2},
                          {"cup_orientation": "sideways"})
        merged = self.root / "merged"
        plan, episodes = build_plan([second, first], merged)
        execute_plan(plan, episodes)
        remapped = RawDatasetStore(merged)
        self.assertEqual(remapped.reviews()["1"]["note"], "source a")
        self.assertEqual(episode_identity(episodes[1]), episode_identity(remapped.sources[1]))
        again = self.root / "merged_again"
        plan, episodes = build_plan([third, merged], again)
        execute_plan(plan, episodes)
        reviews = RawDatasetStore(again).reviews()
        self.assertEqual(reviews["2"]["labels"], {"cup_orientation": "sideways"})
        self.assertEqual(reviews["2"]["trim"]["start_frame"], 1)

    def test_merge_refuses_changed_content_after_review_snapshot(self):
        source = self.source("raw", episodes=1)
        RawDatasetStore(source).save_review(0, "keep", "reviewed data")
        output = self.root / "merged"
        plan, episodes = build_plan([source], output)
        path = episodes[0].path / "frames.npz"
        with np.load(path) as data:
            values = dict(data)
        values["action"] = values["action"] + .1
        np.savez(path, **values)
        with self.assertRaisesRegex(RuntimeError, "content changed"):
            execute_plan(plan, episodes)
        self.assertTrue((output / "MERGE_INCOMPLETE.txt").is_file())
        self.assertFalse((output / "session.json").exists())

    def test_one_pass_conversion_keeps_frame_alignment_and_absolute_actions(self):
        source = self.source("raw", episodes=1)
        _, episodes, _ = discover_episodes(source, False)
        writer = Mock()
        rows = convert_episode(writer, episodes[0], ["global", "wrist"],
                               start_frame=2, end_frame=5)
        self.assertEqual(rows, 3)
        self.assertEqual([call.args[0]["action"][0] for call in writer.add_frame.call_args_list],
                         [2.5, 3.5, 4.5])
        self.assertEqual([call.args[0]["observation.state"][0] for call in writer.add_frame.call_args_list],
                         [2, 3, 4])
        for call, expected in zip(writer.add_frame.call_args_list, [60, 90, 120]):
            for role in ["global", "wrist"]:
                self.assertLess(abs(float(call.args[0][f"observation.images.{role}"].mean()) - expected), 8)
        writer.save_episode.assert_called_once()

    def test_raw_label_filter_snapshots_selection_for_single_conversion(self):
        source = self.source("raw")
        store = RawDatasetStore(source, export_root=self.root / "exports")
        store.save_review(0, "keep", "", labels={"cup_orientation": "upright"})
        store.save_review(1, "keep", "", labels={"cup_orientation": "sideways"})
        with patch("robotarm.modeling.dataset_browser.threading.Thread") as thread:
            initial = store.start_export("upright", {"field": "cup_orientation", "value": "upright"})
        self.assertEqual(initial["total_episodes"], 1)
        with patch("robotarm.modeling.raw_dataset_browser.convert.main") as convert:
            thread.call_args.kwargs["target"]()
        self.assertEqual(convert.call_count, 1)
        document = convert.call_args.kwargs["review_documents"][source]
        self.assertEqual(document["episodes"]["1"]["status"], "reject")
        self.assertEqual(store.reviews()["1"]["status"], "keep")

    def test_real_curated_export_rebuilds_indices_and_inherits_labels_without_old_trims(self):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            self.skipTest(str(exc))
        self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", self.root / "hf-cache"))
        source = self.source("raw")
        store = RawDatasetStore(source, export_root=self.root / "lerobot")
        store.save_review(0, "reject", "bad")
        store.save_review(1, "keep", "good", {"start_s": .05, "end_s": .2},
                          {"cup_orientation": "sideways"})
        output = self.root / "lerobot" / "clean"
        updates = []
        plan = export_plan(store.info, store.episodes, store.reviews())
        result = store.export_selected(output, plan, store.reviews(), updates.append)
        self.assertEqual(result["frames"], 3)
        self.assertFalse((output / "CONVERSION_INCOMPLETE.txt").exists())
        import pyarrow.parquet as pq
        data_path = next((output / "data").rglob("*.parquet"))
        rows = pq.read_table(data_path).to_pylist()
        self.assertEqual([row["episode_index"] for row in rows], [0, 0, 0])
        self.assertEqual([row["frame_index"] for row in rows], [0, 1, 2])
        np.testing.assert_allclose([row["action"][0] for row in rows], [11.5, 12.5, 13.5])
        review = exported_reviews(output)["episodes"]["0"]
        self.assertEqual(review["labels"], {"cup_orientation": "sideways"})
        self.assertNotIn("trim", review)
        self.assertEqual(updates[-1]["frames_done"], 3)
        manifest = json.loads((output / "meta/piper_conversion.json").read_text())
        self.assertEqual(manifest["episodes"][0]["start_frame"], 1)
        self.assertEqual(manifest["episodes"][0]["end_frame"], 4)

    def test_full_hg_trial_exports_only_contiguous_expert_segments(self):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", self.root / "hf-cache"))
        source = self.source("hg_raw", episodes=1)
        session_path = source / "session.json"
        session = json.loads(session_path.read_text())
        session["robot"] = {"training_export": "expert_mask_only"}
        write_json(session_path, session)
        archive_path = source / "episodes/episode_000000/frames.npz"
        with np.load(archive_path) as data:
            arrays = dict(data)
        arrays.update({
            "control_source": np.asarray([0, 2, 2, 0, 2, 2], dtype=np.int8),
            "expert_mask": np.asarray([0, 1, 1, 0, 1, 1], dtype=np.int8),
            "intervention_id": np.asarray([-1, 0, 0, -1, 1, 1], dtype=np.int64),
            "command_sent": np.ones(6, dtype=np.int8),
        })
        np.savez_compressed(archive_path, **arrays)

        output = self.root / "lerobot" / "hg_expert"
        convert_main([str(source), "--output", str(output)])

        import pyarrow.parquet as pq
        rows = pq.read_table(next((output / "data").rglob("*.parquet"))).to_pylist()
        self.assertEqual([row["episode_index"] for row in rows], [0, 0, 1, 1])
        np.testing.assert_allclose(
            [row["action"][0] for row in rows], [1.5, 2.5, 4.5, 5.5]
        )
        manifest = json.loads((output / "meta/piper_conversion.json").read_text())
        self.assertTrue(manifest["expert_only"])
        self.assertEqual(
            [item["expert_segment_index"] for item in manifest["episodes"]],
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
