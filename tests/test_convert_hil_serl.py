"""HIL-SERL preparation keeps full trials and binds rewards to reviewed outcomes."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow.parquet as pq

from robotarm.modeling.raw_dataset_browser import RawDatasetStore
from robotarm.modeling.raw_hil_replay import RawHILReplayDataset
from robotarm.modeling.manual_hil_outcome import save_manual_outcome
from robotarm.record.convert import write_json
from robotarm.record.convert_hil_serl import main


class HilSerlConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, *, hg=True):
        root = self.root / ("hg" if hg else "demo")
        session = {
            "format_version": "piper_raw_v1",
            "record_fps": 20,
            "task": "Pick component",
            "state_names": [f"joint_{i}" for i in range(1, 7)] + ["gripper_width"],
            "state_units": ["rad"] * 6 + ["m"],
            "action_semantics": "commanded MOVE_JS joint targets and gripper target",
            "robot": {"training_export": "expert_mask_only"} if hg else {},
            "cameras": {
                "global": {
                    "serial": "camera-1",
                    "stream": {"width": 32, "height": 24},
                }
            },
        }
        write_json(root / "session.json", session)
        episode = root / "episodes" / "episode_000000"
        write_json(
            episode / "episode.json",
            {
                "episode_index": 0,
                "frames": 6,
                "status": "complete",
                "task": "Pick component",
                "camera_roles": ["global"],
            },
        )
        values = np.arange(6, dtype=np.float32) / 100
        arrays = {
            "timestamp": np.arange(6) / 20,
            "observation_state": np.repeat(values[:, None], 7, axis=1),
            "action": np.repeat((values + 0.01)[:, None], 7, axis=1),
            "observation_tcp_pose": np.zeros((6, 6)),
            "teleop_delta": np.zeros((6, 6)),
            "camera_host_timestamp": np.zeros((6, 1)),
            "camera_sensor_timestamp": np.zeros((6, 1)),
            "camera_frame_number": np.arange(6)[:, None],
            "camera_frame_age": np.zeros((6, 1)),
            "camera_skew": np.zeros(6),
        }
        if hg:
            arrays.update(
                {
                    "control_source": np.asarray([0, 1, 2, 2, 0, 0], dtype=np.int8),
                    "expert_mask": np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int8),
                    "intervention_id": np.asarray([-1, 0, 0, 0, -1, -1], dtype=np.int64),
                    "command_sent": np.asarray([1, 1, 1, 1, 0, 1], dtype=np.int8),
                    "novice_action": np.repeat((values + 0.02)[:, None], 7, axis=1),
                }
            )
        np.savez(episode / "frames.npz", **arrays)
        writer = cv2.VideoWriter(
            str(episode / "global.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (32, 24)
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(6):
                writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
        finally:
            writer.release()
        return root

    def test_sampled_mixed_replay_preserves_failure_reward_and_bc_expert_source(self):
        hg = self.source(hg=True)
        bc = self.source(hg=False)
        for root, outcome in ((hg, "failure"), (bc, "success")):
            RawDatasetStore(root, export_root=self.root / "unused").save_review(
                0, "keep", "reviewed", labels={"task_outcome": outcome}
            )
        output = self.root / "sampled_hil"
        self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", self.root / "hf-cache"))
        main([str(hg), str(bc), "--output", str(output), "--space", "joint",
              "--frame", "absolute", "--selection", "hg-failures-plus-success",
              "--hg-success-count", "0", "--bc-success-count", "1"])
        manifest = json.loads((output / "meta/hil_serl_conversion.json").read_text())
        self.assertEqual(manifest["selection"]["actual_counts"], {
            "hg_failure": 1, "hg_success": 0, "bc_failure": 0, "bc_success": 1,
        })
        rows = [row for file in (output / "data").rglob("*.parquet")
                for row in pq.read_table(file).to_pylist()]
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(row["next.reward"] for row in rows), 1)
        self.assertEqual(sum(row["next.done"] for row in rows), 2)
        self.assertEqual(sum(row["complementary_info.is_intervention"] for row in rows), 8)

    def test_manual_outcome_ignores_prewarmed_directory_without_metadata(self):
        from robotarm.record.convert import discover_episodes
        root = self.source(hg=True)
        (root / "episodes/.episode_000001.in_progress").mkdir()
        episode = root / "episodes/episode_000000"
        self.assertEqual(save_manual_outcome(root, episode, "failure"), 0)
        for include_incomplete in (False, True):
            _, selected, skipped = discover_episodes(root, include_incomplete)
            self.assertEqual([item.path for item in selected], [episode])
            self.assertTrue(any("prewarming" in reason for reason in skipped))
        # A malformed published episode is still an error, not silently hidden.
        (root / "episodes/episode_000001").mkdir()
        with self.assertRaisesRegex(RuntimeError, "cannot read JSON"):
            discover_episodes(root, False)

    def test_recover_completed_raw_trial_submits_failure_once(self):
        from types import SimpleNamespace
        from robotarm.modeling.hil_serl.config import Config
        from robotarm.modeling.hil_serl.recover import recover
        from robotarm.modeling.hil_serl.replay import read_episode
        root = self.source(hg=True)
        run = self.root / "run"
        cfg = Config("fake", "hash", 30, ["global"], task="Pick component")
        cfg.save(run / "config.json")
        session = json.loads((root / "session.json").read_text())
        session["robot"]["hil_config_signature"] = cfg.signature
        write_json(root / "session.json", session)
        (root / "episodes/.episode_000001.in_progress").mkdir()
        args = SimpleNamespace(run_dir=run, raw_episode=root / "episodes/episode_000000",
                               outcome="failure", device="cpu", execute=True)
        from unittest.mock import Mock
        prior = Mock()
        prior.roles = ["global"]
        prior.encode.return_value = (np.zeros(30, np.float32), np.zeros(7, np.float32))
        with patch("robotarm.modeling.hil_serl.recover.ACTPrior.load", return_value=prior), patch(
            "robotarm.modeling.hil_serl.recover.checkpoint_hash", return_value="hash"
        ):
            self.assertEqual(recover(args), 0)
            self.assertEqual(recover(args), 0)
        outputs = list((run / "online").glob("episode_*.npz"))
        self.assertEqual(len(outputs), 1)
        data, meta = read_episode(outputs[0], cfg)
        self.assertEqual(len(data["state"]), 5)  # One non-commanding RAW frame is excluded.
        self.assertEqual(data["reward"].sum(), 0)
        self.assertEqual(data["expert"].sum(), 2)
        self.assertTrue(meta["recovered_from_raw"])
        self.assertEqual(prior.encode.call_count, 6)
        reviews = json.loads((root / "annotations/reviews.json").read_text())
        self.assertEqual(reviews["episodes"]["0"]["labels"]["task_outcome"], "failure")

    def test_full_hg_trial_exports_reward_done_and_intervention_mask(self):
        source = self.source(hg=True)
        store = RawDatasetStore(source, export_root=self.root / "unused")
        store.save_review(
            0,
            "keep",
            "successful recovery",
            labels={"task_outcome": "success"},
        )
        output = self.root / "hil"
        self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", self.root / "hf-cache"))
        main(
            [
                str(source),
                "--output",
                str(output),
                "--space",
                "joint",
                "--frame",
                "absolute",
            ]
        )

        rows = pq.read_table(next((output / "data").rglob("*.parquet"))).to_pylist()
        self.assertEqual(len(rows), 6)
        self.assertEqual([row["next.reward"] for row in rows], [0, 0, 0, 0, 0, 1])
        self.assertEqual([row["next.done"] for row in rows], [False] * 5 + [True])
        self.assertEqual(
            [row["complementary_info.is_intervention"] for row in rows],
            [False, False, True, True, False, False],
        )
        self.assertEqual(
            [row["complementary_info.novice_action_valid"] for row in rows],
            [True, False, False, False, False, True],
        )
        self.assertAlmostEqual(rows[0]["complementary_info.novice_action"][0], 0.02)
        manifest = json.loads((output / "meta/hil_serl_conversion.json").read_text())
        self.assertEqual(manifest["buffer_role"], "replay")
        self.assertEqual(manifest["episodes"][0]["intervention_frames"], 2)
        self.assertFalse((output / "CONVERSION_INCOMPLETE.txt").exists())

        from lerobot.datasets import LeRobotDataset
        from lerobot.rl.buffer import ReplayBuffer

        dataset = LeRobotDataset(
            repo_id=manifest["repo_id"], root=output, download_videos=False
        )
        replay = ReplayBuffer.from_lerobot_dataset(
            dataset,
            device="cpu",
            storage_device="cpu",
            state_keys=["observation.state", "observation.images.global"],
            use_drq=False,
        )
        self.assertEqual(len(replay), 6)
        self.assertEqual(replay.rewards.tolist(), [0, 0, 0, 0, 0, 1])

    def test_kept_trial_requires_outcome_and_hg_cannot_be_offline_demo(self):
        source = self.source(hg=True)
        RawDatasetStore(source).save_review(0, "keep", "missing result")
        with self.assertRaisesRegex(ValueError, "task_outcome"):
            main([str(source), "--output", str(self.root / "missing"), "--space", "joint"])
        RawDatasetStore(source).save_review(
            0, "keep", "", labels={"task_outcome": "failure"}
        )
        with self.assertRaisesRegex(ValueError, "offline-demo refuses full HG"):
            main(
                [
                    str(source),
                    "--output",
                    str(self.root / "offline"),
                    "--space",
                    "joint",
                    "--buffer-role",
                    "offline-demo",
                ]
            )

    def test_dry_run_validates_without_creating_dataset(self):
        source = self.source(hg=True)
        RawDatasetStore(source).save_review(
            0, "keep", "", labels={"task_outcome": "failure"}
        )
        output = self.root / "preview_only"
        self.assertEqual(
            main([str(source), "--output", str(output), "--space", "joint", "--dry-run"]), 0
        )
        self.assertFalse(output.exists())

    def test_reviewed_hg_trial_loads_directly_from_raw(self):
        source = self.source(hg=True)
        RawDatasetStore(source).save_review(
            0, "keep", "", labels={"task_outcome": "success"}
        )
        dataset = RawHILReplayDataset(source)
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(float(dataset[0]["next.reward"].item()), 0.0)
        self.assertTrue(dataset[2]["complementary_info.is_intervention"].item())
        self.assertFalse(dataset[2]["complementary_info.novice_action_valid"].item())
        self.assertEqual(float(dataset[5]["next.reward"].item()), 1.0)
        self.assertTrue(dataset[5]["next.done"].item())

        from lerobot.rl.buffer import ReplayBuffer
        replay = ReplayBuffer.from_lerobot_dataset(
            dataset, device="cpu", storage_device="cpu",
            state_keys=["observation.state", "observation.images.global"],
            use_drq=False,
        )
        self.assertEqual(len(replay), 6)
        self.assertEqual(replay.rewards.tolist(), [0, 0, 0, 0, 0, 1])

    def test_operator_outcome_is_saved_to_same_raw_review_sidecar(self):
        source = self.source(hg=True)
        episode = source / "episodes" / "episode_000000"
        self.assertEqual(save_manual_outcome(source, episode, "failure"), 0)
        review = RawDatasetStore(source).reviews()["0"]
        self.assertEqual(review["labels"]["task_outcome"], "failure")
        self.assertEqual(review["review_origin"], "manual")
        with self.assertRaisesRegex(ValueError, "already has a manual review"):
            save_manual_outcome(source, episode, "success")

        next_episode = source / "episodes" / "episode_000001"
        shutil.copytree(episode, next_episode)
        metadata_path = next_episode / "episode.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["episode_index"] = 1
        metadata_path.write_text(json.dumps(metadata))
        self.assertEqual(save_manual_outcome(source, next_episode, "success"), 1)
        reviews = RawDatasetStore(source).reviews()
        self.assertEqual(reviews["0"]["labels"]["task_outcome"], "failure")
        self.assertEqual(reviews["1"]["labels"]["task_outcome"], "success")

    def test_ordinary_demonstration_is_all_human_offline_data(self):
        source = self.source(hg=False)
        RawDatasetStore(source).save_review(
            0, "keep", "", labels={"task_outcome": "success"}
        )
        output = self.root / "offline_demo"
        self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", self.root / "hf-cache"))
        main(
            [
                str(source),
                "--output",
                str(output),
                "--space",
                "joint",
                "--buffer-role",
                "offline-demo",
            ]
        )
        rows = pq.read_table(next((output / "data").rglob("*.parquet"))).to_pylist()
        self.assertTrue(all(row["complementary_info.is_intervention"] for row in rows))


if __name__ == "__main__":
    unittest.main()
