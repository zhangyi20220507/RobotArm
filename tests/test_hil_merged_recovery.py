"""Raw consolidation must preserve the replay identity of an online trial."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotarm.modeling.hil_serl.recover import recover


class MergedRecoveryTests(unittest.TestCase):
    def test_original_and_merged_trials_find_same_existing_submission(self):
        for merged in (False, True):
            with self.subTest(merged=merged), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                origin = root / "original/episodes/episode_000003"
                episode = root / "merged/episodes/episode_000000" if merged else origin
                episode.mkdir(parents=True)
                session = episode.parent.parent
                (session / "session.json").write_text(json.dumps({"robot": {"hil_config_signature": "sig"}}))
                meta = {"status": "complete", "frames": 3, "task": "Pick"}
                if merged:
                    meta["merge_source"] = {"episode_path": str(origin)}
                (episode / "episode.json").write_text(json.dumps(meta))
                run = root / "run"
                (run / "online").mkdir(parents=True)
                digest = hashlib.sha256(str(origin).encode()).hexdigest()
                spool = run / "online" / f"episode_{digest}.npz"
                spool.touch()
                cfg = SimpleNamespace(signature="sig", task="Pick")
                args = SimpleNamespace(run_dir=run, raw_episode=episode, outcome="success", execute=True)
                with patch("robotarm.modeling.hil_serl.recover.Config.load", return_value=cfg), patch(
                    "robotarm.modeling.hil_serl.recover.read_episode",
                    return_value=({}, {"raw_episode": str(origin), "outcome": "success"}),
                ) as reader, patch("robotarm.modeling.hil_serl.recover.write_episode") as writer:
                    self.assertEqual(recover(args), 0)
                    reader.assert_called_once_with(spool, cfg)
                    writer.assert_not_called()
                    args.outcome = "failure"
                    with self.assertRaisesRegex(ValueError, "conflicts"):
                        recover(args)
