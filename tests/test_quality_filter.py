from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotarm.modeling.quality_filter import default_reviews_path, derive_reviews, select_training_episodes
from robotarm.modeling.dataset_trim import exported_reviews
from robotarm.modeling.dataset_browser import DatasetStore


class QualityFilterTest(unittest.TestCase):
    def make_files(self, total: int, episodes: dict[str, dict]) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "dataset"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": total}))
        reviews = Path(temporary.name) / "reviews.json"
        reviews.write_text(json.dumps({"dataset_root": str(root.resolve()), "episodes": episodes}))
        return root, reviews, temporary

    def test_includes_keep_and_unreviewed(self) -> None:
        root, reviews, temporary = self.make_files(
            5,
            {
                "0": {"status": "keep"},
                "1": {"status": "reject"},
                "2": {"status": "review"},
                "3": {"status": ""},
            },
        )
        self.addCleanup(temporary.cleanup)

        selected, counts = select_training_episodes(root, reviews)

        self.assertEqual(selected, [0, 3, 4])
        self.assertEqual(dict(counts), {"keep": 1, "reject": 1, "review": 1, "unreviewed": 2})

    def test_rejects_reviews_for_another_dataset(self) -> None:
        root, reviews, temporary = self.make_files(1, {})
        self.addCleanup(temporary.cleanup)
        document = json.loads(reviews.read_text())
        document["dataset_root"] = str(root.parent / "other")
        reviews.write_text(json.dumps(document))

        with self.assertRaisesRegex(ValueError, "belongs to"):
            select_training_episodes(root, reviews)

    def test_rejects_out_of_range_episode(self) -> None:
        root, reviews, temporary = self.make_files(1, {"1": {"status": "keep"}})
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "outside dataset range"):
            select_training_episodes(root, reviews)

    def test_unapplied_crop_cannot_silently_train_on_full_episode(self):
        root, reviews, temporary = self.make_files(2, {"0": {
            "status": "keep", "trim": {"start_frame": 100, "end_frame": 280}}})
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "unapplied trim"):
            select_training_episodes(root, reviews)

    def test_rejected_episode_crop_does_not_block_other_episodes(self):
        root, reviews, temporary = self.make_files(2, {"0": {
            "status": "reject", "trim": {"start_frame": 100, "end_frame": 280}}})
        self.addCleanup(temporary.cleanup)
        self.assertEqual(select_training_episodes(root, reviews)[0], [1])


class ExportedReviewsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "trimmed"
        (self.root / "meta").mkdir(parents=True)
        self.info = {"fps": 20, "total_episodes": 2, "total_frames": 10}
        (self.root / "meta/info.json").write_text(json.dumps(self.info))
        self.manifest = {
            "format": "robotarm_dataset_trim_v1", "fps": 20,
            "range_semantics": "[start_frame, end_frame)", "total_frames": 10,
            "source_dataset": "/unavailable/source", "reviews_snapshot": {}, "episodes": [],
        }
        for index, source_id in enumerate((2, 8)):
            review = {"status": "keep" if index == 0 else "", "note": "保留备注",
                      "trim": {"start_frame": 3, "end_frame": 8, "source_length": 12, "fps": 20}}
            self.manifest["reviews_snapshot"][str(source_id)] = review
            self.manifest["episodes"].append({"source_episode": source_id, "output_episode": index,
                "start_frame": 3, "end_frame": 8, "source_length": 12, "frames": 5, "review": review})
        self.path = self.root / "meta/trim_provenance.json"
        self.write_manifest()
        self.enterContext(patch("robotarm.modeling.quality_filter.ARTIFACTS_ROOT", Path(temporary.name) / "artifacts"))
        self.reviews = default_reviews_path(self.root)

    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_existing_export_is_remapped_without_writes_or_source_access(self):
        document = exported_reviews(self.root)
        self.assertEqual(set(document["episodes"]), {"0", "1"})
        self.assertEqual(document["episodes"]["0"], {"status": "keep", "note": "保留备注"})
        selected, counts = select_training_episodes(self.root, self.reviews, 1000)
        self.assertEqual(sorted(selected), [0, 1])
        self.assertEqual(counts["keep"], 1)
        self.assertEqual(counts["unreviewed"], 1)
        self.assertEqual(selected, select_training_episodes(self.root, self.reviews, 1000)[0])
        self.assertFalse(self.reviews.exists())

    def test_browser_inherits_then_saved_review_takes_precedence(self):
        rows = [{"episode_index": i, "length": 5} for i in range(2)]
        with patch.object(DatasetStore, "_load_episode_rows", return_value=rows):
            store = DatasetStore(self.root, self.reviews)
        self.assertEqual(store.reviews()["0"]["status"], "keep")
        self.assertNotIn("trim", store.reviews()["0"])
        store.save_review(0, "reject", "new review")
        self.assertEqual(select_training_episodes(self.root, self.reviews)[0], [1])
        self.assertEqual(store.reviews()["1"]["note"], "保留备注")
        store.save_review(1, "keep", "", {"start_s": .05, "end_s": .2})
        with self.assertRaisesRegex(ValueError, "unapplied trim"):
            select_training_episodes(self.root, self.reviews)

    def test_missing_custom_review_or_non_export_still_errors(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            select_training_episodes(self.root, self.root / "typo.json")
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "not found"):
            select_training_episodes(self.root, self.reviews)

    def test_incomplete_export_is_rejected_even_with_manual_reviews(self):
        (self.root / "TRIM_INCOMPLETE.txt").write_text("failed")
        self.reviews.parent.mkdir(parents=True)
        self.reviews.write_text(json.dumps({"episodes": {}}))
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            select_training_episodes(self.root, self.reviews)

    def test_invalid_manifest_is_not_silently_accepted(self):
        mutations = [
            lambda m: m.update(total_frames=11),
            lambda m: m.update(fps=30),
            lambda m: m["episodes"][1].update(output_episode=0),
            lambda m: m["episodes"][1].update(source_episode=2),
            lambda m: m["episodes"][0].update(frames=6),
            lambda m: m["episodes"][0]["review"].update(status="reject"),
        ]
        original = json.dumps(self.manifest)
        for mutation in mutations:
            self.manifest = json.loads(original)
            mutation(self.manifest)
            self.write_manifest()
            with self.subTest(manifest=self.manifest), self.assertRaisesRegex(ValueError, "invalid trim provenance"):
                select_training_episodes(self.root, self.reviews)


class ShuffledSplitTest(unittest.TestCase):
    """LeRobot holds out the LAST ceil(n*eval_split) of the list it receives, so
    the order this filter prints is the train/eval split."""

    def setUp(self) -> None:
        # make_files owns the temp dir and returns it; hold the handle or it is
        # collected before the test runs.
        root, reviews, temporary = QualityFilterTest.make_files(
            self, 10, {str(i): {"status": "keep"} for i in range(10)}
        )
        self.addCleanup(temporary.cleanup)
        self.root, self.reviews = root, reviews

    def test_default_order_is_ascending(self) -> None:
        selected, _ = select_training_episodes(self.root, self.reviews)
        self.assertEqual(selected, list(range(10)))

    def test_seed_is_deterministic(self) -> None:
        first, _ = select_training_episodes(self.root, self.reviews, shuffle_seed=7)
        second, _ = select_training_episodes(self.root, self.reviews, shuffle_seed=7)
        self.assertEqual(first, second)

    def test_shuffle_is_a_permutation(self) -> None:
        selected, counts = select_training_episodes(self.root, self.reviews, shuffle_seed=7)
        self.assertEqual(sorted(selected), list(range(10)))
        self.assertEqual(counts["keep"], 10)

    def test_different_seeds_give_different_holdouts(self) -> None:
        # The held-out tail is what LeRobot uses for eval.
        tails = set()
        for seed in range(8):
            selected, _ = select_training_episodes(self.root, self.reviews, shuffle_seed=seed)
            tails.add(tuple(selected[-2:]))
        self.assertGreater(len(tails), 1, "seeds produced identical holdouts")

    def test_shuffled_holdout_is_not_the_sequential_tail(self) -> None:
        selected, _ = select_training_episodes(self.root, self.reviews, shuffle_seed=7)
        self.assertNotEqual(selected[-2:], [8, 9])


class DeriveReviewsTest(unittest.TestCase):
    """A dataset rebuilt from the same sessions (e.g. a camera subset) inherits
    the review decisions of its source."""

    def make_target(self, total: int) -> tuple[Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "rebuilt"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": total}))
        return root, temporary

    def make_source(self, temporary_root: Path, total: int, episodes: dict) -> Path:
        source_root = temporary_root / "original"
        (source_root / "meta").mkdir(parents=True)
        (source_root / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": total})
        )
        path = temporary_root / "source_reviews.json"
        path.write_text(
            json.dumps({"dataset_root": str(source_root.resolve()), "episodes": episodes})
        )
        return path

    def test_carries_decisions_over_and_retargets_root(self) -> None:
        root, temporary = self.make_target(4)
        self.addCleanup(temporary.cleanup)
        source = self.make_source(
            Path(temporary.name), 4, {"0": {"status": "keep"}, "1": {"status": "review"}}
        )
        destination = Path(temporary.name) / "derived.json"

        derived = derive_reviews(root, source, destination)

        self.assertEqual(derived["dataset_root"], str(root.resolve()))
        self.assertEqual(
            derived["episodes"],
            {"0": {"status": "keep"}, "1": {"status": "review"}},
        )
        self.assertEqual(derived["derived_from"]["path"], str(source.resolve()))
        # The derived file must satisfy the guard that rejects foreign reviews.
        selected, counts = select_training_episodes(root, destination)
        self.assertEqual(selected, [0, 2, 3])
        self.assertEqual(counts["review"], 1)

    def test_rejects_episode_count_mismatch(self) -> None:
        root, temporary = self.make_target(3)
        self.addCleanup(temporary.cleanup)
        source = self.make_source(
            Path(temporary.name), 5, {"4": {"status": "keep"}}
        )
        destination = Path(temporary.name) / "derived.json"

        with self.assertRaisesRegex(ValueError, "cannot derive reviews onto rebuilt"):
            derive_reviews(root, source, destination)
        self.assertFalse(destination.exists())

    def test_trim_ranges_are_not_copied_to_a_rebuilt_timeline(self):
        root, temporary = self.make_target(1)
        self.addCleanup(temporary.cleanup)
        source = self.make_source(Path(temporary.name), 1, {"0": {
            "status": "keep", "trim": {"start_frame": 2, "end_frame": 5}}})
        destination = Path(temporary.name) / "derived.json"
        with self.assertRaisesRegex(ValueError, "containing trim"):
            derive_reviews(root, source, destination)
        self.assertFalse(destination.exists())

    def test_refuses_to_overwrite_without_flag(self) -> None:
        root, temporary = self.make_target(1)
        self.addCleanup(temporary.cleanup)
        source = self.make_source(Path(temporary.name), 1, {"0": {"status": "keep"}})
        destination = Path(temporary.name) / "derived.json"
        destination.write_text('{"sentinel": true}')

        with self.assertRaisesRegex(ValueError, "already exists; pass --overwrite"):
            derive_reviews(root, source, destination)
        self.assertEqual(json.loads(destination.read_text()), {"sentinel": True})

        derive_reviews(root, source, destination, overwrite=True)
        self.assertEqual(json.loads(destination.read_text())["dataset_root"], str(root.resolve()))


if __name__ == "__main__":
    unittest.main()
