from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotarm.modeling.quality_filter import derive_reviews, select_training_episodes


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
