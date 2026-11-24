"""Outcome sampling keeps whole trajectories and their source classification."""
from types import SimpleNamespace
import unittest

from robotarm.record.convert_hil_serl import sample_hil_trials


def trial(name, hg, outcome):
    return {"episode_id": name, "outcome": outcome,
            "episode": SimpleNamespace(session={
                "robot": {"training_export": "expert_mask_only"} if hg else {}
            })}


class HilTrialSamplingTests(unittest.TestCase):
    def setUp(self):
        self.items = (
            [trial(f"failure-{i}", True, "failure") for i in range(4)]
            + [trial(f"hg-success-{i}", True, "success") for i in range(8)]
            + [trial(f"bc-success-{i}", False, "success") for i in range(10)]
            + [trial("bc-failure", False, "failure")]
        )

    def select(self, items=None, **kwargs):
        return sample_hil_trials(self.items if items is None else items,
                                 hg_success_count=2, bc_success_count=3, seed=1000, **kwargs)

    def test_all_hg_failures_and_only_requested_successes(self):
        result = self.select()
        ids = {x["episode_id"] for x in result}
        self.assertEqual(len(result), 9)
        self.assertTrue({f"failure-{i}" for i in range(4)} <= ids)
        self.assertNotIn("bc-failure", ids)
        self.assertEqual(sum(x.startswith("hg-success") for x in ids), 2)
        self.assertEqual(sum(x.startswith("bc-success") for x in ids), 3)

    def test_sampling_is_reproducible_and_independent_of_source_order(self):
        self.assertEqual(self.select(), self.select(list(reversed(self.items))))

    def test_short_pool_does_not_repeat_trials(self):
        items = [trial("one", True, "success"), trial("two", False, "success")]
        self.assertEqual(len(self.select(items)), 2)

    def test_duplicate_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            self.select(self.items + [self.items[0]])

    def test_zero_success_counts_preserve_only_failures(self):
        result = sample_hil_trials(self.items, hg_success_count=0, bc_success_count=0, seed=1)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(x["outcome"] == "failure" for x in result))
