import unittest

import torch
from torch import nn

from robotarm.modeling.act_latent_report import latent_mode, l1_totals, sample_indices, summarize, render_report


class LatentReportTests(unittest.TestCase):
    def test_only_outer_branch_flag_changes_and_restores_on_error(self):
        policy = nn.Module()
        policy.model = nn.Sequential(nn.Dropout(.9), nn.BatchNorm1d(3))
        policy.train()
        policy.model[1].eval()
        previous = [m.training for m in policy.modules()]
        with self.assertRaisesRegex(RuntimeError, "test"):
            with latent_mode(policy, True):
                self.assertTrue(policy.model.training)
                self.assertFalse(policy.training)
                self.assertFalse(policy.model[0].training)
                self.assertFalse(policy.model[1].training)
                raise RuntimeError("test")
        self.assertEqual(previous, [m.training for m in policy.modules()])
        with latent_mode(policy, False):
            self.assertTrue(all(not m.training for m in policy.modules()))

    def test_padding_excluded_and_weighted_by_valid_elements(self):
        pred = torch.tensor([[[1., 3.], [float("nan"), 999.]], [[2., 4.], [4., 6.]]])
        target = torch.zeros_like(pred)
        total, count = l1_totals(pred, target, torch.tensor([[False, True], [False, False]]))
        self.assertEqual(count, 3)
        self.assertEqual(total.tolist(), [7., 13.])
        report = summarize([total, total * 2], count)
        self.assertAlmostEqual(report["l1"], 5.)
        self.assertEqual(report["valid_elements_per_repeat"], 6)
        self.assertGreater(report["repeat_std"], 0.)

    def test_invalid_valid_element_rejected(self):
        with self.assertRaises(ValueError):
            l1_totals(torch.tensor([[[float("nan")]]]), torch.zeros(1, 1, 1), torch.zeros(1, 1, dtype=torch.bool))

    def test_sampling(self):
        self.assertEqual(sample_indices(10, 3), [0, 3, 6])
        self.assertEqual(sample_indices(3, 0), [0, 1, 2])

    def test_report_escapes_metadata(self):
        metric = summarize([torch.tensor([1., 2.])], 1)
        split = dict(reconstruction=metric, zero=metric, episodes=[1], evaluated_frames=1, total_frames=2)
        report = dict(train=split, validation=split, checkpoint="<script>bad</script>", sampled=True, repeats=1, seed=0)
        page = render_report(report)
        self.assertNotIn("<script>", page)
        self.assertIn("抽样报告", page)
        self.assertIn("D−B", page)


if __name__ == "__main__":
    unittest.main()
