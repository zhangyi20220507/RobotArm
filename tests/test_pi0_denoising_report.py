import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('denoising_report',
    Path(__file__).resolve().parents[1] / 'scripts/compare_pi0_denoising.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DenoisingReportTests(unittest.TestCase):
    def record(self, steps=5):
        return {'prediction': [[math.pi / 180] * 6 + [.002], [999.] * 7],
                'target': [[0.] * 7] * 2, 'valid': [True, False],
                'state': [0.] * 7, 'steps': steps, 'index': 0, 'seed': 1000}

    def test_padding_excluded_and_units_preserved(self):
        result = module.summarize([self.record()])
        self.assertEqual(result['valid_steps'], 1)
        np.testing.assert_allclose(result['mae'], [1.] * 6 + [2.])
        self.assertEqual(result['either_prediction_bound_fraction'], 0)
        self.assertIsNone(module.summarize([self.record()], 1))

    def test_bounds_detect_both_sides(self):
        values = np.zeros((2, 7))
        values[0, 1:3] = np.deg2rad([-7.03, 4.13])
        values[1, 1:3] = np.deg2rad([183., -173.])
        self.assertEqual(module.audit(values)['outside_count'], [2, 2])

    def test_paired_zero_difference_and_missing_pair_rejected(self):
        records = [self.record(5), self.record(10)]
        result = module.paired_difference(records, 0)
        np.testing.assert_allclose(result['mae_difference_5_minus_10'], [0.] * 7)
        with self.assertRaises(ValueError): module.paired_difference(records[:1], 0)

    def test_cohorts_disjoint_and_near_starts_episode_diverse(self):
        cols = {'episode_index': np.repeat([0, 1, 2], 10),
                'frame_index': np.tile(np.arange(10), 3),
                'observation.state': np.zeros((30, 7))}
        result = module.select_indices(cols, 4, 3)
        self.assertEqual(len(set(i for i, _ in result)), len(result))
        near = [i for i, group in result if group == 'near_zero_start']
        self.assertEqual(len(near), 3)
        self.assertEqual(len(set(cols['episode_index'][i] for i in near)), 3)

    def test_html_escapes_cells(self):
        self.assertNotIn('<script>', module.table(['name'], [['<script>']]))


if __name__ == '__main__':
    unittest.main()
