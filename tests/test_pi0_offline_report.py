import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'pi0_report', Path(__file__).resolve().parents[1] / 'scripts/analyze_pi0_lora.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Pi0OfflineReportTests(unittest.TestCase):
    def test_physical_metrics_exclude_padding_and_keep_units(self):
        prediction = [[math.pi/180]*6+[.002], [999.]*7]
        record = {'prediction':prediction,'target':[[0.]*7,[0.]*7],
                  'valid':[True,False],'state':[0.]*7,
                  'normalized_abs_errors':[[.5]*7],'inference_ms':10.}
        result=module.summarize_predictions([record])
        self.assertEqual(result['valid_action_steps'],1)
        np.testing.assert_allclose(result['mae'],[1.]*6+[2.])
        self.assertEqual(result['normalized_l1'],.5)
        self.assertEqual(result['hold_mae'],[0.]*7)
        self.assertIsNone(result['horizon_mae']['19'])
        self.assertEqual(result['first_joint_gap_over_9deg_fraction'],0)


    def test_points_reject_nonfinite_and_sort(self):
        self.assertEqual(module.points([{'_step':3,'x':float('nan')},
                              {'_step':2,'x':.2},{'_step':1,'x':.3}],'x'),[(1,.3),(2,.2)])


    def test_svg_escapes_labels_and_handles_constant_series(self):
        svg=module.chart([('<script>',[(1,0.),(2,0.)])],'<title>')
        self.assertNotIn('<script>',svg)
        self.assertIn('&lt;script&gt;',svg)
        self.assertNotIn('nan',svg.lower())
