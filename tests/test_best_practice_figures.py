"""Documentation charts keep measurements raw and remain hardware-free."""
import subprocess
import sys
import unittest

from robotarm.modeling.reports.best_practice_figures import history_points


class BestPracticeFigureTests(unittest.TestCase):
    def test_points_keep_original_values_and_skip_missing_nonfinite(self):
        rows = [{'_step': 400, 'loss': .18}, {'_step': 200, 'loss': .3},
                {'_step': 300}, {'loss': .2}, {'_step': 500, 'loss': float('nan')},
                {'_step': float('inf'), 'loss': .1}]
        self.assertEqual(history_points(rows, 'loss'), [(200, .3), (400, .18)])
        self.assertEqual(history_points(rows, 'loss', 300), [(400, .18)])

    def test_help_does_not_need_source_reports_or_hardware(self):
        result = subprocess.run(
            [sys.executable, '-m', 'robotarm.modeling.reports.best_practice_figures', '--help'],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--output-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
