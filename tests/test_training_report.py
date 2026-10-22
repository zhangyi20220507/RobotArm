"""Offline history merge and exact saved-checkpoint selection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotarm.modeling.training_report import read_run_history, summarize_history


class TrainingReportTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)

    def file(self, name, data):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
        return p

    def test_resume_merge_and_copied_latest_run_are_deduplicated(self):
        first = self.file('wandb/offline-1/run-a.wandb', 'first')
        second = self.file('wandb/offline-2/run-a.wandb', 'second')
        self.file('wandb/latest-run/run-a.wandb', 'second')
        def read(path):
            if path.read_text() == 'first':
                return [{'_step': 40000, 'train/loss': .1, '_timestamp': 1}]
            return [{'_step': 60000, 'train/loss': .05, '_timestamp': 2},
                    {'_step': 60000, 'eval/eval_loss': .18, '_timestamp': 2}]
        with patch('robotarm.modeling.training_report.read_history', side_effect=read) as mock:
            rows, sources = read_run_history(self.root)
        self.assertEqual(mock.call_count, 2)
        self.assertEqual([r['_step'] for r in rows], [40000, 60000])
        self.assertEqual(rows[-1]['eval/eval_loss'], .18)
        self.assertEqual(rows[-1]['train/loss'], .05)
        self.assertEqual(len(sources), 2)

    def test_newer_record_wins_on_overlapping_step(self):
        self.file('wandb/1/run-a.wandb', 'new')
        self.file('wandb/2/run-a.wandb', 'old')
        def read(path):
            new = path.read_text() == 'new'
            return [{'_step': 2, '_timestamp': 20 if new else 10, 'train/loss': .1 if new else .9}]
        with patch('robotarm.modeling.training_report.read_history', side_effect=read):
            rows, _ = read_run_history(self.root)
        self.assertEqual(rows[0]['train/loss'], .1)

    def test_best_saved_is_not_checkpoint_nearest_best_unsaved_step(self):
        for step in [86000, 88000, 90000]:
            self.file(f'checkpoints/{step:06d}/pretrained_model/config.json', '{}')
        rows = [{'_step': s, 'eval/eval_loss': loss} for s, loss in
                [(86000, .19), (87000, .17), (88000, .20), (90000, .18)]]
        rows += [{'_step': 90200, 'train/loss': .04}]
        summary = summarize_history(rows, self.root)
        self.assertEqual(summary['best_eval_step'], 87000)
        self.assertEqual(summary['recommended_checkpoint_step'], 90000)
        self.assertEqual(summary['recommended_checkpoint_eval_loss'], .18)
        self.assertEqual(summary['final_eval_step'], 90000)
        self.assertEqual(summary['final_step'], 90200)
        self.assertIsNone(summary['final_train_l1_loss'])

    def test_no_saved_evaluated_checkpoint_gives_no_recommendation(self):
        summary = summarize_history([{'_step': 1, 'eval/eval_loss': .2}], self.root)
        self.assertIsNone(summary['recommended_checkpoint'])

    def test_missing_history_and_missing_eval_are_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, 'no history'):
            read_run_history(self.root)
        with self.assertRaisesRegex(ValueError, 'no finite validation'):
            summarize_history([{'_step': 1, 'train/loss': .2}], self.root)
