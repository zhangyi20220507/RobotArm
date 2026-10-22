"""Moved report implementations keep CLI and historical script entrypoints."""
import subprocess
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotarm.cli import build_parser, _strip_remainder


class ReportEntrypointTests(unittest.TestCase):
    def test_act_summary_uses_supplied_steps_and_sample_counts(self):
        from robotarm.modeling.reports.act_training import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                'dataset': {'root': 'data/example', 'image_transforms': {'enable': False}},
                'policy': {'type': 'act', 'chunk_size': 30, 'n_action_steps': 10, 'kl_weight': 10},
                'batch_size': 4, 'seed': 42, 'optimizer': {'lr': 1e-5}, 'scheduler': None,
                'eval_steps': 1000, 'save_freq': 2000, 'max_eval_samples': 0,
            }
            paths, history = [], []
            for step in [2000, 4000]:
                cp = root / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
                cp.mkdir(parents=True)
                (cp / 'config.json').write_text('{}')
                (cp / 'train_config.json').write_text(json.dumps(config))
                report = {'checkpoint': str(cp), 'seed': 42, 'batch_size': 2,
                          'repeats': 3, 'chunk_size': 30}
                for split, episode in [('train', 0), ('validation', 1)]:
                    report[split] = {'episodes': [episode], 'sample_indices': [0, 1],
                                     'total_frames': 10, 'evaluated_frames': 2,
                                     'reconstruction': {'l1': .1},
                                     'zero': {'l1': .1, 'per_dimension_l1': [.1] * 7}}
                path = root / f'latent_{step}.json'
                path.write_text(json.dumps(report))
                paths.append(str(path))
                history.append({'_step': step, 'train/l1_loss': .1, 'train/loss': .12,
                                'eval/eval_loss': .2 if step == 2000 else .15})
            output = root / 'summary.html'
            with patch('robotarm.modeling.reports.act_training.read_run_history',
                       return_value=(history, [{'path': 'test-history'}])):
                main(['--run-dir', str(root), '--latent-reports', *paths, '--output', str(output)])
            page = output.read_text()
            self.assertIn('2000 → 4000步', page)
            self.assertIn('评估2/2个相同起点', page)
            self.assertIn('seed=42', page)
            self.assertNotIn('90000', page)
            self.assertEqual(json.loads(output.with_suffix('.json').read_text())['best_saved_step'], 4000)

    def test_unified_commands_forward_arguments(self):
        for command in ['pi0-training-report', 'pi0-denoising-report', 'act-training-report']:
            args = build_parser().parse_args(['modeling', command, '--', '--help'])
            self.assertEqual(_strip_remainder(args.extra_args), ['--help'])
            self.assertTrue(callable(args.handler))

    def test_new_modules_and_compatibility_scripts_have_hardware_free_help(self):
        root = Path(__file__).resolve().parents[1]
        for module, script in [('pi0_training', 'analyze_pi0_lora'),
                               ('pi0_denoising', 'compare_pi0_denoising'),
                               ('act_training', 'analyze_act_resume')]:
            for argv in [[sys.executable, '-m', f'robotarm.modeling.reports.{module}', '--help'],
                         [sys.executable, str(root / 'scripts' / f'{script}.py'), '--help']]:
                with self.subTest(argv=argv):
                    result = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=root)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--output', result.stdout)
