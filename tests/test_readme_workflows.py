"""Keep README command coverage and examples aligned without running hardware."""
import ast
import re
import shlex
import subprocess
import unittest
from pathlib import Path

from robotarm.cli import WORKFLOW_GROUPS


ROOT = Path(__file__).resolve().parents[1]


class ReadmeWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / 'docs/README.md').read_text(encoding='utf-8')
        cls.blocks = re.findall(r'```bash\n(.*?)```', cls.readme, flags=re.S)

    def test_catalog_covers_exactly_the_registered_workflows(self):
        registered = {(group, command) for group, (_, commands) in WORKFLOW_GROUPS.items()
                      for command in commands}
        documented = set(re.findall(r'^\| `robotarm ([\w-]+) ([\w-]+)`', self.readme, re.M))
        self.assertEqual(documented, registered)

    def test_bash_examples_have_valid_syntax_without_execution(self):
        for index, block in enumerate(self.blocks):
            with self.subTest(block=index):
                result = subprocess.run(['bash', '-n'], input=block, text=True,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_command_examples_use_existing_workflows(self):
        for block in self.blocks:
            for line in block.replace('\\\n', ' ').splitlines():
                if not line.startswith('robotarm '):
                    continue
                args = shlex.split(line, comments=True)
                if len(args) >= 3 and not args[2].startswith('-'):
                    self.assertIn(args[1], WORKFLOW_GROUPS)
                    self.assertIn(args[2], WORKFLOW_GROUPS[args[1]][1])

    def test_local_file_links_and_navigation_anchors_exist(self):
        for source in (ROOT / 'docs/README.md', ROOT / 'docs/BestPractice_and_LessonLearn.md',
                       ROOT / 'tests/README.md'):
            prose = re.sub(r'```.*?```', '', source.read_text(encoding='utf-8'), flags=re.S)
            self.assertFalse(re.search(r'^#{7,} ', prose, re.M), source.name)
            for target in re.findall(r'\]\(([^)]+)\)', prose):
                if target.startswith(('https://', 'http://')):
                    continue
                path, _, fragment = target.partition('#')
                destination = source.parent / path if path else source
                with self.subTest(source=source.name, target=target):
                    self.assertTrue(destination.is_file(), target)
                    if fragment and destination.suffix == '.md':
                        contents = re.sub(r'```.*?```', '', destination.read_text(encoding='utf-8'), flags=re.S)
                        anchors = {re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
                                   for title in re.findall(r'^#+ (.+)$', contents, re.M)}
                        self.assertIn(fragment, anchors, target)

    def test_documented_flags_match_static_parsers_for_common_offline_tools(self):
        modules = {
            ('simulation', 'realsense-stream'): 'visualization/realsense_stream.py',
            ('data-acquisition', 'merge-piper-raw'): 'record/merge.py',
            ('data-acquisition', 'convert-piper-raw'): 'record/convert.py',
            ('modeling', 'evaluate'): 'modeling/evaluate.py',
            ('modeling', 'validate-runtime'): 'modeling/validate_runtime.py',
            ('modeling', 'train-pi0-lora'): 'modeling/pi0_lora.py',
            ('modeling', 'check-pi0-lora'): 'modeling/pi0_lora_check.py',
            ('modeling', 'pi0-training-report'): 'modeling/reports/pi0_training.py',
            ('modeling', 'pi0-denoising-report'): 'modeling/reports/pi0_denoising.py',
            ('modeling', 'act-training-report'): 'modeling/reports/act_training.py',
        }
        flags = {}
        for key, relative in modules.items():
            tree = ast.parse((ROOT / 'src/robotarm' / relative).read_text())
            flags[key] = {'--help'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                    flags[key].update(a.value for a in node.args if isinstance(a, ast.Constant)
                                      and isinstance(a.value, str) and a.value.startswith('--'))
        for block in self.blocks:
            for line in block.replace('\\\n', ' ').splitlines():
                if not line.startswith('robotarm '):
                    continue
                args = shlex.split(line, comments=True)
                key = tuple(args[1:3])
                if key in flags:
                    actual = {a.split('=')[0] for a in args[3:] if a.startswith('--') and a != '--'}
                    self.assertFalse(actual - flags[key], (key, actual - flags[key]))


if __name__ == '__main__':
    unittest.main()
