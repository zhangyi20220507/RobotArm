"""Raw merge tests with tiny real NPZ/video fixtures; no robot/LeRobot dependency."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from robotarm.record.convert import discover_episodes, read_json, write_json
from robotarm.record.merge import build_plan, execute_plan, main, sha256


class RawMergeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a = self.source('a', [2, 5])
        self.b = self.source('b', [0])
        self.output = self.root / 'merged'

    def source(self, name, indices):
        root = self.root / name
        session = {'format_version': 'piper_raw_v1', 'record_fps': 20,
                   'task': 'Pick cup', 'setup_id': name,
                   'state_names': [f'j{i}' for i in range(6)] + ['gripper'],
                   'state_units': ['rad'] * 6 + ['m'],
                   'action_semantics': 'commanded MOVE_JS joint targets and gripper target',
                   'cameras': {r: {'serial': r, 'stream': {'width': 16, 'height': 16}}
                               for r in ['global', 'wrist']}}
        write_json(root / 'session.json', session)
        for index in indices:
            episode = root / 'episodes' / f'episode_{index:06d}'
            write_json(episode / 'episode.json', {'episode_index': index, 'frames': 2,
                       'status': 'complete', 'task': 'Pick cup', 'camera_roles': ['global', 'wrist']})
            arrays = {'timestamp': np.arange(2) / 20}
            for key, width in {'observation_state': 7, 'action': 7, 'observation_tcp_pose': 6,
                               'teleop_delta': 6, 'camera_host_timestamp': 2,
                               'camera_sensor_timestamp': 2, 'camera_frame_number': 2,
                               'camera_frame_age': 2, 'camera_skew': None}.items():
                arrays[key] = np.zeros((2,) if width is None else (2, width))
            np.savez(episode / 'frames.npz', **arrays)
            for role in ['global', 'wrist']:
                writer = cv2.VideoWriter(str(episode / f'{role}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), 20, (16, 16))
                self.assertTrue(writer.isOpened())
                try:
                    for _ in range(2):
                        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
                finally:
                    writer.release()
        return root

    def test_raw_merge_renumbers_and_preserves_all_payloads_and_sources(self):
        before = {p: sha256(p) for root in [self.a, self.b] for p in root.rglob('*') if p.is_file()}
        plan, episodes = build_plan([self.a, self.b], self.output)
        self.assertEqual((plan['total_episodes'], plan['total_frames']), (3, 6))
        execute_plan(plan, episodes)
        session, merged, skipped = discover_episodes(self.output, False)
        self.assertEqual(len(merged), 3)
        self.assertFalse(skipped)
        self.assertEqual(session['format_version'], 'piper_raw_v1')
        self.assertFalse((self.output / 'MERGE_INCOMPLETE.txt').exists())
        for index, (original, result) in enumerate(zip(episodes, merged)):
            self.assertEqual(result.path.name, f'episode_{index:06d}')
            self.assertEqual(result.metadata['episode_index'], index)
            self.assertEqual(result.metadata['merge_source']['episode_path'], str(original.path))
            for name in ['frames.npz', 'global.mp4', 'wrist.mp4']:
                self.assertEqual(sha256(original.path / name), sha256(result.path / name))
        for path, digest in before.items():
            self.assertEqual(sha256(path), digest)
        manifest = read_json(self.output / 'merge_manifest.json')
        self.assertEqual(manifest['source_sessions'][1]['metadata']['setup_id'], 'b')
        self.assertEqual(len(manifest['episodes']), 3)

    def test_preview_does_not_write(self):
        self.assertEqual(main([str(self.a), str(self.b), '--output', str(self.output)]), 0)
        self.assertFalse(self.output.exists())

    def test_duplicates_rejected_including_symlink_alias(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.a, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            build_plan([self.a, alias], self.output)

    def test_output_inside_second_source_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'separate'):
            build_plan([self.a, self.b], self.b / 'merged')

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            build_plan([self.a], self.output)
        self.assertEqual((self.output / 'keep').read_text(), 'keep')

    def test_incompatible_camera_rejected(self):
        session = read_json(self.b / 'session.json')
        session['cameras']['global']['serial'] = 'different'
        write_json(self.b / 'session.json', session)
        with self.assertRaisesRegex(RuntimeError, 'incompatible camera'):
            build_plan([self.a, self.b], self.output)

    def test_mixed_tasks_need_explicit_option_and_keep_labels(self):
        path = self.b / 'episodes/episode_000000/episode.json'
        metadata = read_json(path)
        metadata['task'] = 'Place cup'
        write_json(path, metadata)
        with self.assertRaisesRegex(RuntimeError, 'Different tasks'):
            build_plan([self.a, self.b], self.output)
        plan, episodes = build_plan([self.a, self.b], self.output, allow_mixed_tasks=True)
        self.assertEqual(plan['tasks'], ['Pick cup', 'Place cup'])
        self.assertEqual(episodes[-1].metadata['task'], 'Place cup')

    def test_incomplete_skipped_unless_explicit(self):
        path = self.a / 'episodes/episode_000005/episode.json'
        metadata = read_json(path)
        metadata['status'] = 'incomplete'
        write_json(path, metadata)
        plan, _ = build_plan([self.a], self.output)
        self.assertEqual(plan['total_episodes'], 1)
        self.assertEqual(len(plan['skipped']), 1)
        plan, _ = build_plan([self.a], self.output, include_incomplete=True)
        self.assertEqual(plan['total_episodes'], 2)

    def test_active_recording_rejected(self):
        (self.a / 'episodes/.episode_000006.in_progress').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'Stop recording'):
            build_plan([self.a], self.output)

    def test_payload_symlink_rejected(self):
        (self.a / 'episodes/episode_000002/link').symlink_to(self.b / 'session.json')
        with self.assertRaisesRegex(RuntimeError, 'Symlinks'):
            build_plan([self.a], self.output)

    def test_original_plus_merged_duplicate_is_rejected(self):
        plan, episodes = build_plan([self.a], self.output)
        execute_plan(plan, episodes)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate original episode'):
            build_plan([self.a, self.output], self.root / 'another')

    def test_failed_copy_keeps_marker_and_no_published_session(self):
        plan, episodes = build_plan([self.a], self.output)
        with patch('robotarm.record.merge.shutil.copy2', side_effect=OSError('copy failed')):
            with self.assertRaisesRegex(OSError, 'copy failed'):
                execute_plan(plan, episodes)
        self.assertTrue((self.output / 'MERGE_INCOMPLETE.txt').is_file())
        self.assertFalse((self.output / 'session.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'Incomplete previous merge'):
            build_plan([self.output], self.root / 'another')

    def test_changed_session_is_rejected_before_output_created(self):
        plan, episodes = build_plan([self.a], self.output)
        session = read_json(self.a / 'session.json')
        session['task'] = 'changed'
        write_json(self.a / 'session.json', session)
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            execute_plan(plan, episodes)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
