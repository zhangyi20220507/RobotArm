"""Merge Piper raw sessions into another piper_raw_v1 session, not LeRobot.

Source data stays untouched. Videos/NPZ are copied byte-for-byte; only copied
episode.json indices and provenance change. Preview is the default.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from .convert import EpisodeSource, discover_episodes, read_json, validate_episode, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def episode_files(path: Path) -> list[Path]:
    result = []
    for item in sorted(path.rglob('*')):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise RuntimeError(f'Symlinks/special files are not supported: {item}')
        if item.is_file():
            result.append(item)
    return result


def original_source(episode: EpisodeSource) -> dict:
    # Preserve the leaf identity when merging a previously merged session.
    origin = episode.metadata.get('merge_source')
    if origin is not None:
        if not isinstance(origin, dict) or not isinstance(origin.get('episode_path'), str):
            raise RuntimeError(f'Invalid merge_source in {episode.path}')
        return copy.deepcopy(origin)
    return {'session_path': str(episode.session_path), 'episode_path': str(episode.path),
            'episode_index': episode.metadata.get('episode_index')}


def build_plan(sources: list[Path], output: Path, *, include_incomplete=False,
               allow_mixed_tasks=False) -> tuple[dict, list[EpisodeSource]]:
    sources = [p.expanduser().resolve() for p in sources]
    if not sources or len(sources) != len(set(sources)):
        raise RuntimeError('Provide distinct source sessions; duplicate paths are not allowed')
    # Reject even a dangling output symlink, before resolving it.
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f'Output already exists; choose a new directory: {output}')
    output = output.resolve()
    for source in sources:
        if source == output or source in output.parents or output in source.parents:
            raise RuntimeError(f'Output must be separate from every source: {source}')
        if (source / 'MERGE_INCOMPLETE.txt').exists():
            raise RuntimeError(f'Incomplete previous merge: {source}')
        if any((source / 'episodes').glob('*.in_progress')):
            raise RuntimeError(f'Stop recording and resolve in-progress episodes first: {source}')

    discovered = [discover_episodes(p, include_incomplete) for p in sources]
    base = discovered[0][0]
    roles = list(base['cameras'])
    episodes, entries, skipped, seen = [], [], [], set()
    for source, (session, selected, reasons) in zip(sources, discovered):
        if (source / 'session.json').is_symlink():
            raise RuntimeError(f'Symlink session metadata is not supported: {source}')
        for key in ['record_fps', 'state_names', 'state_units', 'action_semantics']:
            if session.get(key) != base.get(key):
                raise RuntimeError(f'{source}: incompatible {key}')
        if list(session['cameras']) != roles:
            raise RuntimeError(f'{source}: incompatible camera roles/order')
        for role in roles:
            for key in ['serial', 'stream', 'color_intrinsics']:
                if session['cameras'][role].get(key) != base['cameras'][role].get(key):
                    raise RuntimeError(f'{source}: incompatible camera {role} {key}')
        for key in ['model', 'joint_frame', 'tcp_pose_units']:
            if session.get('robot', {}).get(key) != base.get('robot', {}).get(key):
                raise RuntimeError(f'{source}: incompatible robot {key}')
        first = len(episodes)
        for episode in selected:
            if episode.path.is_symlink():
                raise RuntimeError(f'Symlink episode is not supported: {episode.path}')
            origin = original_source(episode)
            identity = str(Path(origin['episode_path']).expanduser().resolve())
            if identity in seen:
                raise RuntimeError(f'Duplicate original episode (possibly raw + merged input): {identity}')
            seen.add(identity)
            episode_files(episode.path)
            validate_episode(episode, session, roles)
            episodes.append(episode)
        entries.append({'path': str(source), 'metadata': session,
                        'session_sha256': sha256(source / 'session.json'),
                        'episodes': len(selected),
                        'frames': sum(int(e.metadata['frames']) for e in selected),
                        'first_output_episode': first, 'last_output_episode': len(episodes) - 1})
        skipped.extend({'source': str(source), 'reason': reason} for reason in reasons)
    tasks = sorted({str(e.metadata.get('task') or e.session.get('task') or '') for e in episodes})
    if len(tasks) > 1 and not allow_mixed_tasks:
        raise RuntimeError(f'Different tasks: {tasks}; use --allow-mixed-tasks only intentionally')
    return {'format_version': 'piper_raw_merge_v1', 'output': str(output),
            'source_sessions': entries, 'total_episodes': len(episodes),
            'total_frames': sum(int(e.metadata['frames']) for e in episodes),
            'tasks': tasks, 'camera_roles': roles, 'record_fps': base['record_fps'],
            'include_incomplete': include_incomplete, 'skipped': skipped,
            'estimated_copy_bytes': sum(p.stat().st_size for e in episodes for p in episode_files(e.path)),
            'viewpoint_verified': False}, episodes


def execute_plan(plan: dict, episodes: list[EpisodeSource]) -> Path:
    output = Path(plan['output'])
    # Recheck source metadata before creating any destination. No overwrite/append.
    for source in plan['source_sessions']:
        if sha256(Path(source['path']) / 'session.json') != source['session_sha256']:
            raise RuntimeError(f"Source changed since validation: {source['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < plan['estimated_copy_bytes'] + 1024 * 1024:
        raise RuntimeError('Insufficient free space to make independent copies')
    output.mkdir()  # Exclusive creation also protects against concurrent merges.
    marker = output / 'MERGE_INCOMPLETE.txt'
    marker.write_text('Merge incomplete. Do not convert/train from this directory. Originals are unchanged.\n')
    mapped = []
    try:
        for index, episode in enumerate(episodes):
            destination = output / 'episodes' / f'episode_{index:06d}'
            destination.mkdir(parents=True)
            hashes = {}
            files = episode_files(episode.path)
            if read_json(episode.path / 'episode.json') != episode.metadata:
                raise RuntimeError(f'Source episode changed: {episode.path}')
            for source_file in files:
                relative = source_file.relative_to(episode.path)
                before = sha256(source_file)
                hashes[str(relative)] = before
                if relative == Path('episode.json'):
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target)
                if sha256(target) != before or sha256(source_file) != before:
                    raise RuntimeError(f'Copy verification failed / source changed: {source_file}')
            if episode_files(episode.path) != files or read_json(episode.path / 'episode.json') != episode.metadata:
                raise RuntimeError(f'Source changed while copying: {episode.path}')
            origin = original_source(episode)
            metadata = {**episode.metadata, 'episode_index': index, 'merge_source': origin}
            write_json(destination / 'episode.json', metadata)
            validate_episode(EpisodeSource(destination, metadata, episode.session, output),
                             episode.session, plan['camera_roles'])
            mapped.append({'episode_index': index, 'directory': destination.name,
                           'input_episode_path': str(episode.path), 'original_source': origin,
                           'frames': metadata['frames'], 'source_file_sha256': hashes,
                           'output_episode_json_sha256': sha256(destination / 'episode.json')})
        for source in plan['source_sessions']:
            if sha256(Path(source['path']) / 'session.json') != source['session_sha256']:
                raise RuntimeError(f"Source changed while copying: {source['path']}")
        session = copy.deepcopy(plan['source_sessions'][0]['metadata'])
        now = datetime.now().astimezone().isoformat()
        session.update({'created_at': now, 'task': plan['tasks'][0] if len(plan['tasks']) == 1 else 'Mixed tasks; see episode.json',
                        'tasks': plan['tasks'], 'merged_source_sessions': plan['source_sessions'],
                        'merge_manifest': 'merge_manifest.json',
                        'setup_note': 'Merged raw sessions; consult per-source metadata for camera setup/robot configuration.'})
        manifest = {**plan, 'created_at': now, 'episodes': mapped,
                    'copy_mode': 'independent byte-verified copies; no video re-encoding'}
        write_json(output / 'merge_manifest.json', manifest)
        # Publish session metadata last: partial copies do not look like a usable raw session.
        write_json(output / 'session.json', session)
        marker.unlink()
    except BaseException:
        print(f'Merge failed; partial output retained at {output}. Originals are unchanged.')
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='NEW raw session directory')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Copy files after validation')
    mode.add_argument('--dry-run', action='store_true', help='Validate only (default)')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--allow-mixed-tasks', action='store_true')
    args = parser.parse_args(argv)
    plan, episodes = build_plan(args.sessions, args.output,
                                include_incomplete=args.include_incomplete,
                                allow_mixed_tasks=args.allow_mixed_tasks)
    print(f"Output (raw, NOT LeRobot): {plan['output']}")
    for source in plan['source_sessions']:
        print(f"  {Path(source['path']).name}: {source['episodes']} episodes / {source['frames']} frames "
              f"-> {source['first_output_episode']}..{source['last_output_episode']}")
    print(f"Total: {plan['total_episodes']} episodes, {plan['total_frames']} frames; "
          f"copy size ~{plan['estimated_copy_bytes'] / 1024**3:.3f} GiB; tasks={plan['tasks']}")
    for skipped in plan['skipped']:
        print(f"Skipping: {skipped['source']}: {skipped['reason']}")
    print('Videos/NPZ preserved; metadata compatibility does NOT prove camera viewpoint alignment.')
    print('Stop recording before merging. Validation checks arrays/video headers, not every decoded video frame.')
    if args.execute:
        execute_plan(plan, episodes)
        print(f"DONE: {plan['output']}; original sessions untouched.")
    else:
        print('DRY-RUN: no files written. Add --execute to create the merged raw session.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
