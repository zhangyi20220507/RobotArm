"""Merge Piper raw sessions into a new or rolling piper_raw_v1 dataset, not LeRobot.

Input sessions stay untouched. New videos/NPZ are copied byte-for-byte; only copied
episode.json indices and provenance change. ``--append-to`` retains existing
payloads in place and publishes new metadata transactionally. Preview is the default.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .convert import EpisodeSource, discover_episodes, read_json, validate_episode, write_json
from .file_copy import clone_or_copy
from .raw_reviews import FORMAT, default_raw_reviews_path, episode_identity, load_raw_reviews


APPEND_MARKER = 'APPEND_INCOMPLETE.json'
APPEND_STAGING_PREFIX = '.raw_append_'


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
    merged_reviews, episode_ids = {}, []
    for source, (session, selected, reasons) in zip(sources, discovered):
        reviews_path = default_raw_reviews_path(source)
        source_reviews = load_raw_reviews(source, selected)
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
        for source_index, episode in enumerate(selected):
            if episode.path.is_symlink():
                raise RuntimeError(f'Symlink episode is not supported: {episode.path}')
            origin = original_source(episode)
            identity = str(Path(origin['episode_path']).expanduser().resolve())
            if identity in seen:
                raise RuntimeError(f'Duplicate original episode (possibly raw + merged input): {identity}')
            seen.add(identity)
            episode_files(episode.path)
            validate_episode(episode, session, roles)
            if str(source_index) in source_reviews:
                merged_reviews[str(len(episodes))] = copy.deepcopy(source_reviews[str(source_index)])
            episode_ids.append(episode_identity(episode))
            episodes.append(episode)
        entries.append({'path': str(source), 'metadata': session,
                        'session_sha256': sha256(source / 'session.json'),
                        'reviews_sha256': sha256(reviews_path) if reviews_path.exists() else None,
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
            'reviews': merged_reviews, 'episode_ids': episode_ids,
            'estimated_copy_bytes': sum(p.stat().st_size for e in episodes for p in episode_files(e.path)),
            'viewpoint_verified': False}, episodes


def execute_plan(plan: dict, episodes: list[EpisodeSource]) -> Path:
    output = Path(plan['output'])
    # Recheck source metadata before creating any destination. No overwrite/append.
    for source in plan['source_sessions']:
        if sha256(Path(source['path']) / 'session.json') != source['session_sha256']:
            raise RuntimeError(f"Source changed since validation: {source['path']}")
        review_path = default_raw_reviews_path(Path(source['path']))
        if (sha256(review_path) if review_path.exists() else None) != source.get('reviews_sha256'):
            raise RuntimeError(f"Source reviews changed since validation: {source['path']}")
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
                clone_or_copy(str(source_file), str(target))
                if sha256(target) != before or sha256(source_file) != before:
                    raise RuntimeError(f'Copy verification failed / source changed: {source_file}')
            if episode_files(episode.path) != files or read_json(episode.path / 'episode.json') != episode.metadata:
                raise RuntimeError(f'Source changed while copying: {episode.path}')
            origin = original_source(episode)
            metadata = {**episode.metadata, 'episode_index': index, 'merge_source': origin}
            write_json(destination / 'episode.json', metadata)
            copied = EpisodeSource(destination, metadata, episode.session, output)
            if episode_identity(copied) != plan['episode_ids'][index]:
                raise RuntimeError(f'Source content changed since validation: {episode.path}')
            validate_episode(copied, episode.session, plan['camera_roles'])
            mapped.append({'episode_index': index, 'directory': destination.name,
                           'input_episode_path': str(episode.path), 'original_source': origin,
                           'frames': metadata['frames'], 'source_file_sha256': hashes,
                           'output_episode_json_sha256': sha256(destination / 'episode.json')})
        for source in plan['source_sessions']:
            if sha256(Path(source['path']) / 'session.json') != source['session_sha256']:
                raise RuntimeError(f"Source changed while copying: {source['path']}")
            review_path = default_raw_reviews_path(Path(source['path']))
            if (sha256(review_path) if review_path.exists() else None) != source.get('reviews_sha256'):
                raise RuntimeError(f"Source reviews changed while copying: {source['path']}")
        session = copy.deepcopy(plan['source_sessions'][0]['metadata'])
        now = datetime.now().astimezone().isoformat()
        session.update({'created_at': now, 'task': plan['tasks'][0] if len(plan['tasks']) == 1 else 'Mixed tasks; see episode.json',
                        'tasks': plan['tasks'], 'merged_source_sessions': plan['source_sessions'],
                        'merge_manifest': 'merge_manifest.json',
                        'setup_note': 'Merged raw sessions; consult per-source metadata for camera setup/robot configuration.'})
        manifest = {**plan, 'created_at': now, 'episodes': mapped,
                    'copy_mode': 'independent byte-verified copies; no video re-encoding'}
        write_json(output / 'merge_manifest.json', manifest)
        if plan['reviews']:
            write_json(default_raw_reviews_path(output), {
                'format_version': FORMAT, 'dataset_root': str(output),
                'episode_ids': plan['episode_ids'], 'episodes': plan['reviews'],
            })
        # Publish session metadata last: partial copies do not look like a usable raw session.
        write_json(output / 'session.json', session)
        marker.unlink()
    except BaseException:
        print(f'Merge failed; partial output retained at {output}. Originals are unchanged.')
        raise
    return output


def _optional_sha256(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def _check_contiguous_append_base(root: Path, episodes: list[EpisodeSource], skipped: list[str]) -> None:
    if skipped:
        raise RuntimeError(
            f'Append target contains skipped/incomplete episodes: {skipped}; '
            'clean or merge it into a canonical raw dataset first'
        )
    expected = [root / 'episodes' / f'episode_{index:06d}' for index in range(len(episodes))]
    actual = [episode.path for episode in episodes]
    if actual != expected or any(
        episode.metadata.get('episode_index') != index
        for index, episode in enumerate(episodes)
    ):
        raise RuntimeError(
            'Append target episodes must already have contiguous directories and episode_index values'
        )


def build_append_plan(
    sources: list[Path], target: Path, *, allow_mixed_tasks: bool = False,
) -> tuple[dict, list[EpisodeSource]]:
    """Validate a rolling raw master and plan copying only newly supplied episodes."""
    target = target.expanduser().resolve()
    sources = [source.expanduser().resolve() for source in sources]
    if not sources:
        raise RuntimeError('Provide at least one new raw session to append')
    if not target.is_dir():
        raise RuntimeError(f'Append target does not exist: {target}')
    if (target / APPEND_MARKER).exists():
        raise RuntimeError(
            f'Incomplete append at {target}; run with --append-to {target} '
            '--recover-append --execute first'
        )
    base_session, base_episodes, skipped = discover_episodes(target, False)
    _check_contiguous_append_base(target, base_episodes, skipped)
    # build_plan supplies all compatibility, duplicate-origin, review-binding and payload checks.
    dummy_output = target.parent / f'.{target.name}.append-plan-{uuid.uuid4().hex}'
    combined, all_episodes = build_plan(
        [target, *sources], dummy_output, allow_mixed_tasks=allow_mixed_tasks,
    )
    base_count = len(base_episodes)
    if [episode.path for episode in all_episodes[:base_count]] != [
        episode.path for episode in base_episodes
    ]:
        raise RuntimeError('Append target changed while building the plan')
    new_episodes = all_episodes[base_count:]
    if not new_episodes:
        raise RuntimeError('No eligible new episodes to append')
    manifest_path = target / 'merge_manifest.json'
    base_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    prior_records = base_manifest.get('episodes')
    if not isinstance(prior_records, list) or len(prior_records) != base_count:
        prior_records = [
            {
                'episode_index': index,
                'directory': episode.path.name,
                'input_episode_path': str(episode.path),
                'original_source': original_source(episode),
                'frames': episode.metadata['frames'],
                'retained_from_base': True,
            }
            for index, episode in enumerate(base_episodes)
        ]
    new_copy_bytes = sum(
        path.stat().st_size for episode in new_episodes for path in episode_files(episode.path)
    )
    combined.update({
        'mode': 'transactional_append',
        'output': str(target),
        'append_target': str(target),
        'base_episode_count': base_count,
        'new_episode_count': len(new_episodes),
        'estimated_copy_bytes': new_copy_bytes,
        'base_session': base_session,
        'base_manifest': base_manifest,
        'base_episode_records': prior_records,
        'base_manifest_sha256': _optional_sha256(manifest_path),
        'base_reviews_existed': default_raw_reviews_path(target).is_file(),
    })
    return combined, new_episodes


def _recheck_append_sources(plan: dict) -> None:
    for source in plan['source_sessions']:
        root = Path(source['path'])
        if sha256(root / 'session.json') != source['session_sha256']:
            raise RuntimeError(f"Source changed since append validation: {root}")
        reviews = default_raw_reviews_path(root)
        if _optional_sha256(reviews) != source.get('reviews_sha256'):
            raise RuntimeError(f"Source reviews changed since append validation: {root}")
    target_manifest = Path(plan['append_target']) / 'merge_manifest.json'
    if _optional_sha256(target_manifest) != plan.get('base_manifest_sha256'):
        raise RuntimeError('Append target manifest changed since validation')


def _write_marker(path: Path, document: dict, *, exclusive: bool = False) -> None:
    payload = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload, encoding='utf-8')
    os.replace(temporary, path)


def _validated_staging(target: Path, document: dict) -> Path:
    staging = Path(document.get('staging', '')).expanduser().resolve()
    if staging.parent != target or not staging.name.startswith(APPEND_STAGING_PREFIX):
        raise RuntimeError(f'Invalid append staging path in {target / APPEND_MARKER}')
    return staging


def rollback_append(target: Path) -> dict:
    """Restore the pre-append master described by its durable transaction marker."""
    target = target.expanduser().resolve()
    marker = target / APPEND_MARKER
    if not marker.is_file():
        raise RuntimeError(f'No incomplete transactional append at {target}')
    document = read_json(marker)
    if Path(document.get('target', '')).expanduser().resolve() != target:
        raise RuntimeError(f'Append marker belongs to another dataset: {marker}')
    staging = _validated_staging(target, document)
    base_count = document.get('base_episode_count')
    names = document.get('appended_episode_names')
    if type(base_count) is not int or base_count < 0 or not isinstance(names, list):
        raise RuntimeError(f'Invalid append recovery metadata: {marker}')
    for offset, name in enumerate(names):
        if name != f'episode_{base_count + offset:06d}':
            raise RuntimeError(f'Unsafe episode name in append recovery metadata: {name!r}')
    if document.get('commit_started'):
        metadata = document.get('metadata_before')
        if not isinstance(metadata, dict) or not staging.is_dir():
            raise RuntimeError('Append commit started but its recovery backup is missing')
        for relative, existed in metadata.items():
            if relative not in {'session.json', 'merge_manifest.json', 'annotations/reviews.json'}:
                raise RuntimeError(f'Unsafe metadata path in append recovery: {relative!r}')
            destination = target / relative
            backup = staging / 'backup' / relative
            if existed:
                if not backup.is_file():
                    raise RuntimeError(f'Missing append recovery backup: {backup}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + '.rollback')
                shutil.copy2(backup, temporary)
                os.replace(temporary, destination)
            elif destination.exists():
                destination.unlink()
    for name in names:
        destination = target / 'episodes' / name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            raise RuntimeError(f'Cannot safely remove non-directory appended path: {destination}')
    marker.unlink()
    if staging.is_dir():
        shutil.rmtree(staging)
    return {'target': str(target), 'removed_episodes': len(names),
            'transaction_id': document.get('transaction_id')}


def _copy_append_episode(
    episode: EpisodeSource, destination: Path, output_index: int,
    expected_id: str, camera_roles: list[str], target: Path,
) -> dict:
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
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(str(source_file), str(copied))
        if sha256(copied) != before or sha256(source_file) != before:
            raise RuntimeError(f'Copy verification failed / source changed: {source_file}')
    if episode_files(episode.path) != files or read_json(episode.path / 'episode.json') != episode.metadata:
        raise RuntimeError(f'Source changed while copying: {episode.path}')
    origin = original_source(episode)
    metadata = {**episode.metadata, 'episode_index': output_index, 'merge_source': origin}
    write_json(destination / 'episode.json', metadata)
    copied_episode = EpisodeSource(destination, metadata, episode.session, target)
    if episode_identity(copied_episode) != expected_id:
        raise RuntimeError(f'Source content changed since validation: {episode.path}')
    validate_episode(copied_episode, episode.session, camera_roles)
    return {'episode_index': output_index, 'directory': destination.name,
            'input_episode_path': str(episode.path), 'original_source': origin,
            'frames': metadata['frames'], 'source_file_sha256': hashes,
            'output_episode_json_sha256': sha256(destination / 'episode.json')}


def execute_append_plan(plan: dict, new_episodes: list[EpisodeSource]) -> Path:
    """Append only new payloads, publishing metadata last with rollback support."""
    target = Path(plan['append_target']).resolve()
    marker = target / APPEND_MARKER
    _recheck_append_sources(plan)
    required = plan['estimated_copy_bytes'] + 1024 * 1024
    if shutil.disk_usage(target).free < required:
        raise RuntimeError('Insufficient free space for new append payloads')
    transaction_id = uuid.uuid4().hex
    staging = target / f'{APPEND_STAGING_PREFIX}{transaction_id}'
    staging.mkdir()
    base_count = plan['base_episode_count']
    names = [f'episode_{base_count + offset:06d}' for offset in range(len(new_episodes))]
    metadata_relatives = ['session.json', 'merge_manifest.json', 'annotations/reviews.json']
    document = {
        'format_version': 'piper_raw_append_transaction_v1',
        'transaction_id': transaction_id,
        'target': str(target),
        'staging': str(staging),
        'base_episode_count': base_count,
        'appended_episode_names': names,
        'metadata_before': {relative: (target / relative).is_file()
                            for relative in metadata_relatives},
        'commit_started': False,
    }
    try:
        _write_marker(marker, document, exclusive=True)
        # The marker now blocks conversion/browsing writes. Recheck before taking backups.
        _recheck_append_sources(plan)
        for relative, existed in document['metadata_before'].items():
            if existed:
                backup = staging / 'backup' / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target / relative, backup)
        copied_records = []
        for offset, episode in enumerate(new_episodes):
            output_index = base_count + offset
            copied_records.append(_copy_append_episode(
                episode, staging / 'episodes' / names[offset], output_index,
                plan['episode_ids'][output_index], plan['camera_roles'], target,
            ))

        now = datetime.now().astimezone().isoformat()
        session = copy.deepcopy(plan['base_session'])
        prior_sources = session.get('merged_source_sessions')
        if not isinstance(prior_sources, list):
            prior_sources = [{
                'path': str(target), 'metadata': copy.deepcopy(plan['base_session']),
                'session_sha256': plan['source_sessions'][0]['session_sha256'],
                'episodes': base_count,
                'frames': sum(record['frames'] for record in plan['base_episode_records']),
            }]
        new_sources = copy.deepcopy(plan['source_sessions'][1:])
        session.update({
            'updated_at': now,
            'task': plan['tasks'][0] if len(plan['tasks']) == 1 else 'Mixed tasks; see episode.json',
            'tasks': plan['tasks'],
            'merged_source_sessions': [*prior_sources, *new_sources],
            'merge_manifest': 'merge_manifest.json',
            'append_generation': int(session.get('append_generation', 0)) + 1,
            'setup_note': 'Rolling merged raw dataset; consult merge_manifest.json for provenance.',
        })
        old_manifest = plan['base_manifest']
        manifest_sources = old_manifest.get('source_sessions')
        if not isinstance(manifest_sources, list):
            manifest_sources = copy.deepcopy(prior_sources)
        history = old_manifest.get('append_history', [])
        if not isinstance(history, list):
            history = []
        manifest = {
            'format_version': 'piper_raw_merge_v2',
            'output': str(target),
            'created_at': old_manifest.get('created_at', plan['base_session'].get('created_at')),
            'updated_at': now,
            'source_sessions': [*manifest_sources, *new_sources],
            'total_episodes': base_count + len(new_episodes),
            'total_frames': plan['total_frames'],
            'tasks': plan['tasks'],
            'camera_roles': plan['camera_roles'],
            'record_fps': plan['record_fps'],
            'reviews': plan['reviews'],
            'episode_ids': plan['episode_ids'],
            'episodes': [*copy.deepcopy(plan['base_episode_records']), *copied_records],
            'append_history': [*history, {
                'transaction_id': transaction_id, 'appended_at': now,
                'first_episode': base_count,
                'episodes': len(new_episodes),
                'frames': sum(record['frames'] for record in copied_records),
                'sources': [entry['path'] for entry in new_sources],
            }],
            'copy_mode': ('transactional append; existing payload retained; '
                          'new payload byte-verified; no video re-encoding'),
        }
        staged_metadata = staging / 'metadata'
        write_json(staged_metadata / 'session.json', session)
        write_json(staged_metadata / 'merge_manifest.json', manifest)
        if plan['base_reviews_existed'] or plan['reviews']:
            write_json(staged_metadata / 'annotations/reviews.json', {
                'format_version': FORMAT, 'dataset_root': str(target),
                'episode_ids': plan['episode_ids'], 'episodes': plan['reviews'],
            })

        _recheck_append_sources(plan)
        document['commit_started'] = True
        _write_marker(marker, document)
        for name in names:
            destination = target / 'episodes' / name
            if destination.exists():
                raise RuntimeError(f'Append destination appeared during transaction: {destination}')
            os.replace(staging / 'episodes' / name, destination)
        for relative in ['annotations/reviews.json', 'merge_manifest.json', 'session.json']:
            source = staged_metadata / relative
            if source.is_file():
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
        if read_json(target / 'session.json').get('append_generation') != session['append_generation']:
            raise RuntimeError('Append session metadata read-back failed')
        if len(read_json(target / 'merge_manifest.json').get('episodes', [])) != manifest['total_episodes']:
            raise RuntimeError('Append manifest read-back failed')
        if (plan['base_reviews_existed'] or plan['reviews']):
            reviews = read_json(default_raw_reviews_path(target))
            if reviews.get('episode_ids') != plan['episode_ids']:
                raise RuntimeError('Append review mapping read-back failed')
        marker.unlink()  # Commit point: the dataset is now usable.
        if staging.is_dir():
            # A post-commit cleanup failure must not report the committed append as failed.
            shutil.rmtree(staging, ignore_errors=True)
        return target
    except BaseException:
        if marker.exists():
            rollback_append(target)
        elif staging.is_dir():
            shutil.rmtree(staging)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='*', type=Path,
                        help='raw sessions to merge, or new sessions to append')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path, help='NEW raw session directory')
    destination.add_argument('--append-to', type=Path,
                             help='existing canonical merged raw dataset to extend transactionally')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Copy files after validation')
    mode.add_argument('--dry-run', action='store_true', help='Validate only (default)')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--allow-mixed-tasks', action='store_true')
    parser.add_argument('--recover-append', action='store_true',
                        help='roll back an interrupted --append-to transaction')
    args = parser.parse_args(argv)
    if args.append_to is not None:
        target = args.append_to.expanduser().resolve()
        if args.include_incomplete:
            parser.error('--include-incomplete is not supported with --append-to')
        if args.recover_append:
            if args.sessions:
                parser.error('--recover-append does not accept source sessions')
            marker = target / APPEND_MARKER
            if not marker.is_file():
                raise RuntimeError(f'No incomplete transactional append at {target}')
            document = read_json(marker)
            print(f"Interrupted append: transaction={document.get('transaction_id')}; "
                  f"target={target}; planned new episodes={len(document.get('appended_episode_names', []))}")
            if args.execute:
                result = rollback_append(target)
                print(f"RECOVERED: {target}; removed {result['removed_episodes']} appended episodes.")
            else:
                print('DRY-RUN: no files changed. Add --execute to roll back this append.')
            return 0
        plan, episodes = build_append_plan(
            args.sessions, target, allow_mixed_tasks=args.allow_mixed_tasks,
        )
        print(f"Append target (raw, NOT LeRobot): {target}")
        for source in plan['source_sessions'][1:]:
            print(f"  + {Path(source['path']).name}: {source['episodes']} episodes / "
                  f"{source['frames']} frames -> "
                  f"{source['first_output_episode']}..{source['last_output_episode']}")
        print(f"Existing: {plan['base_episode_count']} episodes; append: "
              f"{plan['new_episode_count']} episodes; final: {plan['total_episodes']} episodes; "
              f"new copy size ~{plan['estimated_copy_bytes'] / 1024**3:.3f} GiB")
        print('Existing videos/NPZ are retained in place; only new episodes are copied and verified.')
        if args.execute:
            execute_append_plan(plan, episodes)
            print(f'DONE: appended transactionally to {target}; source sessions untouched.')
        else:
            print('DRY-RUN: no files written. Add --execute to append to the rolling raw master.')
        return 0
    if args.recover_append:
        parser.error('--recover-append requires --append-to')
    if not args.sessions:
        parser.error('provide at least one source session')
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
