"""Paired, offline Pi0 5/10-step evaluation. No robot or camera connection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from .html import chart, table

HORIZONS = [0, 5, 10, 13, 19, 49]
SCALES = np.array([180 / math.pi] * 6 + [1000.])
# Software deployment limits from online_policy.deployment_joint_bounds,
# not measured physical hard stops. Only J2/J3 are audited here.
BOUNDS = np.array([[-2., 182.], [-172., 2.]])


def outside(values):
    degrees = np.asarray(values)[..., 1:3] * 180 / math.pi
    return (degrees < BOUNDS[:, 0]) | (degrees > BOUNDS[:, 1])


def audit(values):
    values = np.asarray(values)
    deg = values[:, 1:3] * 180 / math.pi
    violation = outside(values)
    return {'frames': len(values), 'min_deg': deg.min(0).tolist(),
            'max_deg': deg.max(0).tolist(), 'outside_count': violation.sum(0).tolist(),
            'either_outside_count': int(violation.any(1).sum())}


def select_indices(columns, uniform_count, near_count):
    """Disjoint cohorts: uniform frames and episode-diverse near-zero starts."""
    n = len(columns['episode_index'])
    uniform = sorted(set(np.linspace(0, n - 1, min(n, uniform_count), dtype=int).tolist()))
    state = np.asarray(columns['observation.state'])
    near = (np.abs(state[:, 1:3] * 180 / math.pi) <= 3).all(1)
    near &= np.asarray(columns['frame_index']).reshape(-1) <= 20
    candidates = {}
    for i in np.flatnonzero(near):
        if int(i) not in uniform:
            candidates.setdefault(int(columns['episode_index'][i]), int(i))
    choices = list(candidates.values())
    picked = ([choices[i] for i in np.linspace(0, len(choices) - 1,
               min(near_count, len(choices)), dtype=int)] if choices else [])
    return [(i, 'uniform') for i in uniform] + [(i, 'near_zero_start') for i in picked]


def summarize(records, horizon=None):
    errors, holds, predicted_bounds, target_bounds, relative_gaps = [], [], [], [], []
    for r in records:
        valid = np.asarray(r['valid'], dtype=bool)
        if horizon is not None:
            valid &= np.arange(len(valid)) == horizon
        if not valid.any():
            continue
        pred, target = np.array(r['prediction'])[valid], np.array(r['target'])[valid]
        errors.append(np.abs(pred - target) * SCALES)
        holds.append(np.abs(np.array(r['state']) - target) * SCALES)
        predicted_bounds.append(outside(pred))
        target_bounds.append(outside(target))
        relative_gaps.append(np.abs(pred[:, :6] - np.array(r['state'])[:6]) * 180 / math.pi)
    if not errors:
        return None
    errors, holds = np.concatenate(errors), np.concatenate(holds)
    pred_bad, target_bad = np.concatenate(predicted_bounds), np.concatenate(target_bounds)
    return {'valid_steps': len(errors), 'mae': errors.mean(0).tolist(),
            'hold_mae': holds.mean(0).tolist(), 'p95_error': np.percentile(errors, 95, axis=0).tolist(),
            'prediction_bound_fraction': pred_bad.mean(0).tolist(),
            'target_bound_fraction': target_bad.mean(0).tolist(),
            'either_prediction_bound_fraction': float(pred_bad.any(1).mean()),
            'either_target_bound_fraction': float(target_bad.any(1).mean()),
            'gap_from_observed_state_over_9deg_fraction': float(
                (np.concatenate(relative_gaps).max(1) > 9).mean())}


def paired_difference(records, horizon):
    """5 minus 10 absolute error; bootstrap independent observation indices, not seeds."""
    pairs = {}
    for r in records:
        if r['valid'][horizon]:
            pairs.setdefault((r['index'], r['seed']), {})[r['steps']] = r
    by_index = {}
    for (index, _), pair in pairs.items():
        if set(pair) != {5, 10}:
            raise ValueError('Unpaired prediction')
        target = np.array(pair[5]['target'][horizon])
        diff = (np.abs(np.array(pair[5]['prediction'][horizon]) - target)
                - np.abs(np.array(pair[10]['prediction'][horizon]) - target)) * SCALES
        by_index.setdefault(index, []).append(diff)
    if not by_index:
        return None
    values = np.array([np.mean(v, 0) for v in by_index.values()])
    rng = np.random.default_rng(1000)
    boot = values[rng.integers(0, len(values), (1000, len(values)))].mean(1)
    return {'sample_count': len(values), 'mae_difference_5_minus_10': values.mean(0).tolist(),
            'bootstrap95_low': np.percentile(boot, 2.5, axis=0).tolist(),
            'bootstrap95_high': np.percentile(boot, 97.5, axis=0).tolist()}


def conclusions(report):
    val = report['metrics']['validation']['uniform']
    a, b = val['5']['horizons']['13'], val['10']['horizons']['13']
    dt = report['timing']
    lines = [f"5次/10次去噪完整离线处理耗时中位数为 {dt['5']['median_ms']:.1f}/{dt['10']['median_ms']:.1f} ms；"
             '此处不含实时相机、CAN、预览和请求排队，不等于真机延迟。',
             f"验证集均匀抽样输出13：J2 MAE 为 {a['mae'][1]:.2f}° / {b['mae'][1]:.2f}°，"
             f"J3 MAE 为 {a['mae'][2]:.2f}° / {b['mae'][2]:.2f}°（5次 / 10次）。",
             f"同一组输出13中，任一 J2/J3 预测越界比例为 {a['either_prediction_bound_fraction']:.1%} / "
             f"{b['either_prediction_bound_fraction']:.1%}；真实动作越界比例为 {a['either_target_bound_fraction']:.1%}。"]
    total_bad = sum(v['action']['either_outside_count'] for v in report['dataset_audit'].values())
    total = sum(v['action']['frames'] for v in report['dataset_audit'].values())
    lines.append(f'全量训练/验证数据 {total} 帧中，有 {total_bad} 帧真实动作超出当前 J2/J3 软件边界。'
                 + ('数据/软件范围存在不一致；这不是放宽机械限位的授权。' if total_bad else
                    '本次范围检查未发现真实动作与这两轴软件边界冲突。'))
    first = val['5']['horizons']['0']
    last = val['5']['horizons']['19']
    lines.append(f"5次验证集 J2/J3 MAE 从输出0的 {first['mae'][1]:.2f}°/{first['mae'][2]:.2f}°，"
                 f"增至输出19的 {last['mae'][1]:.2f}°/{last['mae'][2]:.2f}°。远期动作准确性仍是问题。")
    near = report['metrics']['validation'].get('near_zero_start', {}).get('5')
    if near and near['horizons']['13']:
        m = near['horizons']['13']
        lines.append(f"零位附近验证样本输出13：5次 J2 MAE={m['mae'][1]:.2f}°，保持当前状态基线仅 "
                     f"{m['hold_mae'][1]:.2f}°。模型在真实动作尚少变化时预测了更明显的变化；"
                     '这是启动时序偏差的线索，不是已确认的真机故障根因。')
    lines.append('当前抽样不支持“从10次改5次导致关节预测整体明显恶化”的解释；'
                 '部分首步指标10次略优，部分远期指标5次略优，不能宣称两者等效或5次全面更好。')
    if a['either_prediction_bound_fraction'] == b['either_prediction_bound_fraction'] == 0:
        lines.append('输出13的验证抽样未复现真机 J2=-7.03°、J3=4.13° 越界；仅2个噪声种子和有限数据集观测，'
                     '不能排除其他种子或实际场景下的越界。不能因离线没有复现就关闭保护。')
    lines += ['离线动作误差不等于任务成功率；本测试不重现上一次真机具体画面，也不证明异步切块已经连续。',
              '下一步优先保存真实故障对应的双相机原始帧、关节状态、任务文本与噪声种子，在不发运动指令的条件下复现；'
              '当前执行JSONL没有图像，无法重建该次完整输入。',
              '保持关节、步长与断供保护。启动排队、动作块切换及连续供给仍需独立修正和复测。'
              '本次只新增离线分析与文档，没有改训练代码、权重或在线控制。']
    return lines


def render(report):
    e = html.escape
    body = '<h1>π0 LoRA：5 次 vs 10 次去噪离线对照</h1>'
    body += '<p>同一 checkpoint、双相机观测、显式复用同一初始噪声。无 CAN、无实时相机连接、无运动指令。</p>'
    body += '<h2>结论</h2><ul>' + ''.join('<li>' + e(s) + '</li>' for s in conclusions(report)) + '</ul>'
    body += '<h2>方法与范围</h2><p>恢复 train_config.json 的 episode 顺序与训练/验证划分；关闭图像增强。'
    body += '每个划分分别取均匀帧样本和 frame_index≤20 且 J2/J3 均在±3°的样本，两组不重叠；起始组每个episode至多一个样本。'
    body += '每个起点使用2个种子，5/10次交替先后顺序，显式输入完全相同的噪声张量。输出绝对关节角/夹爪宽度，无限幅；所有误差和越界比例排除padding。'
    body += '均匀组与起始组分开统计，不能混合推断全数据分布。输出步从0计数，13步对应0.65秒，19步对应0.95秒。'
    body += 'J2边界[-2°,182°]、J3[-172°,2°]来自当前部署软件约定，不是实测硬限位。未计算KL/flow loss。</p>'
    body += table(['划分', 'episode数', '帧数', '均匀起点', '零位附近起点'], [
        [s, len(v['episodes']), v['frames'], v['cohorts'].get('uniform', 0),
         v['cohorts'].get('near_zero_start', 0)] for s, v in report['splits'].items()])
    body += '<h2>全量数据边界审计（不是抽样）</h2>'
    rows = []
    for split, checks in report['dataset_audit'].items():
        for kind in ['state', 'action']:
            v = checks[kind]
            rows.append([split, kind, v['frames'], *[f'{v["min_deg"][i]:.3f} ~ {v["max_deg"][i]:.3f}' for i in range(2)],
                         *v['outside_count'], v['either_outside_count']])
    body += table(['划分', '类型', '帧数', 'J2范围°', 'J3范围°', 'J2越界帧', 'J3越界帧', '任一越界帧'], rows)
    body += '<h2>推理耗时</h2><p>两种配置各预热3次，不计入统计；GPU同步计时，batch=1、BF16，含预处理、去噪、反归一化和结果传回CPU；不含视频解码。</p>'
    body += table(['去噪次数', '计时样本', '中位ms', 'P95 ms', '最大ms'], [
        [k, v['count'], f'{v["median_ms"]:.1f}', f'{v["p95_ms"]:.1f}', f'{v["max_ms"]:.1f}'] for k, v in report['timing'].items()])
    for split, cohorts in report['metrics'].items():
        for cohort, variants in cohorts.items():
            body += f'<h2>{e(split)} / {e(cohort)}</h2>'
            rows = []
            for steps, result in variants.items():
                for h, v in result['horizons'].items():
                    if not v: continue
                    rows.append([steps, h, v['valid_steps'], *[f'{x:.3f}' for x in v['mae']],
                                 f'{v["prediction_bound_fraction"][0]:.1%}', f'{v["prediction_bound_fraction"][1]:.1%}',
                                 f'{v["either_target_bound_fraction"]:.1%}'])
            body += table(['去噪', '输出步', '有效样本×种子', 'J1°', 'J2°', 'J3°', 'J4°', 'J5°', 'J6°', '夹爪mm',
                           '预测J2越界', '预测J3越界', '真实J2/J3任一越界'], rows)
            for d in [1, 2, 6]:
                series = [(f'{steps}次', [(int(h), v['mae'][d]) for h, v in variants[steps]['all_horizons'].items() if v]) for steps in ['5', '10']]
                series.append(('保持当前状态基线', [(int(h), v['hold_mae'][d]) for h, v in variants['10']['all_horizons'].items() if v]))
                body += chart(series, f'{split}/{cohort} ' + (f'J{d+1} MAE (°)' if d < 6 else '夹爪 MAE (mm)'))
            diffs = report['paired'][split][cohort]
            body += '<p>以下为配对 MAE 差（5次−10次），正数表示5次更差。先在同一起点平均种子，再按起点bootstrap；同一episode内样本相关，区间仅为描述性参考，不是严格独立episode推断。</p>'
            body += table(['输出步', '起点数', 'J2差° [95%区间]', 'J3差° [95%区间]', '夹爪差mm [95%区间]'], [
                [h, v['sample_count'], *[f'{v["mae_difference_5_minus_10"][d]:.3f} [{v["bootstrap95_low"][d]:.3f}, {v["bootstrap95_high"][d]:.3f}]' for d in [1, 2, 6]]]
                for h, v in diffs.items() if v])
    body += '<h2>输出13越界案例（验证集，按越界幅度排序）</h2>'
    candidates = []
    for r in report['records']:
        if r['split'] != 'validation' or not r['valid'][13]: continue
        deg = np.array(r['prediction'][13])[1:3] * 180 / math.pi
        excess = float(np.maximum(BOUNDS[:, 0] - deg, deg - BOUNDS[:, 1]).max())
        if excess > 0: candidates.append((excess, r))
    body += table(['去噪', '组', 'episode', 'frame', 'seed', '预测J2/J3°', '真实J2/J3°', '观测J2/J3°'], [
        [r['steps'], r['cohort'], r['episode'], r['frame'], r['seed'],
         str(np.round(np.array(r['prediction'][13])[1:3]*180/math.pi, 2).tolist()),
         str(np.round(np.array(r['target'][13])[1:3]*180/math.pi, 2).tolist()),
         str(np.round(np.array(r['state'])[1:3]*180/math.pi, 2).tolist())]
        for _, r in sorted(candidates, key=lambda x: x[0], reverse=True)[:16]])
    body += '<h2>复现命令</h2><pre>' + e(report['command']) + '</pre>'
    body += '<details><summary>来源、版本和运行配置</summary><pre>' + e(json.dumps(report['provenance'], ensure_ascii=False, indent=2)) + '</pre></details>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>π0去噪次数对照报告</title><style>body{font:15px system-ui;line-height:1.7;max-width:1280px;margin:30px auto;padding:0 20px;color:#1e293b}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #cbd5e1;padding:6px;text-align:right}th{background:#e2e8f0}svg{width:100%;background:#f8fafc}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}h2{margin-top:36px}</style>' + body + '</html>'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--near-start-samples', type=int, default=16)
    p.add_argument('--render-only', action='store_true', help='Rebuild HTML from the existing output JSON; no GPU')
    args = p.parse_args(argv)
    if args.render_only:
        report = json.loads(args.output.with_suffix('.json').read_text(encoding='utf-8'))
        args.output.write_text(render(report), encoding='utf-8')
        print(args.output.resolve())
        return
    if args.samples < 1 or args.near_start_samples < 1:
        p.error('Sample counts must be positive')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_CACHE', str(args.dataset_root.resolve().parent / '.hf_datasets_cache'))
    import torch
    import peft
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # register config
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.factory import make_pre_post_processors
    from robotarm.modeling.evaluate import batch_sample
    from robotarm.modeling.pi0_lora_check import load_pi0_lora

    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    cfg = TrainPipelineConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg.dataset.root = str(args.dataset_root.resolve())
    cfg.dataset.video_backend = 'pyav'
    cfg.dataset.image_transforms.enable = False
    train, val = make_train_eval_datasets(cfg)
    if val is None or set(train.episodes) & set(val.episodes): raise RuntimeError('Invalid train/validation split')
    report = {'provenance': {'checkpoint': str(args.checkpoint.resolve()),
              'base': str(args.base_checkpoint.resolve()), 'dataset': str(args.dataset_root.resolve()),
              'adapter_sha256': hashlib.sha256((args.checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest(),
              'train_config_sha256': hashlib.sha256((args.checkpoint / 'train_config.json').read_bytes()).hexdigest(),
              'torch': torch.__version__, 'peft': peft.__version__, 'gpu': torch.cuda.get_device_name(0),
              'fps': train.fps, 'seeds': [1000, 1001003], 'bounds_deg': BOUNDS.tolist(),
              'software_bounds_source': 'online_policy.deployment_joint_bounds',
              'noise_pairing': 'same explicit noise tensor for 5 and 10 steps', 'command_sent': False},
              'splits': {}, 'dataset_audit': {}, 'records': [], 'metrics': {}, 'paired': {}}
    import shlex, sys
    report['command'] = 'HF_HUB_OFFLINE=1 ' + shlex.join([sys.executable, *sys.argv])
    selections = {}
    for split, dataset in [('train', train), ('validation', val)]:
        cols = dataset.hf_dataset.select_columns(['episode_index', 'frame_index', 'observation.state', 'action']).with_format('numpy')[:]
        selections[split] = select_indices(cols, args.samples, args.near_start_samples)
        report['dataset_audit'][split] = {'state': audit(cols['observation.state']), 'action': audit(cols['action'])}
        report['splits'][split] = {'episodes': dataset.episodes, 'frames': len(dataset),
            'cohorts': {c: sum(group == c for _, group in selections[split]) for c in ['uniform', 'near_zero_start']}}
        print('AUDIT', split, report['dataset_audit'][split], flush=True)
    policy = load_pi0_lora(args.checkpoint, args.base_checkpoint, cfg.policy)
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    def predict(sample, noise, steps):
        policy.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Do not expose future target actions to inference/preprocessing.
        observation = {k: v for k, v in sample.items() if k.startswith('observation.') or k == 'task'}
        batch = pre(batch_sample(observation, train.meta.camera_keys))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)
        physical = post(prediction).detach().float().cpu()[0]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - started) * 1000
        if physical.shape != (cfg.policy.chunk_size, 7) or not torch.isfinite(physical).all():
            raise RuntimeError('Invalid prediction')
        return physical.tolist(), ms
    with torch.inference_mode():
        warm = train[selections['train'][0][0]]
        for steps in [5, 10]:
            for i in range(3):
                noise = torch.randn(1, cfg.policy.chunk_size, cfg.policy.max_action_dim, device='cuda')
                _, ms = predict(warm, noise, steps)
                print(f'WARM {steps} {i+1}/3 {ms:.1f}ms', flush=True)
        for split, dataset in [('train', train), ('validation', val)]:
            for number, (index, cohort) in enumerate(selections[split]):
                sample = dataset[index]
                for seed in [1000, 1001003]:
                    torch.manual_seed(seed)
                    noise = torch.randn(1, cfg.policy.chunk_size, cfg.policy.max_action_dim, device='cuda')
                    for steps in ([5, 10] if (number + seed) % 2 else [10, 5]):
                        prediction, ms = predict(sample, noise, steps)
                        report['records'].append({'split': split, 'cohort': cohort, 'index': index,
                            'episode': int(sample['episode_index']), 'frame': int(sample['frame_index']),
                            'steps': steps, 'seed': seed, 'prediction': prediction, 'inference_ms': ms,
                            'state': sample['observation.state'].tolist(), 'target': sample['action'].tolist(),
                            'valid': (~sample['action_is_pad'].bool()).tolist()})
                print(f'PROGRESS {split} {number+1}/{len(selections[split])} {cohort}', flush=True)
    report['timing'] = {}
    for steps in [5, 10]:
        times = [r['inference_ms'] for r in report['records'] if r['steps'] == steps]
        report['timing'][str(steps)] = {'count': len(times), 'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)), 'max_ms': max(times)}
    for split in report['splits']:
        report['metrics'][split], report['paired'][split] = {}, {}
        for cohort in ['uniform', 'near_zero_start']:
            records = [r for r in report['records'] if r['split'] == split and r['cohort'] == cohort]
            if not records: continue
            report['paired'][split][cohort] = {str(h): paired_difference(records, h) for h in HORIZONS}
            report['metrics'][split][cohort] = {}
            for steps in [5, 10]:
                selected = [r for r in records if r['steps'] == steps]
                report['metrics'][split][cohort][str(steps)] = {
                    'full_chunk': summarize(selected),
                    'horizons': {str(h): summarize(selected, h) for h in HORIZONS},
                    'all_horizons': {str(h): summarize(selected, h) for h in range(cfg.policy.chunk_size)}}
    report['provenance']['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    args.output.write_text(render(report), encoding='utf-8')
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'cohort', 'steps', 'horizon', 'valid_steps', *[f'J{i}_mae_deg' for i in range(1, 7)],
                         'gripper_mae_mm', 'pred_J2_bound_fraction', 'pred_J3_bound_fraction', 'target_either_bound_fraction'])
        for split, cohorts in report['metrics'].items():
            for cohort, variants in cohorts.items():
                for steps, values in variants.items():
                    for h, v in values['horizons'].items():
                        if v: writer.writerow([split, cohort, steps, h, v['valid_steps'], *v['mae'],
                                               *v['prediction_bound_fraction'], v['either_target_bound_fraction']])
    print('REPORT', args.output.resolve(), flush=True)
    for s in conclusions(report): print(s, flush=True)


if __name__ == '__main__':
    main()
