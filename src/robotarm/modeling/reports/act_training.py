"""Merge ACT resume histories and existing paired latent reports; no hardware."""
import argparse
import csv
import html
import json
import os
import statistics
from pathlib import Path

from .html import chart, table
from robotarm.modeling.training_report import read_run_history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--latent-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    history, sources = read_run_history(args.run_dir)
    merged = {int(row['_step']): row for row in history}
    evals = [(int(r['_step']), r['eval/eval_loss']) for r in history if 'eval/eval_loss' in r]
    best = min(evals, key=lambda x: x[1])
    saved = {int(p.parent.parent.name) for p in args.run_dir.glob('checkpoints/*/pretrained_model/config.json')
             if p.parent.parent.name.isdigit()}
    best_saved = min((v for v in evals if v[0] in saved), key=lambda x: x[1])
    latent = {Path(r['checkpoint']).parent.name: r for r in
              (json.loads(p.read_text()) for p in args.latent_reports)}
    latent = dict(sorted(latent.items(), key=lambda item: int(item[0])))
    if len(latent) < 2:
        raise ValueError('Provide at least two distinct checkpoint reports')
    for step, report in latent.items():
        expected = args.run_dir / 'checkpoints' / step / 'pretrained_model'
        if Path(report['checkpoint']).resolve() != expected.resolve():
            raise ValueError('Latent report belongs to a different run')
        if int(step) not in merged or 'eval/eval_loss' not in merged[int(step)]:
            raise ValueError(f'No validation history for checkpoint {step}')
    base = next(iter(latent.values()))
    for report in latent.values():
        for split in ['train', 'validation']:
            for key in ['episodes', 'sample_indices', 'total_frames']:
                if report[split][key] != base[split][key]:
                    raise ValueError('Latent reports are not matched')
        for key in ['seed', 'batch_size', 'repeats', 'chunk_size']:
            if report[key] != base[key]: raise ValueError('Latent evaluation configuration mismatch')
    configs = {s: json.loads((args.run_dir / 'checkpoints' / s / 'pretrained_model/train_config.json').read_text())
               for s in latent}
    config = configs[max(configs)]
    for c in configs.values():
        if c['policy']['type'] != 'act': raise ValueError('Expected ACT checkpoints')
        if c['dataset'] != config['dataset']: raise ValueError('Dataset configs changed across resume')
    first_step, final_step = min(map(int, latent)), max(map(int, latent))
    first_eval, final_eval = merged[first_step]['eval/eval_loss'], merged[final_step]['eval/eval_loss']
    delta = final_eval - first_eval
    latent_gap = max(abs(r[split]['reconstruction']['l1'] - r[split]['zero']['l1'])
                     for r in latent.values() for split in ['train', 'validation'])
    overview = {'run': str(args.run_dir), 'sources': sources, 'best_logged_step': best[0],
                'best_logged_eval': best[1], 'best_saved_step': best_saved[0],
                'best_saved_eval': best_saved[1], 'history': history, 'latent': latent,
                'config': config, 'command_sent': False}
    e = html.escape
    body = f'<h1>ACT 续训对照：{e(args.run_dir.name)}</h1>'
    body += f'<p>合并{len(sources)}段去重后的原始WandB日志，读取同起点A/B/C/D离线报告。不连接机械臂。</p>'
    body += f'<h2>结果摘要</h2><ul><li>{first_step} → {final_step}步，日志验证loss从{first_eval:.6f}变为{final_eval:.6f}，差值{delta:+.6f}（负数为改善）。</li>'
    body += '<li>同一批样本的推理式验证L1：' + '；'.join(
        f"{s}步={r['validation']['zero']['l1']:.6f}" for s,r in latent.items()) + '。抽样与日志口径不同。</li>'
    body += f'<li>各组重建式/推理式L1最大绝对差={latent_gap:.6f}；差值接近0提示潜变量模式切换并非主要误差来源，但不能单独证明泛化根因。</li>'
    body += '<li>应同时查看训练/验证曲线和逐维度结果；训练loss下降不代表真机成功率同步提高，也不能仅凭差距断言严重过拟合。</li>'
    body += f'<li>最低日志点为{best[0]}步（{best[1]:.6f}），该步{"已" if best[0] in saved else "未"}保存。已有checkpoint中最低验证loss对应{best_saved[0]}步（{best_saved[1]:.6f}）；不使用“最近checkpoint”冒充最低点。</li></ul>'
    body += '<h2>实际训练配置</h2>'
    body += table(['项目', '数值'], [
        ['run', args.run_dir.name], ['dataset', Path(config['dataset']['root']).name],
        ['训练/验证', f"{len(base['train']['episodes'])}/{len(base['validation']['episodes'])} episodes；{base['train']['total_frames']}/{base['validation']['total_frames']} frames"],
        ['batch / seed', f"{config['batch_size']} / {config['seed']}"],
        ['chunk_size / n_action_steps', f"{config['policy']['chunk_size']} / {config['policy']['n_action_steps']}"],
        ['学习率 / scheduler', f"{config['optimizer']['lr']} / {config['scheduler']}"],
        ['KL权重 / 增强', f"{config['policy']['kl_weight']} / {config['dataset']['image_transforms']['enable']}"],
        ['验证/保存周期', f"{config['eval_steps']} / {config['save_freq']} steps"],
        ['max_eval_samples', config['max_eval_samples']],
        [f'{final_step}步累计样本/epoch', f"{merged[final_step].get('train/samples', '未记录')} / {merged[final_step].get('train/epochs', '未记录')}"],
    ])
    body += '<h2>训练与验证曲线</h2>'
    body += chart([('训练L1（训练模式）', [(int(r['_step']), r['train/l1_loss']) for r in history if 'train/l1_loss' in r]),
                   ('验证L1（eval/z=0）', evals)], '全训练过程：归一化L1')
    body += chart([('验证L1', [(s, v) for s, v in evals if s >= first_step])], f'{first_step}步之后的验证变化（注意纵轴缩放）')
    body += table(['步数', '训练L1日志值', '训练总loss', '验证loss', '最后1000步训练L1均值'], [
        [s, f"{merged[s]['train/l1_loss']:.6f}", f"{merged[s]['train/loss']:.6f}", f"{merged[s]['eval/eval_loss']:.6f}",
         f"{statistics.mean(r['train/l1_loss'] for r in history if s-1000 < r['_step'] <= s):.6f}"]
        for s in map(int, latent)])
    body += '<p>训练日志为训练器聚合值。训练总loss含kl_weight×KL，训练L1使用真实动作后验采样并开启dropout；'
    body += '当前安装的ACT在eval时使用z=0，不加KL，L1排除padding。验证循环对各batch loss取算术平均；'
    body += 'max_eval_samples=0表示完整留出集评估。两种模式的训练/验证日志不能直接作为严格同条件比较。</p>'
    body += '<h2>同条件潜变量对照（读取已有离线实测报告）</h2>'
    body += f"<p>每个checkpoint在Train/Validation评估{base['train']['evaluated_frames']}/{base['validation']['evaluated_frames']}个相同起点，batch={base['batch_size']}、seed={base['seed']}、重建重复{base['repeats']}次；"
    body += '关闭增强、dropout和BN更新，FP32。A/C使用真实动作编码后验采样，B/D使用z=0；'
    body += '只计算L1，排除padding，按有效元素加权。是否全量见详细报告；不能换算成任务成功率。</p>'
    body += table(['checkpoint', 'A 训练重建', 'B 训练推理', 'C 验证重建', 'D 验证推理', 'D−B'], [
        [s, *[f"{r[split][mode]['l1']:.6f}" for split, mode in
              [('train', 'reconstruction'), ('train', 'zero'), ('validation', 'reconstruction'), ('validation', 'zero')]],
         f"{r['validation']['zero']['l1']-r['train']['zero']['l1']:.6f}"] for s, r in latent.items()])
    body += '<h3>验证集推理式L1：按动作维度</h3><p>归一化单位，不是角度或毫米；不要用不同维度数值直接判断物理误差大小。</p>'
    body += table(['checkpoint', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', '夹爪'], [
        [s, *[f'{x:.6f}' for x in r['validation']['zero']['per_dimension_l1']]] for s, r in latent.items()])
    body += f'<h2>如何使用结果</h2><ol><li>把{best_saved[0]}作为验证指标优先候选，保留其他checkpoint作对照；不承诺真机一定更好。</li>'
    body += '<li>在相同场景、起始姿态和运行参数下比较候选，记录失败类型和完整任务成功率；不要仅凭训练loss决定继续堆步数。</li>'
    body += '<li>若后续做训练实验，可单独测试较低学习率或针对失败状态补纠错示范；它们是候选实验，不是本报告已经验证的改进。</li>'
    body += '<li>不依据这些loss解除软件限位，也不能把此前π0异步问题直接归因到ACT。</li></ol>'
    body += '<h2>详细离线报告</h2><ul>' + ''.join(
        f'<li><a href="{e(os.path.relpath(p.with_suffix(".html").resolve(), args.output.parent.resolve()))}">{e(p.stem)}</a></li>' for p in args.latent_reports) + '</ul>'
    body += '<details><summary>日志来源</summary><pre>' + e(json.dumps(sources, ensure_ascii=False, indent=2)) + '</pre></details>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ACT 续训对照</title><style>body{font:16px system-ui;line-height:1.7;max-width:1150px;margin:30px auto;padding:0 20px;color:#1e293b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cbd5e1;padding:8px;text-align:right}th{background:#e2e8f0}svg{width:100%;background:#f8fafc}pre{white-space:pre-wrap}</style>' + body + '</html>'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    keys = ['_step', 'train/loss', 'train/l1_loss', 'train/kld_loss', 'eval/eval_loss', 'train/lr']
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(history)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
