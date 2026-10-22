"""Read Pi0 LoRA logs and compare sampled open-loop chunks. Never sends commands."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import time
from pathlib import Path

from robotarm.modeling.training_report import read_run_history
from .html import chart


def points(rows, key):
    return sorted((int(r['_step']), r[key]) for r in rows
                  if '_step' in r and key in r and math.isfinite(r[key]))


def summarize_predictions(records):
    import numpy as np
    scales = np.array([180/math.pi]*6+[1000.])
    errors, holds, gaps, normalized = [], [], [], []
    horizon = {str(h): [] for h in [0,4,19,49]}
    for r in records:
        pred, target = np.asarray(r['prediction']), np.asarray(r['target'])
        valid = np.asarray(r['valid'], dtype=bool)
        errors.append(np.abs(pred-target)[valid]*scales)
        holds.append(np.abs(np.asarray(r['state'])[None,:]-target)[valid]*scales)
        gaps.append(np.abs(pred[0]-np.asarray(r['state']))*scales)
        normalized.extend(r['normalized_abs_errors'])
        for h in horizon:
            if int(h)<len(valid) and valid[int(h)]:
                horizon[h].append((np.abs(pred[int(h)]-target[int(h)])*scales).tolist())
    error, hold, gap = np.concatenate(errors), np.concatenate(holds), np.asarray(gaps)
    gripper_first=np.array([r['prediction'][0][6]*1000 for r in records])
    return {'units': ['deg']*6+['mm'], 'valid_action_steps': len(error),
            'mae': error.mean(axis=0).tolist(), 'hold_mae': hold.mean(axis=0).tolist(),
            'p95_abs_error': np.percentile(error,95,axis=0).tolist(),
            'max_abs_error': error.max(axis=0).tolist(),
            'normalized_l1': float(np.mean(normalized)),
            'first_joint_gap_over_9deg_fraction': float((gap[:,:6].max(axis=1)>9).mean()),
            'first_gripper_gap_over_18mm_fraction': float((gap[:,6]>18).mean()),
            'first_gripper_outside_0_100mm_fraction': float(np.mean([
                not 0<=r['prediction'][0][6]*1000<=100 for r in records])),
            'first_gripper_range_mm':[float(gripper_first.min()),float(gripper_first.max())],
            'first_gripper_max_boundary_violation_mm':float(np.maximum(
                np.maximum(-gripper_first,gripper_first-100),0).max()),
            'horizon_mae': {h: np.mean(v,axis=0).tolist() if v else None for h,v in horizon.items()},
            'median_chunk_inference_ms': statistics.median(r['inference_ms'] for r in records)}


def evaluate(report, args):
    import torch
    import peft
    from peft import PeftConfig, PeftModel
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from robotarm.modeling.evaluate import batch_sample
    from robotarm.modeling.pi0_lora_driver import load_verified_base

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; prediction evaluation requires a GPU')
    torch.set_num_threads(4)
    report['runtime'] = {'torch':torch.__version__, 'peft':peft.__version__,
                         'gpu':torch.cuda.get_device_name(0)}
    checkpoints = [Path(x) for x in report['evaluated_checkpoints']]
    cfg = TrainPipelineConfig.from_pretrained(checkpoints[-1],local_files_only=True)
    cfg.dataset.root = str(args.dataset_root.resolve())
    cfg.dataset.image_transforms.enable = False
    cfg.dataset.video_backend = 'pyav'
    train, val = make_train_eval_datasets(cfg)
    if val is None or set(train.episodes)&set(val.episodes):
        raise RuntimeError('Missing or overlapping validation split')
    report['splits'] = {n:{'episodes':d.episodes,'total_frames':len(d)}
                        for n,d in [('train',train),('validation',val)]}
    config = cfg.policy
    config.use_peft = False
    config.device = 'cuda'
    config.pretrained_path = str(args.base_checkpoint.resolve())
    base = load_verified_base(PI0Policy.from_pretrained,args.base_checkpoint,args.base_checkpoint,
                             config=config).to('cuda').eval()
    report['base_projection_weights_verified'] = True
    adapter_cfg = PeftConfig.from_pretrained(checkpoints[0])
    adapter_cfg.base_model_name_or_path = str(args.base_checkpoint.resolve())
    policy = PeftModel.from_pretrained(base,checkpoints[0],config=adapter_cfg,
                                      adapter_name='cp0',is_trainable=False).eval()
    report['offline'] = {}
    for cp_index, checkpoint in enumerate(checkpoints):
        if cp_index:
            adapter_cfg = PeftConfig.from_pretrained(checkpoint)
            adapter_cfg.base_model_name_or_path = str(args.base_checkpoint.resolve())
            policy.load_adapter(checkpoint,adapter_name=f'cp{cp_index}',
                                config=adapter_cfg,is_trainable=False)
        policy.set_adapter(f'cp{cp_index}')
        policy.eval()
        pre, post = make_pre_post_processors(policy_cfg=config,pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor':{'device':'cuda'}})
        name = checkpoint.parent.name
        report['offline'][name] = {}
        for split, dataset in [('train',train),('validation',val)]:
            count = min(args.samples,len(dataset))
            indices = [i*len(dataset)//count for i in range(count)]
            records = []
            with torch.inference_mode():
                for i, index in enumerate(indices):
                    sample = dataset[index]
                    batch = pre(batch_sample(sample,dataset.meta.camera_keys))
                    for repeat in range(args.repeats):
                        seed = args.seed+i+repeat*1000003
                        torch.manual_seed(seed)
                        policy.reset()
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            prediction = policy.predict_action_chunk(batch)
                        torch.cuda.synchronize()
                        duration=(time.perf_counter()-start)*1000
                        if not torch.isfinite(prediction).all():
                            raise RuntimeError('Nonfinite prediction')
                        physical=post(prediction).float().cpu()[0]
                        valid=~sample['action_is_pad'].bool()
                        normalized=(prediction.float()-batch['action'].float()).abs().cpu()[0][valid]
                        records.append({'index':index,'episode':int(sample['episode_index']),
                            'frame':int(sample['frame_index']),'seed':seed,'inference_ms':duration,
                            'state':sample['observation.state'].tolist(),'valid':valid.tolist(),
                            'prediction':physical.tolist(),'target':sample['action'].tolist(),
                            'normalized_abs_errors':normalized.tolist()})
                    print(f'{name} {split} {i+1}/{count}',flush=True)
            report['offline'][name][split] = {'sample_indices':indices,'repeats':args.repeats,
                'metrics':summarize_predictions(records),'records':records}
        torch.cuda.empty_cache()
    report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3


def render(report):
    e=html.escape
    summary=report['summary']
    config=report['config']
    body=f'''<h1>π0 LoRA 离线训练效果分析</h1><p>数据：{e(config['dataset_name'])}；不连接 CAN、不发机械臂指令。</p>
    <h2>结论与范围</h2><ul>
    <li>已完成 {summary['final_step']} 步。实际 batch={config['batch_size']}，不是建议过的 batch=4。</li>
    <li>首个验证点 {summary['first_eval']:.6f} → 最后 {summary['last_eval']:.6f}，下降 {summary['eval_drop_pct']:.1f}%。这不是与未微调基础模型的对比。</li>
    <li>最低验证 loss={summary['best_eval']:.6f}，step={summary['best_step']}；最后一点比最低点高 {summary['last_vs_best_pct']:.1f}%。优先候选是最低点和最终点，波动不证明过拟合。</li>
    <li>累计采样 {summary['samples_seen']:.0f}，日志 epoch={summary['epochs']:.3f}。chunk 包含未来监督，不能简单等同于“只学习了这些帧”，也不能据此认定未收敛。</li>
    <li>验证每次最多 {config['max_eval_samples']} 个起点，不是全验证集；flow-matching 有随机噪声/时间采样。日志不能直接换算成关节角误差，也不能与 ACT L1 横比。</li>
    </ul>'''
    body+=chart([('训练 flow loss',report['train_points']),('验证 flow loss',report['eval_points'])],'训练与验证 loss（日志原值）')
    body+=chart([('学习率',report['lr_points'])],'实际学习率')
    body+='<h2>训练配置</h2><pre>'+e(json.dumps(config,ensure_ascii=False,indent=2))+'</pre>'
    body+='<h2>离线动作预测</h2><p>恢复保存的 episode 顺序及训练/验证划分，分别均匀采样起点；真实图像和状态输入，预测整个 chunk。没有 online 限幅。随机种子在两个 checkpoint 间配对。所有动作误差排除 padding；这是 teacher-forced open-loop，不是闭环任务成功率。归一化 L1 也不是 flow loss。</p>'
    if report.get('prediction_error'):
        body+='<p class="warning">预测评估未完成：'+e(report['prediction_error'])+'</p>'
    offline=report.get('offline',{})
    if len(offline)==2 and all('validation' in s for s in offline.values()):
        names=list(offline)
        a,b=[offline[n]['validation']['metrics']['normalized_l1'] for n in names]
        body+=f'<p><strong>抽样比较：</strong>{e(names[0])} 验证动作 L1={a:.5f}；{e(names[1])}={b:.5f}。动作 L1 与日志 flow loss 的排名可能不同；样本很少，不应把这点差异认定为显著优势。</p>'
    if offline:
        final_name=list(offline)[-1]
        final_val=offline[final_name].get('validation',{}).get('metrics')
        if final_val and final_val['horizon_mae']['19'] and final_val['horizon_mae']['49']:
            h=final_val['horizon_mae']
            grip_gain=100*(1-final_val['mae'][6]/max(1e-12,final_val['hold_mae'][6]))
            body+=f'<p><strong>主要问题：</strong>最终模型的验证 J2 MAE 从首步 {h["0"][1]:.2f}° 增至输出19的 {h["19"][1]:.2f}°、输出49的 {h["49"][1]:.2f}°。长时预测明显比首步困难。全chunk夹爪 MAE={final_val["mae"][6]:.2f}mm，保持当前宽度基线={final_val["hold_mae"][6]:.2f}mm，仅改善约 {grip_gain:.1f}%；这组抽样尚不能说明夹爪时序学得可靠。</p>'
    for checkpoint, splits in offline.items():
        body+=f'<h3>Checkpoint {e(checkpoint)}</h3>'
        for split, result in splits.items():
            m=result['metrics']
            body+=f'<h4>{e(split)}：{len(result["sample_indices"])} 起点 × {result["repeats"]} 次，归一化 L1={m["normalized_l1"]:.5f}</h4>'
            body+='<table><tr><th>维度</th><th>单位</th><th>全 chunk MAE</th><th>P95误差</th><th>最大误差</th><th>保持当前状态 MAE</th><th>输出0 MAE</th><th>输出19 MAE</th></tr>'
            for d in range(7):
                cells=[m[k][d] for k in ['mae','p95_abs_error','max_abs_error','hold_mae']]
                cells += [m['horizon_mae'][h][d] if m['horizon_mae'][h] else float('nan') for h in ['0','19']]
                body+=f'<tr><td>{"J"+str(d+1) if d<6 else "夹爪"}</td><td>{m["units"][d]}</td>'+''.join(f'<td>{v:.3f}</td>' for v in cells)+'</tr>'
            body+='</table>'
            body+=f'<p>首步相对真实状态：任一关节差值&gt;9°占 {m["first_joint_gap_over_9deg_fraction"]:.1%}；夹爪差值&gt;18mm占 {m["first_gripper_gap_over_18mm_fraction"]:.1%}；夹爪输出不在[0,100]mm占 {m["first_gripper_outside_0_100mm_fraction"]:.1%}。夹爪首步范围 {m["first_gripper_range_mm"][0]:.3f}–{m["first_gripper_range_mm"][1]:.3f}mm，最大边界越出 {m["first_gripper_max_boundary_violation_mm"]:.3f}mm；轻微负值与大幅越界应区分。这些是旧运行门控的参考阈值，不是机械关节物理限位或安全认证。chunk 推理中位耗时 {m["median_chunk_inference_ms"]:.1f}ms，非完整控制循环延迟。</p>'
            for d in range(7):
                recs=[r for r in result['records'] if r['seed']==report['seed']+result['sample_indices'].index(r['index'])]
                scale=180/math.pi if d<6 else 1000
                body+=chart([('真实动作',[(i,r['target'][0][d]*scale) for i,r in enumerate(recs)]),
                             ('预测动作',[(i,r['prediction'][0][d]*scale) for i,r in enumerate(recs)])],
                            f'{split} '+('J'+str(d+1) if d<6 else '夹爪')+' 首步动作：横轴是采样点编号，非连续时间')
    latency=statistics.median(m['metrics']['median_chunk_inference_ms']
        for s in offline.values() for m in s.values()) if offline else None
    latency_note=f'本次GPU chunk 推理中位耗时约 {latency:.0f}ms，而5步只覆盖250ms，未来部署需要重新设计推理/动作调度。' if latency else ''
    body+='<h2>解释和下一步</h2><p>保持当前状态是短步预测的重要基线；预测误差应逐维与它比较，不能只看一个混合度/mm的平均值。低 flow loss 不保证 rollout 好。若两个候选差异很小，应增加验证起点/随机种子，并在同样初始条件下做安全的闭环测试。</p><p>本次预测独立评估完整chunk，不受在线 n_action_steps 截断影响。按20Hz数据，第19输出相对起点约0.95秒，第49输出约2.45秒；消耗20步时必须留意长预测误差，不能照搬ACT的最佳设置。'+latency_note+'当前 online-policy 的 π0 执行是实验路径；离线报告不替代 shadow 和闭环验证。</p>'
    body+='<details><summary>来源与复现信息</summary><pre>'+e(json.dumps({k:v for k,v in report.items() if k not in ['offline','history']},ensure_ascii=False,indent=2))+'</pre></details>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>π0 LoRA 离线报告</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#182334;line-height:1.65}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:7px;text-align:right}pre{white-space:pre-wrap;background:#f1f5f9;padding:14px}svg{width:100%;background:#f8fafc}li{margin:8px 0}.warning{color:#b91c1c}</style>'+body+'</html>'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--logs-only',action='store_true')
    args=p.parse_args(argv)
    if min(args.samples,args.repeats)<1: p.error('samples/repeats must be positive')
    run=args.run_dir.resolve()
    rows, sources=read_run_history(run)
    eps=points(rows,'eval/eval_loss')
    if not eps: raise RuntimeError('No validation history')
    best_step,best=min(eps,key=lambda x:x[1])
    final_step=max(int(r['_step']) for r in rows)
    checkpoints=[run/'checkpoints'/f'{s:06d}'/'pretrained_model' for s in dict.fromkeys([best_step,final_step])]
    for cp in checkpoints:
        if not (cp/'adapter_config.json').is_file(): raise RuntimeError(f'Checkpoint missing: {cp}')
    cfg=json.loads((checkpoints[-1]/'train_config.json').read_text())
    if cfg['policy']['type']!='pi0':raise RuntimeError('Expected Pi0')
    last=max(rows,key=lambda r:r['_step'])
    report={'run_dir':str(run),'history_sources':sources,'seed':args.seed,
        'evaluated_checkpoints':[str(x) for x in checkpoints],
        'config_sha256':hashlib.sha256((checkpoints[-1]/'train_config.json').read_bytes()).hexdigest(),
        'config':{'dataset_name':args.dataset_root.name,'batch_size':cfg['batch_size'],
            'steps':cfg['steps'],'max_eval_samples':cfg['max_eval_samples'],'seed':cfg['seed'],
            'chunk_size':cfg['policy']['chunk_size'],'n_action_steps':cfg['policy']['n_action_steps'],
            'dtype':cfg['policy']['dtype'],'peft':cfg['peft'],'optimizer':cfg['optimizer'],
            'scheduler':cfg['scheduler'],'saved_base_path':cfg['policy']['pretrained_path'],
            'local_base_path':str(args.base_checkpoint.resolve())},
        'summary':{'final_step':final_step,'first_eval':eps[0][1],'last_eval':eps[-1][1],
            'best_step':best_step,'best_eval':best,'eval_drop_pct':100*(1-eps[-1][1]/eps[0][1]),
            'last_vs_best_pct':100*(eps[-1][1]/best-1),'samples_seen':last.get('train/samples',0),
            'epochs':last.get('train/epochs',0),'final_train_loss':last.get('train/loss'),
            'training_runtime_s':last.get('_runtime')},
        'history':rows,'train_points':points(rows,'train/loss'),'eval_points':eps,
        'lr_points':points(rows,'train/lr')}
    if not args.logs_only:
        try:evaluate(report,args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            report['prediction_error']=f'{type(exc).__name__}: {exc}'
    else:report['prediction_error']='仅日志分析，未运行动作预测'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render(report),encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with args.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    print('REPORT',args.output.resolve(),flush=True)
    return 1 if report.get('prediction_error') and not args.logs_only else 0


if __name__=='__main__':
    raise SystemExit(main())
