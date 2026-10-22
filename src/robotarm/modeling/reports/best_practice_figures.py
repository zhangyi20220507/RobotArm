"""Render the dated Best Practice figures from saved JSON evidence; no inference.

This is a curated October 2026 experiment snapshot, not a live training monitor.
PNG is embedded in Markdown; SVG and a source-hash manifest accompany each build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCES = {
    'act': 'artifacts/reports/act_training/act_lf180_90000_20261015.json',
    'pi0': 'artifacts/reports/pi0_lora/pi0_lora_lf180_v1_20261014.json',
    'denoising': 'artifacts/reports/pi0_lora/pi0_denoising_5_vs_10_20261014.json',
}
BLUE, ORANGE, GREEN, PURPLE = '#2563eb', '#ea580c', '#15803d', '#7c3aed'


def history_points(rows, key, start=0):
    """Keep raw, finite measurements; do not smooth or interpolate missing steps."""
    return sorted((int(r['_step']), r[key]) for r in rows
                  if '_step' in r and key in r and math.isfinite(r['_step'])
                  and math.isfinite(r[key]) and r['_step'] >= start)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'docs/best_practice_assets/20261016')
    args = parser.parse_args(argv)
    reports, provenance = {}, {}
    for name, relative in SOURCES.items():
        raw = (args.source_root / relative).read_bytes()
        reports[name] = json.loads(raw)
        provenance[name] = {'path': relative, 'sha256': hashlib.sha256(raw).hexdigest()}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter

    # Prefer available CJK fonts; no download or system-font mutation.
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'SimHei']
    font = next((name for name in candidates if name in fonts), None)
    if font is None:
        raise RuntimeError('A CJK font is required to render Chinese figure labels')
    plt.rcParams.update({'font.family': font, 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'robotarm-best-practice',
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    def finish(fig, stem, sources, note):
        fig.text(.02, .02, note, fontsize=9, color='#475569', va='bottom')
        fig.tight_layout(rect=(0, .11, 1, .93))
        for suffix in ['png', 'svg']:
            fig.savefig(args.output_dir / f'{stem}.{suffix}', dpi=170,
                        metadata={'Date': None} if suffix == 'svg' else None)
        plt.close(fig)
        figures.append({'file': f'{stem}.png', 'sources': sources, 'note': note})

    def decorate(ax, title, ylabel, steps=True):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
        if steps:
            ax.set_xlabel('训练步数')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}k'))

    def line(ax, pairs, **kwargs):
        ax.plot(*zip(*pairs), **kwargs)

    act, pi0, denoise = (reports[k] for k in ['act', 'pi0', 'denoising'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle('ACT · LF180 · 训练至 90000 步', fontsize=16, fontweight='bold')
    line(axes[0], history_points(act['history'], 'train/l1_loss'), color=BLUE, label='训练 L1（训练模式）')
    line(axes[0], history_points(act['history'], 'eval/eval_loss'), color=ORANGE, label='全量验证 L1（z=0）')
    decorate(axes[0], '全程原始日志（未平滑）', '归一化 L1')
    axes[0].set_ylim(bottom=0)
    for s in [40000, 60000]:
        axes[0].axvline(s, color='#94a3b8', linestyle=':', linewidth=1)
    axes[0].legend()
    line(axes[1], history_points(act['history'], 'eval/eval_loss', 40000), color=ORANGE, linewidth=1.5)
    for key, label, color in [('best_logged', '最低日志点（未保存）', PURPLE),
                              ('best_saved', '最低已保存 checkpoint', GREEN)]:
        step, loss = act[f'{key}_step'], act[f'{key}_eval']
        axes[1].scatter([step], [loss], color=color, zorder=5)
        axes[1].annotate(f'{step}: {loss:.6f}', (step, loss),
                         xytext=(-8, 40) if key == 'best_saved' else (-100, 16),
                         textcoords='offset points', ha='right', fontsize=9, color=color,
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85},
                         arrowprops={'arrowstyle': '-', 'color': color, 'lw': .8})
        axes[1].plot([], [], 'o', color=color, label=label)
    decorate(axes[1], '40000 步后验证变化（纵轴放大）', '全量验证 L1')
    axes[1].legend(loc='upper right', fontsize=9)
    finish(fig, 'act_loss', ['act'],
           '数据：LF180（182 episodes）；三段续训合并。训练/验证模式不同，左图不构成严格同条件泛化比较。\n'
           '右图为局部纵轴，不从 0 开始；87000 步未保存，不能把其成绩归给相邻权重。')

    latent = sorted(act['latent'].items(), key=lambda item: int(item[0]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle('ACT · 同条件离线 L1：潜变量模式与逐维度误差', fontsize=16, fontweight='bold')
    for split, mode, label, color, style in [
        ('train', 'reconstruction', 'A 训练集 · 后验采样', BLUE, '-'),
        ('train', 'zero', 'B 训练集 · z=0', GREEN, '--'),
        ('validation', 'reconstruction', 'C 验证集 · 后验采样', ORANGE, '-'),
        ('validation', 'zero', 'D 验证集 · z=0', PURPLE, '--'),
    ]:
        line(axes[0], [(int(s), r[split][mode]['l1']) for s, r in latent],
             label=label, color=color, linestyle=style, marker='o', markersize=5)
    decorate(axes[0], 'A≈B、C≈D：曲线几乎重合', '归一化 L1')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    for (step, r), color in zip(latent, [BLUE, ORANGE, GREEN]):
        axes[1].plot(['J1', 'J2', 'J3', 'J4', 'J5', 'J6', '夹爪'],
                     r['validation']['zero']['per_dimension_l1'], marker='o', label=f'{int(step)} 步', color=color)
    decorate(axes[1], '验证集推理式 D：各动作维度', '归一化 L1（不是度 / 毫米）', False)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=9)
    finish(fig, 'act_latent', ['act'],
           '每个 checkpoint：Train / Validation 各 128 个相同起点；seed=1000，batch=8，重建重复 2 次。\n'
           '关闭增强/dropout/BN 更新；仅 L1，不含 KL，排除 padding。点间连线仅辅助阅读，不是额外测量。')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle('π0 LoRA · LF180 · 训练至 20000 步', fontsize=16, fontweight='bold')
    line(axes[0], history_points(pi0['history'], 'train/loss'), color=BLUE)
    decorate(axes[0], '训练 flow loss（原始日志）', 'Flow loss')
    axes[0].set_ylim(bottom=0)
    line(axes[1], history_points(pi0['history'], 'eval/eval_loss'), color=ORANGE, marker='o', markersize=4)
    best = pi0['summary']
    axes[1].scatter([best['best_step']], [best['best_eval']], color=GREEN, zorder=5)
    axes[1].annotate(f"最低：{best['best_step']} 步 / {best['best_eval']:.6f}",
                     (best['best_step'], best['best_eval']), xytext=(-5, 17),
                     textcoords='offset points', ha='right', color=GREEN)
    decorate(axes[1], '验证 flow loss（纵轴放大）', 'Flow loss')
    finish(fig, 'pi0_loss', ['pi0'],
           'batch=1，lr=1e-4，LoRA rank=16 / alpha=32；验证日志每次最多 256 起点。无平滑。\n'
           'Flow loss 不是动作 L1，也不能与 ACT 的 L1 直接比较；最低 flow loss 不保证最低动作误差。')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    fig.suptitle('π0 · 20000 checkpoint · 不同输出步的验证误差', fontsize=16, fontweight='bold')
    for row, (group, title) in enumerate([('uniform', '均匀验证组'), ('near_zero_start', '近零位验证组')]):
        group_data = denoise['metrics']['validation'][group]
        for col, (dim, name, unit) in enumerate([(1, 'J2', '°'), (2, 'J3', '°'), (6, '夹爪', 'mm')]):
            ax = axes[row, col]
            for n, color in [('5', BLUE), ('10', ORANGE)]:
                horizons = group_data[n]['horizons']
                points = [(int(h), m['mae'][dim]) for h, m in horizons.items() if m is not None]
                line(ax, sorted(points), color=color, marker='o', markersize=4, label=f'{n} 次去噪')
            hold = [(int(h), m['hold_mae'][dim]) for h, m in group_data['5']['horizons'].items() if m is not None]
            line(ax, sorted(hold), color='#64748b', linestyle='--', label='保持当前状态')
            decorate(ax, f'{title} · {name}', f'MAE ({unit})', False)
            ax.set_xlabel('输出步 k（20 FPS，时间偏移 k / 20 秒）')
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
    finish(fig, 'pi0_horizon_error', ['denoising'],
           '均匀组 32 起点、近零位组 7 起点；每起点 2 个种子，5/10 次复用同一噪声。排除 padding；远期有效样本可能减少。\n'
           '仅在 k=0/5/10/13/19/49 实测，连线辅助阅读；MAE 是对真实动作的误差，不是相对反馈的 raw gap，也不代表成功率。')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.suptitle('π0 · 5 / 10 次去噪的离线推理耗时', fontsize=16, fontweight='bold')
    for offset, field, label, color in [(-.18, 'median_ms', '中位数', BLUE), (.18, 'p95_ms', 'P95', ORANGE)]:
        bars = ax.bar([i + offset for i in range(2)],
                      [denoise['timing'][n][field] for n in ['5', '10']], width=.34, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f ms', padding=4, fontsize=10)
    ax.set_xticks([0, 1], ['5 次去噪', '10 次去噪'])
    decorate(ax, 'RTX A4000 16 GB · 双相机输入 · batch=1', '完整动作块推理耗时 (ms)', False)
    ax.set_ylim(0, max(v['p95_ms'] for v in denoise['timing'].values()) * 1.2)
    ax.legend(loc='upper left')
    finish(fig, 'pi0_latency', ['denoising'],
           '每设置 174 次预测，各丢弃 3 次预热；柱子为中位数/P95，不是均值或误差棒。\n'
           '含预处理和 CPU 回传；不含视频解码、实时相机、CAN、预览或异步排队，不能代替真机控制周期。')

    manifest = {'purpose': '2026-10-16 Best Practice illustration snapshot',
                'sources': provenance, 'figures': figures, 'font': font,
                'matplotlib_version': matplotlib.__version__,
                'inference_performed': False, 'hardware_connected': False}
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {len(figures)} figures (PNG + SVG): {args.output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
