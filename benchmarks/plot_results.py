"""Recreate the README figures from the benchmark exports in benchmarks/data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'benchmarks/data'
OUT = ROOT / 'docs/assets'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'stylegan2-sobieszek',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.titleweight': 'bold',
                     'axes.labelcolor': '#334155', 'text.color': '#0f172a'})
COLORS = ['#94a3b8', '#3b82f6', '#0d9488', '#60a5fa', '#047857']
m4 = json.loads((DATA/'m4_revalidated.json').read_text())
rows = {(r['model'], r['variant'], r['batch']): r for r in m4['records']
        if r['status'] == 'ok' and r.get('comfortable')}

def ms(model, variant, batch):
    return rows[str(model), variant, batch]['median_ms']

def bars(ax, labels, values, colors):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=.65)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.24)
    ax.set_xlabel('Milliseconds per batch · lower is faster')
    ax.grid(axis='x', alpha=.16)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    for i, v in enumerate(values):
        ax.text(v+max(values)*.018, i, f'{v:,.1f} ms', va='center', fontsize=10)

def save(fig, name, note):
    fig.text(.02, .025, note, fontsize=9, color='#475569')
    fig.savefig(OUT/name, bbox_inches='tight', facecolor='white', metadata={'Date': None})
    plt.close(fig)

fig, ax = plt.subplots(figsize=(13.5, 4.7))
variants = ['default_fp32', 'optimized_fp32', 'optimized_metal_fp32',
            'optimized_fp16', 'optimized_metal_fp16']
labels = ['NVIDIA: Pure PyTorch FP32', 'Ours: Optimized PyTorch FP32',
          'Ours: Optimized + Custom Metal Kernels FP32',
          'Ours: Optimized PyTorch FP16', 'Ours: Optimized + Custom Metal Kernels FP16']
bars(ax, labels, [ms(1024,v,1) for v in variants], COLORS)
ax.set_title('Full 1024px StyleGAN2 · Apple M4 · batch 1', loc='left', pad=20)
fig.subplots_adjust(left=.42, bottom=.19, top=.85)
save(fig, 'apple_m4_1024.svg', 'W → image, eager. FP16 applies to the high-resolution blocks.')

fp16 = json.loads((DATA/'cuda_b64_fp16.json').read_text())
fig, ax = plt.subplots(figsize=(13.5, 4.7))
keys = ['fp32-legacy','fp16-control-eager','fp16-candidate-eager',
        'fp16-control-compile','fp16-candidate-compile']
labels = ['NVIDIA: FP32 eager (CUDA ops)', 'NVIDIA: PyTorch FP16 eager',
          'Ours: Optimized PyTorch FP16 eager', 'NVIDIA: PyTorch FP16 compiled',
          'Ours: Optimized PyTorch FP16 compiled']
values = [fp16['forward'][k]['median_ms'] for k in keys]
bars(ax, labels, values, COLORS)
ax.set_title('StyleGAN2 256px · NVIDIA RTX PRO 6000 Blackwell · batch 64', loc='left', pad=20)
fig.subplots_adjust(left=.42, bottom=.19, top=.85)
save(fig, 'cuda_b64.svg', 'W → image. FP16 applies to the high-resolution blocks.')

fig, axes = plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for ax, model in zip(axes, (128,256)):
    batches = (1,8)
    positions = np.arange(2)
    for i,(variant,label,color) in enumerate([
        ('optimized_fp32','Ours: Optimized PyTorch FP32','#3b82f6'),
        ('optimized_metal_fp32','Ours: Optimized + Custom Metal Kernels FP32','#0d9488'),
        ('optimized_metal_fp16','Ours: Optimized + Custom Metal Kernels FP16','#047857')]):
        ratios=[ms(model,'default_fp32',b)/ms(model,variant,b) for b in batches]
        xs=positions+(i-1)*.24
        ax.bar(xs,ratios,width=.22,color=color,label=label)
        for x,v in zip(xs,ratios): ax.text(x,v+.05,f'{v:.2f}×',ha='center',fontsize=9)
    ax.axhline(1,color='#94a3b8',linestyle='--',linewidth=1)
    ax.set_xticks(positions,['Batch 1','Batch 8'])
    ax.set_title(f'{model}px early-output model',pad=12)
    ax.set_ylim(0,3.5)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Speedup over NVIDIA FP32')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), frameon=False, fontsize=9)
fig.suptitle('Early-output generators · Apple M4 · batch scaling',weight='bold',y=.98)
fig.subplots_adjust(bottom=.34,top=.85,wspace=.13)
save(fig,'early_output_scaling.svg','Speedup relative to the reference implementation at the same batch size.')
print('Wrote 3 SVG benchmark figures.')
