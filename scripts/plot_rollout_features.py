#!/usr/bin/env python3
"""Plot archived GRPO/session observations; no model execution or inferred timings."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#2563EB'
GRAY='#94A3B8'
INK='#172B4D'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    source=args.root/'bench/results/rollout-features/observations.json'
    d=json.loads(source.read_text())
    assert d['evidence_class']=='historical_single_run_observations'
    out=args.out or args.root/'docs/assets/performance'
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#CBD5E1',
        'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'qwen3-rollout-features-v1'})
    def bars(ax,vals,labels,title,unit,fmt):
        ax.barh([0,1],vals,color=[GRAY,BLUE],height=.48)
        ax.set_yticks([0,1],labels)
        ax.set_ylim(1.55,-.55)
        ax.set_xlim(0,max(vals)*1.22)
        ax.set_xlabel(unit)
        ax.set_title(title,loc='left',pad=14,fontweight='bold',fontsize=13)
        ax.grid(axis='x',color='#E2E8F0'); ax.set_axisbelow(True)
        for i,v in enumerate(vals): ax.text(v+max(vals)*.02,i,format(v,fmt),va='center',fontsize=12,fontweight='bold')
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,facecolor='white')
        fig.savefig(out/f'{name}.svg',metadata={'Date':None},facecolor='white')
        plt.close(fig)
    s=d['session_kv']; times=s['timing_s']
    fig,axes=plt.subplots(1,2,figsize=(13,5.7))
    fig.subplots_adjust(left=.16,right=.98,top=.69,bottom=.29,wspace=.57)
    fig.suptitle('Session KV in a complete GRPO step',x=.06,y=.96,ha='left',fontsize=22,fontweight='bold')
    fig.text(.06,.885,'CodeScout-4B  |  8 prompts × 8 samples  |  One observed training step per arm',fontsize=11)
    for ax,key,title in zip(axes,['generate','step'],['Generation phase ↓','Complete training step ↓']):
        vals=[times[key]['off'],times[key]['on']]
        bars(ax,vals,['Baseline','Session-enabled'],title,'Wall time (s)',',.2f')
        ax.text(0,1.28,f'{vals[0]/vals[1]:.3f}× observed speedup',transform=ax.transAxes,fontsize=10,color=BLUE)
    c=s['session_counters']
    assert c['resumed']+c['started']==c['turns']
    assert abs(100*c['tokens_reused']/(c['tokens_prefilled']+c['tokens_reused'])-c['reuse_pct'])<.1
    fig.text(.06,.17,f"Session-enabled counters: {c['resumed']} resumed turns / {c['turns']} total · {c['tokens_reused']:,} prompt tokens reused ({c['reuse_pct']}%)",fontsize=11)
    fig.text(.06,.10,'Integration comparison: session revision also offloads weights during sleep; full-step gain is not isolated to KV.',fontsize=10,color='#475569')
    fig.text(.06,.05,'Historical single-seed observations; no uncertainty interval or training-quality claim. Not measured on the current revision.',fontsize=9,color='#64748B')
    save(fig,'grpo-session-kv')
    b=d['batching']; a=b['serial']; c=b['batched']
    assert b['settings']['sessions_on_both_arms'] and b['settings']['same_build_reported']
    fig,axes=plt.subplots(1,3,figsize=(14,5.5))
    fig.subplots_adjust(left=.075,right=.98,top=.68,bottom=.29,wspace=.43)
    fig.suptitle('Let concurrent agent turns share an engine step',x=.055,y=.96,ha='left',fontsize=22,fontweight='bold')
    fig.text(.055,.885,'CodeScout GRPO rollout  |  8 prompts × 8 samples  |  Session KV on in both arms  |  Agent concurrency 4',fontsize=11)
    panels=[([a['generation_span_s'],c['generation_span_s']],'Rollout generation span ↓','Wall time (s)',',.2f'),
            ([a['batching']['steps'],c['batching']['steps']],'Engine execution steps ↓','Engine steps',',.0f'),
            ([a['batching']['mean_batch'],c['batching']['mean_batch']],'Mean requests per step ↑','Requests / engine step','.2f')]
    for ax,(vals,title,unit,fmt) in zip(axes,panels): bars(ax,vals,['Serial','Batched'],title,unit,fmt)
    gain=a['generation_span_s']/c['generation_span_s']
    fig.text(.055,.17,f"{gain:.2f}× observed generation-span speedup · Request-step totals: {a['batching']['batched_requests']:,} vs {c['batching']['batched_requests']:,} · Peak batch: 1 → 4",fontsize=11)
    fig.text(.055,.10,'Same-build control reported; one run per arm, different sampled trajectories (299 vs 297 turns). No error bars.',fontsize=10,color='#475569')
    fig.text(.055,.05,'Span: first turn admitted → last turn retired, including tools. Not full training-step time; do not multiply by the session-KV ratio.',fontsize=9,color='#64748B')
    save(fig,'grpo-batching')
    (out/'rollout-figure-data.json').write_text(json.dumps({'source':'bench/results/rollout-features/observations.json','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'session_timing_s':times,'batching':{'serial':a,'batched':c}},indent=2)+'\n')
    print('Generated 2 rollout feature figures from archived observations')

if __name__=='__main__': main()
