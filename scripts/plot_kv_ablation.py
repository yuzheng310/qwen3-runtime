"""Render four-arm KV diagnostic observations; requires matplotlib."""
import argparse, json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=['none','session-only','apc-only','session+apc']
LABELS=['No reuse','Session KV','APC','Session KV + APC']
COLORS=['#94A3B8','#2563EB','#D97706','#0D9488']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path);a=p.parse_args()
 src=a.root/'bench/results/kv-ablation/observations.json';d=json.loads(src.read_text());out=a.out or a.root/'docs/assets/performance';out.mkdir(parents=True,exist_ok=True)
 assert d['evidence_class']=='historical_diagnostic_not_validated_benchmark'
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#172B4D','axes.labelcolor':'#172B4D','svg.hashsalt':'kv-ablation-v1'})
 def save(fig,name):
  fig.savefig(out/f'{name}.png',dpi=180,facecolor='white');fig.savefig(out/f'{name}.svg',metadata={'Date':None},facecolor='white')
  s=out/f'{name}.svg';s.write_text('\n'.join(line.rstrip() for line in s.read_text().splitlines())+'\n');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5.8));fig.subplots_adjust(left=.17,right=.97,top=.73,bottom=.29,wspace=.5)
 fig.suptitle('Four KV policies on sequential conversation replay',x=.055,y=.96,ha='left',fontsize=20,fontweight='bold')
 fig.text(.055,.875,'Historical diagnostic · CodeScout-4B · 100 tasks / 471 requests · spec=0 · no tool execution',fontsize=11)
 for ax,key,scale,title,unit,fmt in zip(axs,['seconds_per_task','later_turn_prefill_tokens'],[1,1e6],['Recorded replay time ↓','Later-turn prefill work ↓'],['Seconds / task','Million tokens'],['.3f','.3f']):
  values=[r[key]/scale for r in d['sequential']];ax.barh(LABELS,values,color=COLORS,height=.55);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.21);ax.set_title(title,loc='left',fontweight='bold',pad=14);ax.set_xlabel(unit);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
  for i,v in enumerate(values):ax.text(v+max(values)*.02,i,format(v,fmt),va='center')
 fig.text(.055,.16,'Source flags: dirty=true, forced_length_ok=false. Warmup differs: 1 without APC, 0 with APC.',fontsize=11,color='#9A3412')
 fig.text(.055,.09,'Single run per arm; matched recorded output lengths do not override failed validity flags. No error bars.',fontsize=10)
 fig.text(.055,.04,'Diagnostic evidence only: neither current-release performance nor proof that Session KV beats APC.',fontsize=10,color='#64748B');save(fig,'kv-four-arm-replay')
 fig,axs=plt.subplots(1,3,figsize=(15,5.8));fig.subplots_adjust(left=.06,right=.97,top=.68,bottom=.27,wspace=.3)
 fig.suptitle('Four KV policies as conversation concurrency rises',x=.06,y=.96,ha='left',fontsize=21,fontweight='bold')
 fig.text(.06,.885,'Historical diagnostic · 16 conversations / 67 turns · 12,013-block pool · spec=0 · no tool dwell time',fontsize=11)
 for arm,label,color,marker in zip(ARMS,LABELS,COLORS,['o','s','^','D']):
  rows=sorted([r for r in d['concurrent'] if r['arm']==arm],key=lambda r:r['concurrency'])
  assert len(rows)==4 and all(r['turns_completed']==67 and r['num_preemptions']==0 for r in rows)
  for ax,key,scale in zip(axs,['wall_s','later_turn_prefill_tokens','evicted'],[1,1000,1]):
   ax.plot([r['concurrency'] for r in rows],[r[key]/scale for r in rows],label=label,color=color,marker=marker,lw=2,ms=6)
 for ax,title,y in zip(axs,['Replay wall time ↓','Later-turn prefill work ↓','Session evictions'],['Seconds','Thousand tokens','Recorded eviction count']):
  ax.set_title(title,loc='left',fontweight='bold',pad=13);ax.set_xlabel('Conversation concurrency');ax.set_ylabel(y);ax.set_xticks([1,4,8,16]);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.055,.835),ncol=4,frameon=False)
 fig.text(.06,.15,'At c=16: Session KV + APC prefills 161,482 later-turn tokens vs 221,445 with Session KV; APC alone: 161,811.',fontsize=10)
 fig.text(.06,.085,'All cells report zero scheduler preemptions. Session evictions are a different counter; zero for arms without sessions.',fontsize=10)
 fig.text(.06,.035,'One run/cell, environment metadata incomplete. No parked-pool saturation demonstrated; no live GRPO speedup claimed.',fontsize=10,color='#64748B');save(fig,'kv-four-arm-concurrency')
 (out/'kv-ablation-figure-data.json').write_text(json.dumps({'source':'bench/results/kv-ablation/observations.json','sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'sequential':d['sequential'],'concurrent':d['concurrent']},indent=2)+'\n')
 print('Validated 4 sequential arms and 16 concurrency cells; generated two figures')
if __name__=='__main__':main()
