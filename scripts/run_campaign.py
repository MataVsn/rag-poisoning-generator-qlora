"""Bounded single-GPU campaign. Keeps 4B run, benchmarks 14B, evaluates matched cases.

Start with runtime.env loaded in a persistent terminal/session. No cloud API or
public RAG requests are made. Base model cache may be volatile; adapters are not.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def choose_trial(trials,total_gib):
    eligible=[t for t in trials if t.get('success') and t['peak_reserved_gib']<=total_gib*.88]
    if not eligible:raise RuntimeError('No trial passed with GPU memory headroom. Read benchmark logs.')
    return max(eligible,key=lambda t:t['steps_per_second'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=3.75);p.add_argument('--generation-limit',type=int,default=64);a=p.parse_args()
    if not 1<=a.hours<=4:raise SystemExit('Use a campaign budget between 1 and 4 hours.')
    if a.generation_limit<=0:raise SystemExit('generation-limit must be positive')
    target=ROOT/'outputs/campaign-14b'
    target.mkdir(parents=True,exist_ok=True)
    claim=target/'started.json'
    with claim.open('x') as f:json.dump({'utc':datetime.now(timezone.utc).isoformat(),'hours':a.hours},f)
    started=time.monotonic();deadline=started+a.hours*3600
    env4=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',HF_HUB_DISABLE_XET='1')
    env14=dict(env4,HF_HOME='/dev/shm/rag-endterm-hf-14b')
    journal=[]
    def status(stage,**extra):
        record={'stage':stage,'elapsed_seconds':round(time.monotonic()-started,2),'remaining_seconds':round(deadline-time.monotonic(),2),**extra}
        journal.append(record);(target/'status.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));(target/'journal.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2));print(json.dumps(record,ensure_ascii=False),flush=True)
    def run(label,args,env=env14,allow_fail=False):
        if deadline-time.monotonic()<30:raise RuntimeError('Campaign wall-time budget exhausted')
        status(label)
        with (target/f'{label}.log').open('w') as log:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()))
        if result.returncode and not allow_fail:raise RuntimeError(f'{label} failed ({result.returncode}); read '+str(target/f'{label}.log'))
        return result.returncode==0
    def generate(label,model_config,adapter=None,env=env14):
        args=['scripts/generate_candidates.py','--model-config',model_config,'--input','data/generation_prompts/validation.jsonl','--output',str(target/f'{label}.jsonl'),'--limit',str(a.generation_limit),'--max-new-tokens','256']
        if adapter:args+=['--adapter',adapter]
        run(label,args,env)
    try:
        # Leave the existing experiment intact and avoid GPU contention.
        status('waiting_for_4b')
        marker=ROOT/'outputs/qwen3-4b-qlora/training_manifest.json'
        while not marker.exists():
            current=ROOT/'outputs/remote/latest-log.txt'
            if current.exists():
                code=Path(ROOT/current.read_text().strip()).with_suffix('.status')
                if code.exists() and code.read_text().strip().startswith('exit_code=') and code.read_text().strip()!='exit_code=0':raise RuntimeError('Existing 4B run failed; refusing to ignore failure')
            if deadline-time.monotonic()<3600:raise RuntimeError('Not enough budget left after waiting for 4B')
            time.sleep(15)
        # The manifest is written immediately before the 4B process exits.
        for _ in range(12):
            if subprocess.run(['tmux','has-session','-t','rag-endterm-training'],capture_output=True).returncode!=0:break
            time.sleep(5)
        else:raise RuntimeError('4B session still active after manifest; refusing GPU contention')
        run('validate',['scripts/validate_datasets.py'],env4)
        generate('base-4b','configs/model.json',env=env4)
        generate('tuned-4b','configs/model.json','outputs/qwen3-4b-qlora/adapter',env4)
        run('prefetch-14b',['scripts/prefetch_model.py','--model-config','configs/model_14b.json'])
        run('tokens-14b',['scripts/check_tokens.py','--model-config','configs/model_14b.json','--max-length','512','--report',str(target/'token-report-14b.json')])
        template=json.loads((ROOT/'configs/qlora_14b.json').read_text())
        trials=[]
        for batch,gc in [(4,True),(8,True),(4,False),(2,True)]:
            name=f'b{batch}-gc{int(gc)}';cfg=dict(template,batch_size=batch,gradient_accumulation_steps=8//batch,gradient_checkpointing=gc,output_dir=f'outputs/campaign-14b/benchmark-{name}')
            path=target/f'config-{name}.json';path.write_text(json.dumps(cfg,indent=2))
            passed=run('benchmark-'+name,['scripts/train_qlora.py','--config',str(path),'--smoke-steps','8','--benchmark-only'],allow_fail=True)
            record={'name':name,'success':passed,'config':cfg}
            if passed:
                result=ROOT/cfg['output_dir']/'smoke'
                manifest=json.loads((result/'training_manifest.json').read_text());metrics=json.loads((result/'training_metrics.json').read_text())
                record.update(peak_reserved_gib=manifest['peak_cuda_reserved_gib'],steps_per_second=metrics['train_steps_per_second'])
            trials.append(record);(target/'benchmarks.json').write_text(json.dumps(trials,indent=2))
        total_mib=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        best=choose_trial(trials,total_mib/1024)
        selected=dict(best['config'],output_dir='outputs/qwen3-14b-qlora')
        # Leave 30 minutes for generation, packaging and bounded operation.
        selected['max_training_seconds']=min(template['max_training_seconds'],max(0,int(deadline-time.monotonic()-1800)))
        if selected['max_training_seconds']<900:raise RuntimeError('Insufficient remaining budget for main training')
        selected_path=target/'selected-config.json';selected_path.write_text(json.dumps(selected,indent=2))
        generate('base-14b','configs/model_14b.json')
        selected['max_training_seconds']=min(selected['max_training_seconds'],max(0,int(deadline-time.monotonic()-1800)))
        if selected['max_training_seconds']<900:raise RuntimeError('Baseline generation left insufficient training budget')
        selected_path.write_text(json.dumps(selected,indent=2))
        run('train-14b',['scripts/train_qlora.py','--config',str(selected_path)])
        generate('tuned-14b','configs/model_14b.json','outputs/qwen3-14b-qlora/adapter')
        compare=['scripts/compare_generators.py','--output-dir',str(target/'comparison')]
        for label in ['base-4b','tuned-4b','base-14b','tuned-14b']:compare+=['--input',label+'='+str(target/f'{label}.jsonl')]
        run('compare-generators',compare)
        status('completed',test_split_used=False,rag_executed=False)
    except Exception as exc:
        status('failed',error=str(exc));raise

if __name__=='__main__':main()
