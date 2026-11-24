"""Download tokenizer only and check every chosen example, before renting/training."""
import argparse,json,math
from pathlib import Path
from transformers import AutoTokenizer
from tokenize_data import encode_example
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mixed','synthetic_ru','poisonedrag'],default='mixed');p.add_argument('--max-length',type=int,default=1024);p.add_argument('--model-config',type=Path,default=ROOT/'configs/model.json');p.add_argument('--report',type=Path);a=p.parse_args()
    m=json.loads(a.model_config.read_text())
    t=AutoTokenizer.from_pretrained(m['model_id'],revision=m['revision'],trust_remote_code=False)
    report={'model':m,'dataset':a.dataset,'max_length':a.max_length,'splits':{}}
    failed=[]
    for split in ['train','validation','test']:
        lengths=[];completion=[]
        for line in (ROOT/'data'/'sft'/a.dataset/f'{split}.jsonl').open():
            row=json.loads(line)
            try:e=encode_example(t,row,10**9,m.get('chat_template_kwargs'))
            except ValueError as exc:failed.append(str(exc));continue
            lengths.append(len(e['input_ids']));completion.append(sum(v!=-100 for v in e['labels']))
            if lengths[-1]>a.max_length:failed.append(f'{row["example_id"]}: {lengths[-1]} tokens')
        s=sorted(lengths)
        report['splits'][split]={'n':len(s),'max_tokens':max(s),'p50_tokens':s[len(s)//2],'p95_tokens':s[min(len(s)-1,math.ceil(len(s)*.95)-1)],'total_tokens':sum(s),'completion_tokens':sum(completion),'over_limit':sum(v>a.max_length for v in s)}
    report['errors']=failed
    target=a.report or ROOT/'data'/f'token_report_{a.dataset}.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
    if failed:raise SystemExit('Token validation failed. See '+str(target))
if __name__=='__main__':main()
