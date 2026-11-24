"""Compare matched generated documents; report surface checks, never RAG ASR."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from score_candidates import norm

def summarize(rows):
    conflicts=[r for r in rows if r['evaluation']['scenario_type']!='clean_control']
    times=sorted(r['duration_ms'] for r in rows)
    return {'n':len(rows),'conflict_n':len(conflicts),
            'nonempty':sum(bool(r['generated_document'].strip()) for r in rows),
            'conflict_target_literal_present':sum(norm(r['evaluation']['target_answer']) in norm(r['generated_document']) for r in conflicts),
            'outputs_at_token_limit':sum(r['output_tokens']>=r['generation']['max_new_tokens'] for r in rows),
            'latency_p50_ms':statistics.median(times),
            'latency_p95_ms':times[max(0,math.ceil(.95*len(times))-1)],
            'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rows)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',action='append',required=True,help='label=path.jsonl');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    summaries={};detail=[];reference=None
    for spec in a.input:
        label,name=spec.split('=',1)
        if label in summaries:raise ValueError('Duplicate label')
        rows=[json.loads(line) for line in Path(name).read_text().splitlines() if line.strip()]
        if not rows:raise ValueError('Empty input: '+name)
        ids=[r['example_id'] for r in rows]
        if len(set(ids))!=len(ids):raise ValueError('Repeated example IDs')
        if reference is None:reference=set(ids)
        elif reference!=set(ids):raise ValueError('Models must be compared on matching cases')
        summaries[label]=summarize(rows)
        for r in rows:
            detail.append({'variant':label,'case_id':r['example_id'],'source':r['source'],'scenario_type':r['evaluation']['scenario_type'],'question':r['evaluation']['question'],'original_answer':r['evaluation']['original_answer'],'target_answer':r['evaluation']['target_answer'],'generated_document':r['generated_document'],'output_tokens':r['output_tokens'],'duration_ms':r['duration_ms']})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report={'variants':summaries,'scope':'Validation generator surface checks only. NOT RAG ASR, no RAG requests executed. Literal presence can be negated or quoted. Review documents manually.'}
    (a.output_dir/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (a.output_dir/'documents_for_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
