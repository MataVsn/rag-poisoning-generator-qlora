"""Prepare local evaluation jobs; no insertion into any live RAG or website."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Output already exists.')
    cases={r['case_id']:r for r in [json.loads(s) for s in (ROOT/'data/rag'/a.split/'cases.jsonl').open()]}
    jobs=[];unsupported=0
    for line in a.candidates.open():
        r=json.loads(line);case=cases.get(r['example_id'])
        if case is None:
            unsupported+=1;continue
        jobs.append({'case_id':case['case_id'],'group_id':case['group_id'],'question':case['question'],'scenario_type':case['scenario_type'],'clean_corpus':str(Path('data/rag')/a.split/'clean_corpus.jsonl'),'reset_to_clean_snapshot_before_case':True,'added_documents':[{'id':case['case_id']+'-generated','text':r['generated_document'],'metadata':{'synthetic':True,'origin':'experimental_generator','case_id':case['case_id']}}],'conditions':['clean_without_added_document','none','prompt','guard'],'evaluation_gold':{'original_answer':case['original_answer'],'target_answer':case['target_answer']},'candidate_generation':{'model':r['model'],'adapter':r['adapter']},'required_evidence':['run_id','snapshot_hash','retrieved_document_ids','passed_document_ids','response','outcome','latency_ms'],'status':'job_description_not_executed'})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    print(json.dumps({'jobs':len(jobs),'skipped_without_clean_corpus':unsupported,'network_requests':0,'rag_executed':False},indent=2))
if __name__=='__main__':main()
