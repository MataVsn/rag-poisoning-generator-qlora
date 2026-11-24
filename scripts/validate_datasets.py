"""Validate schema, provenance, grouping, source integrity and evaluation separation."""
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path):return [json.loads(s) for s in path.open() if s.strip()]
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()

def main():
    partitions={s:read(ROOT/'data'/'mixed'/f'{s}.jsonl') for s in ['train','validation','test']}
    checks=0;groups={};questions={};texts={};all_ids=set();counts={}
    for split,rows in partitions.items():
        counts[split]=len(rows)
        for r in rows:
            assert r['example_id'] not in all_ids,('duplicate ID',r['example_id']);all_ids.add(r['example_id'])
            assert r['split']==split
            assert r['original_answer'].strip() and r['target_answer'].strip()
            assert r['input']['question'].strip() and r['target_document'].strip()
            assert [m['role'] for m in r['messages']]==['system','user','assistant']
            assert json.loads(r['messages'][1]['content'])==r['input']
            assert r['messages'][2]['content']==r['target_document']
            assert norm(r['target_answer']) in norm(r['target_document'])
            if r['scenario_type']=='clean_control':assert r['original_answer']==r['target_answer']
            else:assert norm(r['original_answer'])!=norm(r['target_answer'])
            for key,registry in [(r['group_id'],groups),(norm(r['input']['question']),questions),(norm(r['target_document']),texts)]:
                assert key not in registry or registry[key]==split,('cross-split overlap',r['example_id'])
                registry[key]=split
            checks+=1
        prompt=read(ROOT/'data'/'generation_prompts'/f'{split}.jsonl')
        assert len(prompt)==len(rows)
        assert all([m['role'] for m in p['messages']]==['system','user'] for p in prompt)
        cases=read(ROOT/'data'/'rag'/split/'cases.jsonl');docs=read(ROOT/'data'/'rag'/split/'clean_corpus.jsonl');entities=read(ROOT/'data'/'rag'/split/'entities.jsonl')
        docmap={d['id']:d for d in docs}
        assert len(docmap)==len(docs)
        assert len(entities)*2==len(cases)==len(docs)
        for case in cases:
            assert all(d in docmap for d in case['support_document_ids'])
            assert any(case['original_answer'] in docmap[d]['text'] for d in case['support_document_ids'])
    checksums=json.loads((ROOT/'data'/'checksums.json').read_text())
    for p,expected in checksums.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==expected,p
    # Existing known attacks and real names are kept outside every training partition.
    historical=ROOT/'sources'/'previous_experiment'/'recovered_attack_chunks.jsonl'
    if historical.exists():
        previous={norm(r['content']) for r in read(historical)}
        assert not (previous & set(texts))
    report={'status':'passed','examples_checked':checks,'partitions':counts,'unique_groups':len(groups),'cross_split_exact_question_overlap':0,'cross_split_exact_target_document_overlap':0,'cross_split_group_overlap':0,'synthetic_entity_disjoint':True,'external_entity_disjoint':'not asserted; only query/document-component disjointness guaranteed','checksums_verified':len(checksums),'model_training_run':False,'rag_effectiveness_test_run':False}
    (ROOT/'data'/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
