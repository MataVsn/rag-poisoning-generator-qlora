"""Surface checks only. Does not claim ASR or factual correctness."""
import argparse,json,re,unicodedata
from pathlib import Path

def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    n=nonempty=target=original=at_limit=0;per_source={}
    for line in a.input.open():
        r=json.loads(line);n+=1;t=norm(r.get('generated_document',''));e=r['evaluation']
        nonempty+=bool(t);target+=norm(e['target_answer']) in t;original+=norm(e['original_answer']) in t
        at_limit+=r['output_tokens']>=r['generation']['max_new_tokens']
        per_source[r['source']]=per_source.get(r['source'],0)+1
    report={'n':n,'nonempty_documents':nonempty,'target_literal_present':target,'original_literal_present':original,'outputs_at_token_limit':at_limit,'by_source':per_source,'interpretation':'Literal presence only: negation, quotation and ambiguity need review. These are generator checks, NOT retrieval success or RAG ASR. Clean controls deliberately have original==target.'}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
