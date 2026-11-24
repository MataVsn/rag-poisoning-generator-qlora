"""Download small published research artifacts at a fixed Git revision."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
SHA='f660d72174f06b13fae5163ce656e7b235db858f'
BASE=f'https://raw.githubusercontent.com/sleeepeer/PoisonedRAG/{SHA}/'


def fetch(url):
    return urlopen(Request(url,headers={'User-Agent':'QAIRU-GenAI-coursework-data-preparation'}),timeout=60).read()


def main():
    records=[]
    for path in ['LICENSE','README.md']+[f'results/adv_targeted_results/{d}.json' for d in ['hotpotqa','msmarco','nq']]:
        target=ROOT/'sources'/'poisonedrag'/('raw/'+Path(path).name if path.endswith('.json') else Path(path).name)
        target.parent.mkdir(parents=True,exist_ok=True)
        payload=fetch(BASE+path);target.write_bytes(payload)
        records.append({'url':BASE+path,'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})
        print(target.relative_to(ROOT),len(payload),'bytes')
    model='Qwen/Qwen3-4B-Instruct-2507'
    info=json.loads(fetch('https://huggingface.co/api/models/'+model))
    (ROOT/'configs'/'model.json').write_text(json.dumps({'model_id':model,'revision':info['sha'],'license_declared':'apache-2.0','weights_downloaded':False,'training_verified_on_rented_gpu':False},ensure_ascii=False,indent=2))
    (ROOT/'sources'/'manifest.json').write_text(json.dumps({'source':'sleeepeer/PoisonedRAG','git_revision':SHA,'artifacts':records,'paper':'https://www.usenix.org/conference/usenixsecurity25/presentation/zou-wei','scope':'Published PoisonedRAG examples, not AuthChain annotations. Original license included; upstream QA datasets have their own terms.'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
