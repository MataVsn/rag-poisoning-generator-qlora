"""Fetch immutable safetensors/tokenizer files. Does not execute model code."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.model_config.read_text())
    snapshot=snapshot_download(config['model_id'],revision=config['revision'],allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','README.md','LICENSE'],max_workers=2)
    verified=[]
    for name,expected in config.get('weight_sha256',{}).items():
        digest=hashlib.sha256()
        with (Path(snapshot)/name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected:raise RuntimeError('Weight SHA256 mismatch: '+name)
        verified.append(name)
    print(json.dumps({'snapshot':snapshot,'model':config,'download_complete':True,'weight_sha256_verified':verified},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
