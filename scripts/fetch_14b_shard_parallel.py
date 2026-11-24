"""Fallback for a stalled HF shard: ranged HTTPS + upstream SHA256 verification.

Only fetches the fixed public model/revision below. Never executes downloaded code.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import requests

SIZE=4931923327
SHA='7fc7e11e126568f258d2d0c113477b481323b455a01ad248a55eb693a9c96c12'
CACHE=Path('/dev/shm/rag-endterm-hf-14b/hub/models--unsloth--Qwen3-14B-bnb-4bit/blobs')
URL='https://huggingface.co/unsloth/Qwen3-14B-bnb-4bit/resolve/b455922269e4678c84f854fe38a6993d73614a1e/model-00002-of-00002.safetensors'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    blob=CACHE/SHA
    if blob.exists() and digest(blob)==SHA:
        print('Verified shard already cached',flush=True);return
    if shutil.disk_usage(CACHE).free<SIZE+512*1024*1024:raise RuntimeError('Insufficient RAM-cache capacity for fallback')
    target=CACHE/(SHA+'.parallel')
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
    os.ftruncate(fd,SIZE)
    chunk=64*1024*1024
    url=URL+'?download=true&fresh='+str(time.time_ns())
    started=time.monotonic()
    def fetch(start):
        end=min(SIZE-1,start+chunk-1)
        for attempt in range(4):
            try:
                offset=start
                with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,30)) as response:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':raise RuntimeError('Range response mismatch')
                    for data in response.iter_content(1024*1024):
                        if offset+len(data)>end+1:raise RuntimeError('Oversized range')
                        if os.pwrite(fd,data,offset)!=len(data):raise RuntimeError('Short write')
                        offset+=len(data)
                if offset!=end+1:raise RuntimeError('Incomplete range')
                return end-start+1
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    try:
        done=0
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending=[pool.submit(fetch,start) for start in range(0,SIZE,chunk)]
            for future in as_completed(pending):
                done+=future.result()
                print(f'{done}/{SIZE} bytes; {done/max(1,time.monotonic()-started)/1e6:.1f} MB/s',flush=True)
    finally:os.close(fd)
    if digest(target)!=SHA:raise RuntimeError('SHA256 mismatch; cache was not modified')
    # Stop only our old downloader before replacing its incomplete shard.
    subprocess.run(['tmux','kill-session','-t','rag14b-download'],capture_output=True)
    target.replace(blob)
    incomplete=CACHE/(SHA+'.incomplete')
    if incomplete.exists():incomplete.unlink()
    print('Upstream SHA256 verified; HF blob cache repaired',flush=True)

if __name__=='__main__':main()
