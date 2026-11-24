"""Sample GPU utilization/power alongside campaign stages; no CUDA context."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    target=ROOT/'outputs/campaign-14b'
    deadline=time.monotonic()+4*3600
    with (target/'gpu-telemetry.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['utc','stage','utilization_percent','memory_used_mib','power_w'])
        while time.monotonic()<deadline:
            try:stage=json.loads((target/'status.json').read_text())['stage']
            except (FileNotFoundError,json.JSONDecodeError):stage='unknown'
            result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if result.returncode==0:
                values=next(csv.reader(result.stdout.splitlines()))
                writer.writerow([datetime.now(timezone.utc).isoformat(),stage,*[v.strip() for v in values]]);f.flush()
            if stage in ['completed','failed']:break
            time.sleep(5)

if __name__=='__main__':main()
