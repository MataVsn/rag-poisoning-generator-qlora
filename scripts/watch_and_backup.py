"""Run on the laptop: wait for terminal campaign status, then fetch results.

Never uploads keys or changes the rental's lifecycle. The laptop must stay on.
"""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
import os
KEY=Path(os.environ.get('GPU_SSH_KEY',str(Path.home()/'.ssh/id_ed25519')))
HOST=os.environ['GPU_SSH_HOST']  # e.g. root@<gpu-host>
PORT=os.environ.get('GPU_SSH_PORT','22')
REMOTE='/workspace/rag-genai-endterm'
DEST=Path(os.environ.get('RESULTS_DEST',str(ROOT/'exports')))

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+4.25*3600
    ssh=['ssh','-i',str(KEY),'-p',PORT,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST]
    while time.monotonic()<deadline:
        probe=subprocess.run([*ssh,f'cat {REMOTE}/outputs/campaign-14b/status.json'],capture_output=True,text=True,timeout=35)
        if probe.returncode==0:
            try:state=json.loads(probe.stdout)
            except json.JSONDecodeError:state={}
            (DEST/'remote-status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
            print(time.strftime('%Y-%m-%d %H:%M:%S'),state.get('stage'),flush=True)
            if state.get('stage') in ['completed','failed']:
                export=subprocess.run([*ssh,f'cd {REMOTE} && .venv/bin/python scripts/export_results.py'],capture_output=True,text=True,timeout=900)
                if export.returncode!=0:raise RuntimeError('Remote export failed; results still remain on server')
                archive=next((s.strip() for s in export.stdout.splitlines() if s.startswith(REMOTE+'/exports/') and s.endswith('.zip')),None)
                if archive is None:raise RuntimeError('No expected export path returned')
                local=DEST/Path(archive).name
                subprocess.run(['scp','-i',str(KEY),'-P',PORT,'-o','IdentitiesOnly=yes','-o','BatchMode=yes',HOST+':'+archive,str(local)],check=True,timeout=1800)
                import zipfile
                with zipfile.ZipFile(local) as z:
                    if z.testzip() is not None:raise RuntimeError('Local archive integrity check failed')
                (DEST/'backup-status.json').write_text(json.dumps({'backup_verified':True,'archive':str(local),'campaign_stage':state['stage'],'instance_stopped':False},indent=2))
                print('Verified local backup:',local,flush=True)
                return
        time.sleep(60)
    raise SystemExit('Backup watcher budget expired; fetch server results manually.')

if __name__=='__main__':main()
