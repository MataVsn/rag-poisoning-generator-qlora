"""Package adapters and measured results, without cached base weights/checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = [p for p in (ROOT/'outputs').rglob('*') if p.is_file()
             and 'checkpoints' not in p.relative_to(ROOT/'outputs').parts]
    if not files:
        raise SystemExit('No outputs to export. Run an experiment first.')
    destination = ROOT/'exports'
    destination.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = destination/f'results-{stamp}.zip'
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.relative_to(ROOT))
        artifacts=[*sorted((ROOT/'configs').glob('*.json'))]
        for name in ['data/checksums.json','data/dataset_stats.json','docs/DATA_CARD_RU.md',
                     'docs/CAMPAIGN_14B_RU.md','sources/manifest.json','sources/poisonedrag/LICENSE']:
            if (ROOT/name).is_file():artifacts.append(ROOT/name)
        for path in artifacts:
            archive.write(path,path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    print(target)
    print('Copy this archive to your laptop before deleting the instance.')

if __name__ == '__main__':
    main()
