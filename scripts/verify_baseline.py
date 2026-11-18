"""Write or verify MANIFEST.json for a frozen target-metric baseline run (plan v3 item 0.1).

  python scripts/verify_baseline.py                 # verify every sha256 in the manifest, exit 1 on mismatch
  python scripts/verify_baseline.py --write ...     # create the manifest (run once when freezing)
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / 'baselines/pairs_eval/20261118-125421'
SCENES = ['apartment', 'bedroom', 'boardroom', 'lobby', 'loft']


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(baseline: Path, weights: Path, data_root: Path, eval_commit: str, env: dict) -> None:
    # PNG plots are not tracked in git, so they are left out of the manifest.
    files = {p.name: sha256_of(p) for p in sorted(baseline.iterdir())
             if p.name != 'MANIFEST.json' and p.suffix != '.png'}
    data = {}
    for scene in SCENES:
        scene_dir = data_root / 'test' / scene
        gt = scene_dir / f'{scene}.json'
        data[scene] = {
            'gt_json': str(gt.relative_to(data_root)),
            'gt_json_sha256': sha256_of(gt),
            'num_images': len(list((scene_dir / 'image').glob('*.jpg'))),
            'num_depth_png': len(list((scene_dir / 'depth').glob('*.png'))),
        }
    manifest = {
        'run_id': baseline.name,
        'frozen_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'eval_code_commit': eval_commit,
        'eval_code_note': 'vo/eval_redwood.py mtime (12:50) precedes the run (12:54); the commit holds that exact file.',
        'weights': {'path': str(weights.relative_to(REPO_ROOT)), 'sha256': sha256_of(weights)},
        'data_root': str(data_root),
        'data': data,
        'environment': env,
        'results': {'item03_mape_pct': 40.0591, 'item04_rpy_mean_deg': 0.096318},
        'files': files,
    }
    (baseline / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


def verify(baseline: Path) -> list[str]:
    manifest = json.loads((baseline / 'MANIFEST.json').read_text())
    errors = []
    for name, digest in manifest['files'].items():
        p = baseline / name
        if not p.exists():
            errors.append(f'missing baseline file {name}')
        elif sha256_of(p) != digest:
            errors.append(f'sha256 mismatch: {name}')
    weights = REPO_ROOT / manifest['weights']['path']
    if not weights.exists():
        errors.append(f'missing weights {weights}')
    elif sha256_of(weights) != manifest['weights']['sha256']:
        errors.append('sha256 mismatch: weights')
    data_root = Path(manifest['data_root'])
    for scene, info in manifest['data'].items():
        gt = data_root / info['gt_json']
        if not gt.exists():
            errors.append(f'missing GT {gt}')
        elif sha256_of(gt) != info['gt_json_sha256']:
            errors.append(f'sha256 mismatch: GT {scene}')
        n = len(list((data_root / 'test' / scene / 'image').glob('*.jpg')))
        if n != info['num_images']:
            errors.append(f'{scene}: {n} images, manifest says {info["num_images"]}')
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--data-root', type=Path, default=Path('/mnt/backup/SLAM/redwood'))
    ap.add_argument('--eval-commit')
    ap.add_argument('--env', type=json.loads, default={}, help='JSON dict of python/torch/cuda/device at run time')
    args = ap.parse_args()
    if args.write:
        write_manifest(args.baseline, args.weights.resolve(), args.data_root, args.eval_commit, args.env)
    errors = verify(args.baseline)
    for e in errors:
        print('FAIL', e)
    print('OK: all sha256 match' if not errors else f'{len(errors)} mismatches')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
