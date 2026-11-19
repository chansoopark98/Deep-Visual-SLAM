"""Depth prior read from maps cached on disk (EXP-003): `prior=cache prior_ckpt=<dir>` in KLTWindowOdometry.

Layout of <dir>: <scene>.npy (n, H, W) float32 metres, row i = depth of frame i, and <scene>.paths.txt with the image
path of row i on line i (written by scripts/exp003_cache_depth.py from the same sequence loader). A frame is looked
up by its image path, so one directory serves every scene and a path that is not cached is an error, never a
silent mismatch. The maps are those of any DepthPrior (e.g. SSLDepth), computed once, so tracker sweeps run
without a GPU and reproduce the live prior exactly as far as the network inference is deterministic.
Provenance: <scene>.meta.json (write_meta; also used by the flow cache, klt_flow.py) records the model, the
checkpoint path and its sha256 (when there is a checkpoint file), the map H x W and the creation time. A loader
given hw=(H, W) raises ValueError when a scene's recorded H x W differs; a scene without meta.json (caches written
before 2026-11-20) loads unchecked, as before. The meta of every loaded scene is kept in `.meta[scene]`.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dvslam.odometry.base import Frame


def write_meta(root: Path, scene: str, hw: tuple[int, int], model: str, ckpt: str | Path | None = None) -> None:
    """<root>/<scene>.meta.json: what produced the cache of `scene` (maps of hw = (H, W) pixels)."""
    meta = {'model': model, 'ckpt': None if ckpt is None else str(ckpt),
            'ckpt_sha256': None if ckpt is None else hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
            'hw': [int(hw[0]), int(hw[1])], 'created': datetime.now().astimezone().isoformat(timespec='seconds')}
    (Path(root) / f'{scene}.meta.json').write_text(json.dumps(meta, indent=1) + '\n')


def write_cache(root: Path, scene: str, paths: list, depths: np.ndarray, *, model: str,
                ckpt: str | Path | None = None) -> None:
    """depths (n, H, W) float32 with row i for paths[i]; model / ckpt go to <scene>.meta.json."""
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f'{scene}.npy', np.asarray(depths, np.float32))
    (root / f'{scene}.paths.txt').write_text(''.join(f'{p}\n' for p in paths))
    write_meta(root, scene, np.shape(depths)[1:], model, ckpt)


class CachedDepth:
    name = 'cache'

    def __init__(self, root: str | Path, hw: tuple[int, int] | None = None):
        self.root, self.hw = Path(root), hw            # hw: (H, W) the tracker expects; None = no check
        self._index: dict[str, tuple[str, int]] | None = None
        self._maps: dict[str, np.ndarray] = {}
        self.meta: dict[str, dict] = {}

    def _build_index(self) -> dict[str, tuple[str, int]]:
        index = {}
        for f in sorted(self.root.glob('*.paths.txt')):
            scene = f.name[:-len('.paths.txt')]
            for i, p in enumerate(f.read_text().splitlines()):
                index[p] = (scene, i)
        if not index:
            raise FileNotFoundError(f'no *.paths.txt in {self.root}')
        return index

    def predict(self, frame: Frame) -> tuple[np.ndarray, None]:
        if self._index is None:
            self._index = self._build_index()
        key = str(frame.image_path)
        if key not in self._index:
            raise KeyError(f'{key} is not in the depth cache {self.root}')
        scene, i = self._index[key]
        if scene not in self._maps:
            m = self.root / f'{scene}.meta.json'
            if m.exists():
                self.meta[scene] = json.loads(m.read_text())
                if self.hw is not None and tuple(self.meta[scene]['hw']) != tuple(self.hw):
                    raise ValueError(f'depth cache {m}: maps are {" x ".join(map(str, self.meta[scene]["hw"]))}, '
                                     f'the tracker needs {self.hw[0]} x {self.hw[1]}')
            self._maps[scene] = np.load(self.root / f'{scene}.npy', mmap_mode='r')
        return np.array(self._maps[scene][i], np.float32), None
