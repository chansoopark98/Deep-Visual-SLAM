"""Dataset lookup for the apps (validation stage V0).

Loaders are imported lazily: the Redwood path must keep working in environments without PyYAML, which the
TUM-VI and EuRoC loaders need.
"""
import importlib
from pathlib import Path

DATASETS = ('redwood', 'tumvi', 'euroc')


def _module(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f'unknown dataset {dataset!r}; choose from {DATASETS}')
    return importlib.import_module(f'dvslam.data.{dataset}')


def load_sequence(dataset: str, scene: str, split: str, root: Path | str | None = None,
                  time_offset_s: float = 0.0, load_images: bool = True):
    """time_offset_s: GT lookup offset in seconds (TUM-VI / EuRoC; not measured yet, plan T.1). load_images=False
    skips the rectified-image cache: evaluation needs only the GT."""
    m = _module(dataset)
    if dataset == 'redwood':
        if time_offset_s:
            raise ValueError('Redwood has no timestamps: a GT time offset cannot be applied')
        return m.load_sequence(scene, split, root)
    return m.load_sequence(scene, split, root, time_offset_s=time_offset_s, load_images=load_images)


def scenes(dataset: str, split: str, root: Path | str | None = None) -> list[str]:
    """Scenes of a split. Redwood: its five rooms. TUM-VI / EuRoC: the sequences of the split present on disk."""
    m = _module(dataset)
    return list(m.SCENES) if dataset == 'redwood' else list(m.available_scenes(split, root))


def expected_scenes(dataset: str, split: str) -> list[str]:
    """The split as defined, whether or not every sequence is on disk (TUM-VI room5-6 are not downloaded)."""
    m = _module(dataset)
    return list(m.SCENES) if dataset == 'redwood' else list(m.SPLITS[split])


def default_root(dataset: str) -> Path:
    return _module(dataset).DEFAULT_ROOT


def frame_rate(dataset: str) -> tuple[float, bool]:
    """(Hz, verified). Redwood's 30 Hz is still unverified (see redwood.py)."""
    m = _module(dataset)
    return m.FRAME_RATE_HZ, getattr(m, 'FRAME_RATE_VERIFIED', True)
