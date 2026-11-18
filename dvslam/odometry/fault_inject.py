"""Fault injection and a minimal supervisor around any odometry (plan v3 item 4.3, §4.2 F11).

  FaultInjectOdometry(base, faults=[Fault('black', start=200, length=10), Fault('exception', 500, 3)])

Faults (frame index in [start, start + length)):
  black      image replaced by an all-zero frame (camera blackout)
  freeze     image and depth replaced by the last frame before the fault (duplicated frames)
  starve     image replaced by flat mean grey plus a small smooth patch of the real image (corner starvation)
  exception  `base._solve_pair` (where='pnp') or `base._bundle_adjust` (where='ba') raises FaultInjected

Supervisor (what the plan asks of the runtime, F11): an exception out of `base.step` is swallowed; the frame is
emitted from the constant-velocity motion model (source motion_model) with state DEGRADED, or LOST once `lost_after`
substituted frames in a row have happened. The base object may be half-updated after an exception, so it is
re-initialised on the next frame (`base.reset` + a fresh first step) and the global scale state (g, n_m) and the last
increment are kept, as §4.2 says "g is kept". That re-initialisation frame is emitted as motion_model / BOOTSTRAP.
Frames the base itself emits from the motion model (blackout, starvation) pass through, but the same rule applies to
them: `lost_after` non-tracked frames in a row are reported LOST (the base may stay DEGRADED while it silently
restarts its window every frame).
"""
import dataclasses
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate

_EXC_TARGET = {'pnp': '_solve_pair', 'ba': '_bundle_adjust'}
_KEEP_ATTRS = ('g', 'n_m', 'last_inc_ba')    # scale / motion state restored after a supervisor re-init


class FaultInjected(RuntimeError):
    pass


@dataclass(frozen=True)
class Fault:
    kind: str                 # black | freeze | starve | exception
    start: int
    length: int
    where: str = 'pnp'        # exception only: pnp | ba

    def __post_init__(self):
        if self.kind not in ('black', 'freeze', 'starve', 'exception'):
            raise ValueError(f'unknown fault kind {self.kind!r}')
        if self.kind == 'exception' and self.where not in _EXC_TARGET:
            raise ValueError(f'where must be pnp|ba, got {self.where!r}')

    def active(self, index: int) -> bool:
        return self.start <= index < self.start + self.length


class FaultInjectOdometry:
    def __init__(self, base, faults: list[Fault], lost_after: int = 3, starve_sigma: float = 10.0):
        self.base, self.faults, self.lost_after, self.starve_sigma = base, list(faults), lost_after, starve_sigma
        self._tmp = None
        self._cur = -1
        for where, name in _EXC_TARGET.items():
            if any(f.kind == 'exception' and f.where == where for f in self.faults):
                if not hasattr(base, name):
                    raise ValueError(f'base has no {name} to inject a {where} exception into')
                setattr(base, name, self._armed(getattr(base, name), where))
        self._clear()

    def _clear(self) -> None:
        self._camera = None
        self._last_good: Frame | None = None
        self._last_out: np.ndarray | None = None     # last emitted increment (motion model)
        self._dirty = False                          # base must be re-initialised before the next step
        self._streak = 0                             # consecutive supervisor-substituted frames
        self.log: list[dict] = []                    # one entry per emitted estimate: index, fault, state, source

    def _armed(self, fn, where: str):
        def wrapped(*a, **k):
            if any(f.kind == 'exception' and f.where == where and f.active(self._cur) for f in self.faults):
                raise FaultInjected(f'injected {where} exception at frame {self._cur}')
            return fn(*a, **k)
        return wrapped

    # ------------------------------------------------------------------ Odometry protocol
    def reset(self, camera: PinholeCamera) -> None:
        self._clear()
        self._camera = camera
        self.base.reset(camera)

    def step(self, frame: Frame) -> PoseEstimate | None:
        self._cur = frame.index
        kind = next((f.kind for f in self.faults if f.kind != 'exception' and f.active(frame.index)), None)
        exc = any(f.kind == 'exception' and f.active(frame.index) for f in self.faults)
        base_frame = self._corrupt(frame, kind) if kind else frame
        if kind is None:
            self._last_good = frame

        if self._dirty and self._last_out is not None:   # re-initialise after an exception, keep g
            return self._reinit(base_frame, frame, kind, exc)
        try:
            est = self.base.step(base_frame)
        except Exception:                                 # BA/PnP exceptions are swallowed -> DEGRADED (F11)
            return self._substitute(frame, kind, exc)
        if est is None:                                   # very first frame
            return None
        if est.source == 'tracked':
            self._streak = 0
        else:                                             # the base's own motion-model frames count toward LOST too
            self._streak += 1
            if self._streak >= self.lost_after and est.tracking_state != 'LOST':
                est = dataclasses.replace(est, tracking_state='LOST')
        self._last_out = est.rel_increment.copy()
        self._record(frame, kind, exc, est.tracking_state, est.source)
        return est

    def finalize(self) -> dict:
        return self.base.finalize() if hasattr(self.base, 'finalize') else {}

    # ------------------------------------------------------------------ supervisor
    def _substitute(self, frame: Frame, kind, exc) -> PoseEstimate:
        self._streak += 1
        self._dirty = True
        state = 'LOST' if self._streak >= self.lost_after else 'DEGRADED'
        return self._emit(frame, kind, exc, state)

    def _reinit(self, base_frame: Frame, frame: Frame, kind, exc) -> PoseEstimate:
        saved = {a: getattr(self.base, a) for a in _KEEP_ATTRS if hasattr(self.base, a)}
        self.base.reset(self._camera)
        try:
            self.base.step(base_frame)                    # first step of a fresh base: no increment
        except Exception:
            return self._substitute(frame, kind, exc)
        for a, v in saved.items():
            setattr(self.base, a, v)
        self._dirty = False
        return self._emit(frame, kind, exc, 'BOOTSTRAP')

    def _emit(self, frame: Frame, kind, exc, state: str) -> PoseEstimate:
        inc = self._last_out if self._last_out is not None else np.eye(4)
        self._record(frame, kind, exc, state, 'motion_model')
        return PoseEstimate(frame.index, frame.timestamp, inc.copy(), state, 'motion_model', False, {'total': 0.0})

    def _record(self, frame: Frame, kind, exc, state: str, source: str) -> None:
        self.log.append({'index': frame.index, 'fault': 'exception' if exc and kind is None else kind,
                         'state': state, 'source': source})

    # ------------------------------------------------------------------ image faults
    def _corrupt(self, frame: Frame, kind: str) -> Frame:
        if kind == 'freeze':
            src = self._last_good or frame
            return dataclasses.replace(frame, image_path=src.image_path, depth_path=src.depth_path)
        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        if kind == 'black':
            img = np.zeros_like(gray)
        else:                                             # starve: flat mean grey + one smooth patch of the real image
            h, w = gray.shape
            yy, xx = np.mgrid[:h, :w]
            blob = np.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * self.starve_sigma ** 2))
            mean = float(gray.mean())
            img = np.clip(mean + (gray.astype(np.float32) - mean) * blob, 0, 255).astype(np.uint8)
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix='fault_inject_')
        path = Path(self._tmp.name) / f'{kind}.png'
        cv2.imwrite(str(path), img)
        return dataclasses.replace(frame, image_path=path)
