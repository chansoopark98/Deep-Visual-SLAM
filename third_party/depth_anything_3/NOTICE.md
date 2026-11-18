# Depth Anything 3 — DA3METRIC-LARGE (minimal inference copy)

- **Upstream code:** https://github.com/ByteDance-Seed/Depth-Anything-3 at commit `3d835ec`
  (local reference clone `assets/references/depth-anything-3`).
- **Copied, unmodified** (package `depth_anything_3/`, from `src/depth_anything_3/`):
  `model/dpt.py`, `model/reference_view_selector.py`, `model/utils/head_utils.py`, `model/dinov2/{dinov2.py,
  vision_transformer.py, layers/}`, `utils/{constants.py, logger.py}`.
- **Not copied:** `api.py`, `model/da3.py`, `cfg.py` and the rest (they pull omegaconf, the camera/GS heads and I/O).
  The `__init__.py` files are new and empty (upstream `model/__init__.py` imports `da3.py`).
  `dvslam/depth/da3.py` builds `DinoV2` + `DPT` directly with the `da3metric-large.yaml` config (identical to the HF
  `config.json`) and re-implements the single-view preprocessing (`utils/io/input_processor.py`) and the mono sky step
  (`model/da3.py _process_mono_sky_estimation`).
- **Runtime deps added to the base env:** `einops 0.8.2`, `addict 2.4.0` (pure Python, installed with `--no-deps`;
  torch/numpy untouched).
- **Code licence:** Apache-2.0 (`LICENSE`, copied from upstream). DINOv2 files: Copyright Meta Platforms, Apache-2.0.

## Weights

| file | source (official, linked from upstream README model table) | HF commit | sha256 | licence |
|---|---|---|---|---|
| `model.safetensors` (1,336,734,448 B), `config.json` | https://huggingface.co/depth-anything/DA3METRIC-LARGE | `4010e39f3634a45bc60553321fb49fb760bd594e` | `bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776` | Apache-2.0 (HF card `license: apache-2.0`; upstream README table: DA3METRIC-LARGE Apache 2.0) |

Stored at `runs/e1_weights/da3/` (git-ignored via `runs/`). State dict loads with `strict=True` after stripping the
`model.` prefix. The sha256 equals the HF LFS sha256.

Metric conversion (upstream README FAQ): `depth_m = focal_px * output / 300`, focal at the processed resolution.
