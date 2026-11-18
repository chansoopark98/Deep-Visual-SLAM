# Depth Anything V2 — metric depth (minimal inference copy)

- **Upstream code:** https://github.com/DepthAnything/Depth-Anything-V2 at commit `a561b84`
  (local reference clone `assets/references/depth-anything-v2`, see `assets/references/MANIFEST.tsv`).
- **Copied:** `metric_depth/depth_anything_v2/` → `dav2_metric/` (dpt.py, dinov2.py, dinov2_layers/, util/). Unmodified.
  It is byte-identical to the repo's `model/depth_anything_v2/` code; the repo's problem was the checkpoint, not the code.
- **Code licence:** Apache-2.0 (`LICENSE`, copied from upstream). DINOv2 files: Copyright Meta Platforms, Apache-2.0.
- **Used by:** `dvslam/depth/dav2.py` (`DAv2Metric`, encoder `vits`, `max_depth=20`).

## Weights

| file | source (official, linked from `metric_depth/README.md`) | HF commit | sha256 | licence |
|---|---|---|---|---|
| `depth_anything_v2_metric_hypersim_vits.pth` (99,222,290 B) | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth | `3bc65d4e14a6786a61acec16453c50e12bf5f338` | `b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545` | Apache-2.0 (HF model card `license: apache-2.0`; upstream README: Small = Apache-2.0, Base/Large/Giant = CC-BY-NC-4.0) |

Stored at `runs/e1_weights/dav2/` (git-ignored via `runs/`); override with `$DVSLAM_DEPTH_WEIGHTS`.
The sha256 equals the HF LFS sha256.

**Checkpoint mismatch in this repo (research 13 §1.1), now confirmed by hash:** `assets/weights/depth/depth_anything_v2_vits.pth`
has sha256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`, which is the HF LFS sha256 of
`depth-anything/Depth-Anything-V2-Small/depth_anything_v2_vits.pth`, i.e. the **relative** (disparity) checkpoint.
`model/depth_anything_v2/test.py` loads it into the metric head (Sigmoid × 20), so that path outputs neither metric depth
nor clean disparity.
