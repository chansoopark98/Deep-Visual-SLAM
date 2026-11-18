# MoGe-2 (minimal inference copy)

- **Upstream code:** https://github.com/microsoft/MoGe at commit `74fbce0`
  (local reference clone `assets/references/moge`).
- **Copied** (package `moge/`):
  - `model/v2.py` — **modified:** the `utils3d` import and `MoGeModel.infer` were removed and
    `normalized_view_plane_uv` is imported from `utils/geometry.py`; the network (`__init__`, `forward`,
    `from_pretrained`) is unchanged. Inference post-processing (shift solve with a known FoV, metric scale, mask) is
    re-implemented in `dvslam/depth/moge.py` following upstream `infer`.
  - `model/utils.py`, `model/modules/{dinov2_encoder.py, mlp.py, conv_stack.py}` — unmodified.
  - `model/modules/dinov2/{hub, layers, models}` — unmodified (DINOv2, Meta, Apache-2.0). The DINOv2 `utils/` training
    helpers were not kept (unused; they shell out to git).
  - `utils/geometry.py` — **new file:** verbatim copies of `normalized_view_plane_uv` and `recover_focal_shift`
    (`utils/geometry_torch.py`) and `solve_optimal_focal_shift`, `solve_optimal_shift` (`utils/geometry_numpy.py`),
    so that `utils3d` is not needed.
  - `__init__.py` files are new and empty.
- **Code licence:** MIT (`LICENSE`, copied from upstream), except `model/modules/dinov2/` which is Apache-2.0 (Meta AI),
  per upstream README "License".

## Weights

| file | source (official, linked from upstream README "Pretrained Models") | HF commit | sha256 | licence |
|---|---|---|---|---|
| `model.pt` (140,550,416 B) — `moge-2-vits-normal` | https://huggingface.co/Ruicheng/moge-2-vits-normal | `26b477f41595707c5db6770294c0d1721e8ed4ed` | `79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc` | MIT (HF card `license: mit`, base model facebook/dinov2-small) |

Stored at `runs/e1_weights/moge/` (git-ignored via `runs/`). Loaded with `torch.load(weights_only=True)` and
`load_state_dict(strict=True)`. The sha256 equals the HF LFS sha256.
