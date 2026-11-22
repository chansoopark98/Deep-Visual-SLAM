# Training

All training code runs on a CUDA GPU. Run directories go to `runs/` (git-ignored). Set `DVSLAM_TUMVI_ROOT` and
`DVSLAM_CACHE_ROOT` first (see [DATASETS.md](DATASETS.md)). The stages are described in [PIPELINE.md](PIPELINE.md).

## SSL depth network

```bash
python apps/train_ssl.py --config configs/ssl_tumvi.yaml --name base [--set loss.flow=0 --set train.epochs=30]
python apps/train_ssl.py --name base --resume          # continue runs/ssl/base/last.pt
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 apps/train_ssl.py --name base   # one process per GPU
```

Writes `runs/ssl/<name>/{config.yaml,train_log.jsonl,val_log.jsonl,last.pt,best.pt}`. The config trains on `room1`–`room4`
and validates on `room5`.

### Stereo depth hints

Stereo depth hints (the `hints_core` release) are made once per room with

```bash
python scripts/cache_stereo_hints.py --out $DVSLAM_CACHE_ROOT/tumvi_stereo_hints room1 room2 room3 room4 room5
```

(cam0/cam1 SGBM, training-time signal only) and enabled with `--set data.stereo_hints=$DVSLAM_CACHE_ROOT/tumvi_stereo_hints
--set loss.stereo_hint=1.0`. The released `hints_core` used weight 1.0, 20 epochs, batch 8, lr 1e-4, seed 0.

## S2 learned photometric weights

Dump the tracker's refinement inputs per room (same arguments as `scripts/exp003_track_log.py`; use the final-candidate
`--opt` list from [PIPELINE.md](PIPELINE.md#tracker-options-final-candidate)), then train `PhotoNet` through the unrolled
refinement:

```bash
python scripts/s2_dump_pairs.py --scene room1 --out runs/s2/dump <final-candidate --opt list>   # once per room
python apps/train_photo_s2.py --dump runs/s2/dump --depth <depth cache dir> --out runs/s2/train/pilot --mode w --steps 2000
```

`<depth cache dir>` is the frozen depth cache of the tracker (`python scripts/exp003_cache_depth.py --ckpt <ckpt>
--out <dir>`). Use the result with `--opt photo_feat=<file>` (requires `photo_backend=torch`, `photo_levels=1`).

## S3 test-time depth adaptation

LoRA test-time adaptation of the depth decoder. It uses the pseudo-label poses of one tracker run on the test sequence
(`--labels`, start frame 0), cam0 images only:

```bash
python apps/calib_s3.py train --ckpt runs/ssl/<run>/last.pt --scene room5 --labels <run>/room5.inc.npz --out runs/s3/room5
python apps/calib_s3.py cache --ckpt runs/ssl/<run>/last.pt --adapter runs/s3/room5/adapter.pt --scene room5 --out runs/s3/depth
```

`cache` writes a depth cache that the tracker reads with `--opt prior=cache --opt prior_ckpt=runs/s3/depth`.
Run `python apps/calib_s3.py <command> -h` for all options.

## Exporting release weights

```bash
python scripts/export_release_weights.py SRC DST [--verify-images IMG.png ...] [--device cuda]
```

Strips a training checkpoint (EMA weights, config, K, size, epoch, step) or a PhotoNet file to an inference-only file and,
with `--verify-images`, checks that the depth output is bit-identical.
