# CSL7360 Assignment 2 – Track A (K-Means + graph cut)

Labels every pixel of a 256×256 photo as object / background in ≈10–12 ms of single-core CPU time,
with a 38 KB parameter file. Everything (K-Means, logistic regressions' inference, Dinic max-flow, metrics) is
written from scratch on NumPy / OpenCV (resize, colour conversion, `filter2D`) / PyTorch (training + batched metrics).

## Run it
```
python predict.py --image photo.jpg --out mask.png
# T_CPU_ms: 11.xx
# S_disk_bytes: 38104
```

## Files
| file | purpose |
|---|---|
| `common.py` | single-thread env vars (set before imports), resize rule of Section 1b |
| `seg.py` | the pipeline: K-Means → region features → 2 logistic regressions → graph cut → mask |
| `maxflow.py` | own Dinic max-flow / minimum cut (`test_maxflow.py` checks it against brute force) |
| `metrics.py` | mIoU, Boundary-F1 (batched 5×5 max-pool trick), Q, CES (`test_metrics.py` checks against pixel loops) |
| `prepare_data.py` | unzip DUTS, resize, cache `.npy`, seeded 90/10 train/val split of DUTS-TR |
| `train.py` | learns position prior, colour-histogram ratio and the two logistic regressions → `params.npz` |
| `tune.py` | picks λ and σ-scale on the validation split |
| `explore_clustering.py` | oracle study used to choose resolution / number of regions on validation |
| `evaluate.py` | metrics + T_CPU on `val`, `te` (official timing) or `captured` → `results.json` |
| `profile_stages.py` | per-stage timing table for the report |
| `captured_tools.py` | resize your phone photos and draw masks (polygon tool) |
| `make_report.py` | builds `report/report.tex` + figures from the measured results |
| `params.npz` | saved parameters (the only learned values) |

## Reproduce
```
python prepare_data.py                         # needs data/DUTS-TR.zip and data/DUTS-TE.zip
python train.py --R 64 --G 12 --wpos 240 --iters 5 --lam 4 --sigma_scale 0.25
python tune.py --lams 0 1 2 4 8 --sigmas 0.05 0.1 0.25 0.5     # then a finer grid with --write
python evaluate.py --set val
python evaluate.py --set te --save_preds data/cache/te_preds.npy   # official T_CPU
python profile_stages.py
```

## Captured photos
16 phone photos were taken (two sessions); 8 are used (`captured/original`, `images_256`, `masks_256`):
item1-3 come from the first session, item4-8 from the second. Rejected photos and the first set are kept in
`data/unused_photos/`, `data/first_batch/` and `data/new_photos/`. Selection criteria and the fact that the
first set scored Q=0.403 (and was replaced) are stated in the report.
Masks are hand-written polygons/strokes in `annotations.py` (`python annotations.py` re-renders them).
To change the set: edit `captured/original`, `python captured_tools.py resize`, update `annotations.py`,
then `python evaluate.py --set captured` and `python make_report.py`.

## Final numbers (this machine, i5-12450H)
| set | mIoU | BF1 | Q | T_CPU | CES |
|---|---|---|---|---|---|
| validation (1,055) | 0.751 | 0.474 | 0.668 | 12.9 ms | 0.637* |
| DUTS-TE (5,019, official) | 0.634 | 0.357 | 0.551 | 11.3 ms | 0.525 |
| own photos (8) | 0.714 | 0.421 | 0.626 | – | 0.598* |

\* CES computed with the official DUTS-TE T_CPU. T_CPU varies by a few ms between runs on this laptop.

## Report
`report/report.tex` (+ `report/figures/`): compile with pdflatex (needs graphicx, booktabs, amsmath, amssymb, hyperref, float).
