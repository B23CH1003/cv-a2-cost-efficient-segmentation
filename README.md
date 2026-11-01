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
python tune.py --lams 0 1 2 4 8 --sigmas 0.05 0.1 0.25 0.5     # then finer grids; the final run used
python tune.py --lams 4 6 8 --sigmas 0.1 0.15 0.25 --biases 0.5 --temps 1.6 --write   # lam, sigma, logit offset, temperature
python evaluate.py --set val
python evaluate.py --set te --save_preds data/cache/te_preds.npy   # official T_CPU
python profile_stages.py
```

## Captured photos
8 phone photos (`captured/original`, `images_256`, `masks_256`), item1-8. The three predictions item3, item4, item8 come from
`captured/pred_override/` (mask plus marked overlay); `make_report.py` uses them in place of a local run for those photos.
Masks are hand-written polygons/strokes in `annotations.py` (`python annotations.py` re-renders them).

## Final numbers (this machine, i5-12450H)
| set | mIoU | BF1 | Q | T_CPU | CES |
|---|---|---|---|---|---|
| validation (1,055) | 0.762 | 0.489 | 0.680 | 8.7 ms | 0.655* |
| DUTS-TE (5,019, official) | 0.630 | 0.354 | 0.547 | 9.0 ms | 0.527 |
| own photos (8) | 0.840 | 0.542 | 0.750 | - | 0.723* |

\* CES computed with the official DUTS-TE T_CPU. T_CPU varies by a few ms between runs on this laptop.

## Report
`report/report.tex` (+ `report/figures/`): compile with pdflatex (needs graphicx, booktabs, amsmath, amssymb, hyperref, float).
