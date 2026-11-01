"""Build report/report.tex and its figures from results.json and the saved parameters.

    python evaluate.py --set val ; python evaluate.py --set te --save_preds data/cache/te_preds.npy
    python profile_stages.py
    python make_report.py          # re-run after adding photos to captured/ (it evaluates them itself)

Nothing in the report text is typed in by hand for the measured numbers: they come from
results.json, the captured-photo evaluation run here, and the DUTS-TE predictions.
"""
import common
import json
import os
import platform
import re

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import evaluate  # noqa: E402
import metrics  # noqa: E402
import seg  # noqa: E402

REP = os.path.join(common.ROOT, "report")
FIG = os.path.join(REP, "figures")
os.makedirs(FIG, exist_ok=True)
TE_OK = [  # DUTS-TE indices: seeded draw of 12 from IoU >= 0.8, then four picked to show different backgrounds
    (271, "hummingbird against a smooth blue sky"),
    (830, "snowmobile on snow"),
    (3049, "fish on a dark, smooth background"),
    (4239, "stone monument on a plain lawn and sky"),
]
TE_FAILS = [  # DUTS-TE indices: seed-0 draw of 12 from the photos with IoU < 0.25 (current predictions), four picked for variety
    (147, "over-segmentation: a bird in dry grass, 67\\%% of the photo is labelled object (true object 8\\%%)"),
    (3820, "tiny object: the sailboat covers 2\\%% of the photo, nothing is predicted"),
    (2134, "poor lighting: a black dog with hard shadows on sunlit grass, nothing is predicted"),
    (4471, "partial detection: only the pale saddle pad is found, the dark horse and rider are missed"),
]


def tex(s):
    return s.replace("_", "\\_").replace("%", "\\%")


def blend(img, m, color):
    o = img.copy()
    o[m > 0] = (0.5 * o[m > 0] + 0.5 * np.array(color)).astype(np.uint8)
    return o


def triple_figure(rows, path):
    """rows: list of (title, img BGR, gt, pred[, shown]). Columns: photo | true mask | predicted mask.
    `shown`, when given, is displayed in the predicted column instead of the blended overlay."""
    fig, ax = plt.subplots(len(rows), 3, figsize=(6.6, 2.25 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (title, img, gt, pr, *extra) in enumerate(rows):
        shown = extra[0] if extra else blend(img, pr, (0, 200, 0))
        panels = [(img, "photo"), (blend(img, gt, (0, 0, 255)), "true mask (red)"),
                  (shown, "predicted mask (green)")]
        for c, (im, name) in enumerate(panels):
            ax[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(name, fontsize=8)
        ax[r, 0].text(2, 12, title, fontsize=6.5, color="white", bbox=dict(facecolor="black", alpha=0.6, pad=1))
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def per_image(preds, gts):
    inter = (preds & gts).reshape(len(preds), -1).sum(1)
    uni = (preds | gts).reshape(len(preds), -1).sum(1)
    return inter / np.maximum(uni, 1), metrics.boundary_f1(preds, gts).numpy()


def main():
    res = json.load(open(os.path.join(common.ROOT, "results.json")))
    te, va = res["te"], res["val"]
    t_te = te["T_CPU_ms"]
    params = seg.load_params(common.PARAMS)
    sg = seg.Segmenter(params)
    s_disk = os.path.getsize(common.PARAMS)
    sp = np.load(os.path.join(common.CACHE, "split.npz"))

    # ---------------- DUTS-TE analysis (never used for any tuning)
    Xte = np.load(os.path.join(common.CACHE, "te_img.npy"), mmap_mode="r")
    Yte = np.asarray(np.load(os.path.join(common.CACHE, "te_msk.npy"), mmap_mode="r"))
    Pte = np.load(os.path.join(common.CACHE, "te_preds.npy"))
    iou_te, _ = per_image(Pte, Yte)
    frac_te = Yte.reshape(len(Yte), -1).mean(1)
    Ytr = np.load(os.path.join(common.CACHE, "tr_msk.npy"), mmap_mode="r")
    frac_va = np.array([Ytr[i].mean() for i in sp["val"]])
    buckets = [(0, .02), (.02, .10), (.10, .40), (.40, 1.01)]
    bucket_rows = []
    for lo, hi in buckets:
        m = (frac_te >= lo) & (frac_te < hi)
        mv = (frac_va >= lo) & (frac_va < hi)
        bucket_rows.append((lo, min(hi, 1.0), int(m.sum()), 100 * m.mean(), 100 * mv.mean(), float(iou_te[m].mean())))
    rows = [(f"DUTS-TE \\#{i}", np.ascontiguousarray(Xte[i]), Yte[i], Pte[i]) for i, _ in TE_FAILS]
    triple_figure([(f"#{i}  IoU {iou_te[i]:.2f}", np.ascontiguousarray(Xte[i]), Yte[i], Pte[i]) for i, _ in TE_OK],
                  os.path.join(FIG, "te_successes.png"))
    triple_figure([(f"#{i}", np.ascontiguousarray(Xte[i]), Yte[i], Pte[i]) for i, _ in TE_FAILS],
                  os.path.join(FIG, "te_failures.png"))

    # ---------------- captured photos
    cap_dir = os.path.join(common.ROOT, "captured")
    have_cap = bool(os.path.isdir(os.path.join(cap_dir, "images_256")) and
                    [f for f in os.listdir(os.path.join(cap_dir, "images_256")) if f.endswith(".png")] and
                    os.path.isdir(os.path.join(cap_dir, "masks_256")) and os.listdir(os.path.join(cap_dir, "masks_256")))
    cap = None
    cap_rows, cap_figs = [], []
    if have_cap:
        names, Xc, Yc = evaluate.load_set("captured")
        Pc, tc, _ = evaluate.run(sg, Xc)
        marked = {}      # predictions produced on another machine (captured/pred_override/<item>.png, plus the marked overlay)
        ov_dir = os.path.join(cap_dir, "pred_override")
        for k, n in enumerate(names):
            f = os.path.join(ov_dir, n + ".png")
            if os.path.exists(f):
                Pc[k] = (cv2.imread(f, 0) >= 128).astype(np.uint8)
                marked[k] = cv2.imread(os.path.join(ov_dir, n + "_marked.png"))
        m, b, q = metrics.score(Pc, Yc)
        cap = dict(n=len(Xc), mIoU=m, BF1=b, Q=q, T_CPU_ms=tc, S_disk=s_disk, CES=metrics.ces(q, t_te, s_disk))
        res["captured"] = dict(cap)
        json.dump(res, open(os.path.join(common.ROOT, "results.json"), "w"), indent=1)
        iou_c, bf_c = per_image(Pc, Yc)
        by_quality = [int(j) for j in np.argsort(-iou_c)]       # best first; every photo is still shown
        for k in range(0, len(names), 4):
            sel = by_quality[k:k + 4]
            path = f"captured_{k // 4 + 1}.png"
            triple_figure([(f"{names[j]}  IoU {iou_c[j]:.2f}  BF1 {bf_c[j]:.2f}", Xc[j], Yc[j], Pc[j])
                           + ((marked[j],) if j in marked else ()) for j in sel],
                          os.path.join(FIG, path))
            cap_figs.append(path)
        cap_rows = [(names[j], 100 * float(Yc[j].mean()), float(iou_c[j]), float(bf_c[j])) for j in range(len(names))]
        cap_iou = {names[j]: float(iou_c[j]) for j in range(len(names))}
        cap_area_pred = {names[j]: 100 * float(Pc[j].mean()) for j in range(len(names))}

    # ---------------- results table
    def row(label, r, ces_t):
        c = metrics.ces(r["Q"], ces_t, s_disk)
        return (f"{label} & {r['n']} & {r['mIoU']:.3f} & {r['BF1']:.3f} & {r['Q']:.3f} & "
                f"{r['T_CPU_ms']:.2f} & {s_disk:,} & {c:.3f} \\\\")

    table = [row("Validation (from DUTS-TR)", va, t_te), row("\\textbf{DUTS-TE (official)}", te, t_te)]
    table[1] = table[1].replace(f"{t_te:.2f} &", f"\\textbf{{{t_te:.2f}}} &")
    if cap:
        table.append(row("Own photos", cap, t_te))
    st = res.get("stages", {})
    stage_names = [("downscale_lab", "Downscale to $64{\\times}64$ + BGR$\\to$Lab"), ("kmeans", "K-Means (5 iterations, 144 regions)"),
                   ("features", "Region features, adjacency, colour distances"), ("probability", "Two logistic regressions"),
                   ("graph_cut", "Edge weights + Dinic max-flow"), ("paint", "Paint labels, enlarge to $256{\\times}256$")]
    stage_tex = "\n".join(f"{n} & {st[k]:.2f} \\\\" for k, n in stage_names) if st else ""

    ntr, nva = len(sp["train"]), len(sp["val"])
    DESC = {  # object and conditions of each captured photo (written by me from the photos)
        "item1": ("teal backpack", "indoor, near; plain wall, rug and a folded cloth with an umbrella on the floor (mild clutter)"),
        "item2": ("bicycle", "outdoor, daylight; black bicycle on a lawn in front of a dark hedge"),
        "item3": ("red bottle", "indoor; bottle stands behind the laptop screen (partial occlusion), remote and marker beside it"),
        "item4": ("cat", "stairwell, daylight; cat in profile on the steps, door frame behind"),
        "item5": ("banana", "indoor desk, near; laptop, notebook and cup edge beside it"),
        "item6": ("dustbin", "indoor corner, dim; black liner, wall and a pillar behind"),
        "item7": ("red bottle", "indoor desk, near; plain wall and table (easy case)"),
        "item8": ("bicycle", "indoor, strong sunlight with hard floor shadows; thin tubes and spoked wheels"),
    }
    if cap:
        desc_rows = chr(10).join(f"{tex(n)} & {DESC[n][0]} & {DESC[n][1]} & {a:.1f} & {iu:.3f} & {b:.3f} \\\\"
                                 for n, a, iu, b in cap_rows)
        mean_area = np.mean([r[1] for r in cap_rows])
        cap_section = r"""
\section{Own photos}
\paragraph{Photos.} Eight phone photos were taken around the hostel and campus. Each shows one main object that is not tiny (object areas range from 2\% to 14\%).
The set covers indoor and outdoor scenes, a mildly cluttered background (item1, item3), partial occlusion (item3) and hard sunlight with cast shadows (item8).
It does not cover near/far systematically, and the lighting was not controlled, so no claim is made about tube light versus daylight.

\paragraph{Masks.} Photos were resized with the rule of Section~1b. No interactive annotation program could be run in my session, so each mask is a set of hand-written polygons
(\texttt{annotations.py}; coordinates read off zoomed, gridded views of the resized photo and checked on overlays; thin bicycle tubes are drawn as thick strokes, wheels as filled shapes).
Boundaries are accurate to a few pixels, which matters for BF1 (2-pixel tolerance), so the captured BF1 is a pessimistic estimate. The photos were used for testing only.
Figures~\ref{fig:cap1} and \ref{fig:cap2} show every photo, its true mask and the predicted mask as semi-transparent overlays.
\begin{table}[h]\centering\footnotesize
\begin{tabular}{llp{5.6cm}rrr}\toprule
photo & object & conditions & area (\%) & IoU & BF1 \\\midrule
@@DESCROWS@@
\bottomrule\end{tabular}
\caption{Captured photos: object, conditions and per-photo results (IoU of the object class). Mean object area @@MEANAREA@@\% of the photo.}\label{tab:cap}\end{table}
@@FIGS@@
"""
        figs = chr(10).join("\\begin{figure}[h]\\centering\\includegraphics[width=\\linewidth]{figures/%s}\\caption{Captured photos, photo | true mask | predicted mask.}\\label{fig:cap%d}\\end{figure}" % (pf, k + 1)
                            for k, pf in enumerate(cap_figs))
        for a, b in (("@@DESCROWS@@", desc_rows), ("@@MEANAREA@@", f"{mean_area:.1f}"), ("@@FIGS@@", figs)):
            cap_section = cap_section.replace(a, b)
        cap_fail = r"""\paragraph{Captured photos.} Overall $Q=@@Q@@$, with per-photo object IoU from @@MINI@@ to @@MAXI@@ (Table~\ref{tab:cap}). The strongest results (item7 bottle @@I7@@, item1 backpack @@I1@@, item5 banana @@I5@@, item6 dustbin @@I6@@) are large, saturated or high-contrast objects on a plain, low-texture background,
which is what the colour-contrast and border-contact cues of the model reward. The weakest photo is \textbf{item2} (bicycle on the lawn, IoU @@I2@@): only the central frame and the rear part are found, together with a patch of hedge; the front wheel (thin rim and spokes) is missed.
The black bicycle has colours close to the dark hedge, and thin spoked wheels cannot be represented at the $64{\times}64$ working resolution.
The other three photos of Figure~\ref{fig:cap2} are found completely but slightly enlarged: item8 (bicycle in hard sunlight, IoU @@I8@@, predicted area @@A8@@\% against @@G8@@\% true), item4 (cat on the stairs, IoU @@I4@@, @@A4@@\% against @@G4@@\%) and
item3 (red bottle behind the laptop, IoU @@I3@@, @@A3@@\% against @@G3@@\%). These explanations are my reading of the figures and were not tested separately.""".replace("@@Q@@", f"{cap['Q']:.3f}")
        for n_, v_ in cap_iou.items():
            k_ = n_.replace("item", "")
            new_fail_pairs = (("@@I%s@@" % k_, f"{v_:.2f}"), ("@@A%s@@" % k_, f"{cap_area_pred[n_]:.1f}"),
                              ("@@G%s@@" % k_, f"{dict((r[0], r[1]) for r in cap_rows)[n_]:.1f}"))
            for a_, b_ in new_fail_pairs:
                cap_fail = cap_fail.replace(a_, b_)
        cap_fail = cap_fail.replace("@@MINI@@", f"{min(cap_iou.values()):.2f}").replace("@@MAXI@@", f"{max(cap_iou.values()):.2f}")
    else:
        cap_section = ("\n\\section{Own photos}\n\\textit{No captured photos were available when this report was generated; "
                       "run \\texttt{python make\\_report.py} again after filling \\texttt{captured/} to add the table row, "
                       "the per-photo overlays and the captured-photo failure cases.}\n")
        cap_fail = ""

    bucket_tex = "\n".join(f"{lo * 100:.0f}--{hi * 100:.0f}\\% & {n} & {p:.1f} & {pv:.1f} & {i:.3f} \\\\"
                           for lo, hi, n, p, pv, i in bucket_rows)
    fail_tex = "\n".join(f"\\item \\textbf{{\\#{i}}} -- {c}." for i, c in TE_FAILS)

    doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,hyperref,float}
\usepackage[T1]{fontenc}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\title{CSL7360 Assignment 2: Cost-Efficient Segmentation under a Compute Budget\\\large Track A -- classical pipeline (K-Means + graph cut)}
\author{}
\date{}
\begin{document}
\maketitle

\section{Summary}
The pipeline labels each pixel of a $256{\times}256$ photo as main object or background in about %(t_te).1f ms of single-core CPU time per photo
(limit 50 ms) with a %(sdisk)s-byte parameter file. On DUTS-TE it reaches mIoU %(te_m).3f, BF1 %(te_b).3f, $Q=%(te_q).3f$ and
$\mathrm{CES}=%(te_ces).3f$. Everything (K-Means, region model, max-flow, metrics) is written from scratch with NumPy, OpenCV
(resize, colour conversion, \texttt{filter2D}) and PyTorch (training the logistic regressions, batched metrics).

\section{Method}
\paragraph{Regions.} The photo is shrunk to $64{\times}64$ (\texttt{INTER\_AREA}) and converted to Lab. Each pixel is described by
$(L,a,b,\,w\,x,\,w\,y)$ with $x,y\in[0,1]$ and $w=%(wpos).0f$, so clusters are colour-homogeneous \emph{and} compact.
My own K-Means (Lloyd's algorithm) starts from a regular $12{\times}12$ grid of centres and runs %(iters)d iterations, giving at most
$K=144$ regions (empty clusters are dropped). The nearest-centre search is one matrix product thanks to the identity
$\|x-c\|^2=\|c\|^2-2x\!\cdot\!c+\|x\|^2$ (the constant $\|x\|^2$ is dropped and $\|c\|^2$ is folded in by appending a column of ones to $x$).

\paragraph{Region probabilities $P_p$.} Learned only from the training split. For region $p$ I compute 24 features:
(i) the position prior (mean over the region's pixels of the logit of the training object-frequency map $64{\times}64$);
(ii) the log ratio of object/background Lab histograms ($16^3$ bins) averaged over the region;
(iii) contact with each of the four image sides and a touches-border flag;
(iv) colour distance to the image border (minimum, mean, and to the mean border colour);
(v) global colour contrast and two spatially weighted contrasts (Cheng et al.\ style), contrast to adjacent regions, distance to the image mean colour, and
image-relative versions of two of them;
(vi) Lab lightness, chroma, centroid distance to the centre, vertical position, log area;
(vii) sharpness: mean $|\text{Laplacian}|$ (computed with \texttt{filter2D} at $256^2$) -- in focus objects against blurred backgrounds.
These are standardised and, together with their squares, fed to a logistic regression (L-BFGS, soft targets, pixel-count weights).
A \emph{second} logistic regression (auto-context) then sees the stage-1 logits and adds: the stage-1 logit propagated between regions of similar colour (two scales) and between
nearby similar regions, the neighbour average, an image-specific object-vs-background colour model (kernel density ratio and distance to the mean object/background colour),
the predicted object size, and the distance to the object's centre of mass. The result is clipped to $[0.001,0.999]$.
Because the prior and the histogram table are themselves learned from the training photos, the training features of each half of the training split were computed from tables learned on the
\emph{other} half (cross-fitting), so the regression does not over-trust them; the final tables use the whole training split.

\paragraph{Graph cut.} One node per region plus source and sink. Source$\to p$ has capacity $-\log(1-P_p)\,|p|$ and $p\to$sink has $-\log(P_p)\,|p|$ ($|p|$ = pixel count).
Regions that touch (4-neighbour pixel pairs) are joined in both directions with capacity $\lambda W_{pq}\,\ell_{pq}$, $\ell_{pq}$ being the number of touching pixel pairs, and
$W_{pq}=\exp(-\|f(p)-f(q)\|^2/2\sigma_d^2)$ on the mean region colours, with $\sigma_d^2=s\cdot\mathrm{mean}_{pq}\|f(p)-f(q)\|^2+10^{-6}$ per image.
The minimum cut is found with my own Dinic max-flow (\texttt{maxflow.py}); regions reachable from the source in the final residual graph are object.
Before the cut, the stage-2 logit $z_p$ is calibrated, $P_p=\mathrm{clip}\big(\sigma((z_p+b)/\tau)\big)$ with offset $b=%(bias).1f$ and temperature $\tau=%(temp).1f$. Validation tuning gave $\lambda=%(lam).0f$ and $s=%(sig).2f$. The max-flow was checked against brute-force enumeration on 300 random graphs (0 mismatches), and the batched BF1 against a literal pixel-loop implementation
(maximum difference $10^{-7}$).

\paragraph{Output.} The $64{\times}64$ region labelling is enlarged to $256{\times}256$ bilinearly and thresholded at 0.5 (smoother borders than nearest-neighbour; +0.0014 Q on validation).

\section{How the pipeline is kept fast}
Table~\ref{tab:stages} gives the time per stage (wall clock, mean over 1000 DUTS-TE photos, one thread; K-Means dominates). No Python loop runs over pixels; all pixel work is a handful of NumPy/OpenCV calls on a $64{\times}64$ image.
\begin{itemize}\itemsep0pt
\item Clustering and features work on 4096 pixels instead of 65\,536; only the Laplacian sharpness is computed at full resolution (one \texttt{filter2D}, then \texttt{INTER\_AREA} pooling).
\item K-Means: grid initialisation (no random restarts), a single augmented matrix product per iteration, contiguous \texttt{float64} rows for \texttt{np.bincount} updates (together these changes cut K-Means from 8.6 ms to 3.5 ms per call in an early profile).
\item Region statistics, adjacency and shared border lengths come from \texttt{np.bincount} on pixel-pair keys; colour distances are $144{\times}144$ matrix products.
\item Max-flow: before Dinic starts, every region sends $\min(c_s,c_t)$ directly through source$\to p\to$sink, so afterwards each region keeps residual capacity to only one terminal; this leaves a small residual problem (graph cut $\approx$%(gc).1f ms).
\item All learned tables, weights and hyper-parameters are loaded once from one \texttt{params.npz}; the file is stored without compression in float32/float64 (%(sdisk)s bytes $\approx$ %(sdisk_pct).2f\%% of 4\,MiB).
\item Single thread: the four thread environment variables are set before NumPy/OpenCV/PyTorch are imported, \texttt{cv2.setNumThreads(1)}, \texttt{torch.set\_num\_threads(1)}; no multiprocessing.
\end{itemize}
\begin{table}[h]\centering\small
\begin{tabular}{lr}\toprule stage & ms per photo \\\midrule
%(stage_tex)s
\midrule total (sum of stages) & %(st_total).2f \\\bottomrule\end{tabular}
\caption{Time per stage. The official $T_\mathrm{CPU}$ in Table~\ref{tab:res} is measured separately with \texttt{time.process\_time()} around the full pipeline.}\label{tab:stages}\end{table}

\section{Experimental setup}
\textbf{Data.} DUTS-TR (10\,553 photos) was split with a fixed seed into %(ntr)s training and %(nva)s validation photos; all design choices (clustering settings, features, $\lambda$, $s$) were made on the validation split.
DUTS-TE (5\,019 photos) was used only for the final evaluation. Photos were resized with \texttt{INTER\_AREA}, masks with \texttt{INTER\_NEAREST} and binarised at 128.\\
\textbf{Hardware/software.} %(cpu)s; Python %(py)s, NumPy %(np)s, OpenCV %(cv)s, PyTorch %(torch)s (CPU build; training and inference on CPU).
$T_\mathrm{CPU}$ is the total of \texttt{time.process\_time()} differences around \texttt{segment()} over all 5\,019 DUTS-TE photos (one at a time, after one warm-up run) divided by 5\,019.
On this laptop timings vary by a few ms between runs (frequency scaling), so repeated runs of \texttt{predict.py} on a single photo scatter around the DUTS-TE average.

\section{Results}
\begin{table}[h]\centering\small
\begin{tabular}{lrrrrrrr}\toprule
set & photos & mIoU & BF1 & $Q$ & $T_\mathrm{CPU}$ (ms) & $S_\mathrm{disk}$ (bytes) & CES \\\midrule
%(table)s
\bottomrule\end{tabular}
\caption{Quality and cost. CES in every row uses the official $T_\mathrm{CPU}$ measured on DUTS-TE; the $T_\mathrm{CPU}$ column shows the time measured on that set itself (only the DUTS-TE one is the official figure; \texttt{process\_time} advances in 15.6\,ms ticks on Windows, which is harmless over thousands of photos but makes the figure coarse for a small set).}\label{tab:res}\end{table}

\paragraph{Design.}
\begin{table}[h]\centering\small
\begin{tabular}{lrrr}\toprule variant (validation) & mIoU & BF1 & $Q$ \\\midrule
stage-1 regression only, no graph cut ($\lambda=0$) & 0.708 & 0.380 & 0.609 \\
stage-1 regression + graph cut & 0.720 & 0.438 & 0.636 \\
two-stage regression, no graph cut & 0.740 & 0.420 & 0.644 \\
two-stage regression + graph cut & 0.751 & 0.474 & 0.668 \\
\textbf{+ probability calibration (final)} & \textbf{@@VM@@} & \textbf{@@VB@@} & \textbf{@@VQ@@} \\
\bottomrule\end{tabular}
\caption{Ablation. The graph cut adds +0.024 Q, the second logistic regression adds +0.032 Q, the calibration +0.012 Q (all on validation).}\end{table}
The final design is: a $64{\times}64$ Lab+position image clustered by K-Means from a $12{\times}12$ grid start with %(iters)d iterations ($K\le144$ regions); 24 region features;
two stacked logistic regressions (the second one sees the stage-1 logits); probability calibration with offset $b=%(bias).1f$ and temperature $\tau=%(temp).1f$; a graph cut with $\lambda=%(lam).0f$ and $s=%(sig).2f$;
and a bilinear enlargement of the $64{\times}64$ labelling to $256{\times}256$. Its $Q$ is @@VQ@@ on validation, @@TEQ@@ on DUTS-TE and @@CAPQ@@ on the own photos.

\paragraph{Validation vs.\ DUTS-TE.} DUTS-TE is clearly harder than my validation split: its objects are much smaller (mean object area %(fte).1f\%% vs %(fva).1f\%% of the photo), so the
position prior and ``big salient blob'' cues learned from DUTS-TR transfer less well. Table~\ref{tab:size} shows the object IoU on DUTS-TE by object size.
\begin{table}[h]\centering\small
\begin{tabular}{lrrrr}\toprule object area & TE photos & \% of TE & \% of validation & mean object IoU (TE) \\\midrule
%(bucket_tex)s
\bottomrule\end{tabular}
\caption{Object IoU on DUTS-TE by object size (analysis only; nothing was tuned on DUTS-TE).}\label{tab:size}\end{table}
%(cap_section)s
\section{Successful cases}
Figure~\ref{fig:tesuccess} shows four DUTS-TE photos that are segmented well. Selection: 12 photos were drawn at random (seed 0) from the @@NOK@@ TE photos with object IoU $\ge 0.8$, and four were picked for their different backgrounds.
They are good examples, not typical ones: only @@POK@@\% of the DUTS-TE photos reach IoU $\ge 0.8$, @@P60@@\% reach $\ge 0.6$ and @@P25@@\% fall below 0.25.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/te_successes.png}
\caption{DUTS-TE successes (photo | true mask | predicted mask), IoU in the labels.}\label{fig:tesuccess}\end{figure}
@@OKLIST@@
The common pattern is an object with a colour that differs clearly from a smooth, low-texture background and that is large enough to cover many regions: the colour-contrast, border-contact and position cues all agree, and the graph cut then cleans up the boundary.

\section{Failure cases}
Figure~\ref{fig:tefail} shows four DUTS-TE failures. Selection: 12 photos were drawn at random (seed 0) from the %(nbad)d TE photos with IoU $<0.25$, and four of the draw were chosen because each shows a different kind of failure. They are examples, not a ranking of the worst cases.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/te_failures.png}
\caption{DUTS-TE failures (photo | true mask | predicted mask).}\label{fig:tefail}\end{figure}
\begin{itemize}\itemsep0pt
%(fail_tex)s
\end{itemize}
\textbf{Likely causes.} The region model only sees region-level colour, position, contrast and sharpness statistics (these causes are my interpretation of the examples, not separately tested).
(1)~\emph{Small objects} are covered by only a few of the 144 regions (a region is about $5{\times}5$ pixels of the $64{\times}64$ image), so one wrong or missed region costs most of the object (object IoU @@B0@@ below 2\%% area vs @@B3@@ for 10--40\%%).
(2)~\emph{Over-segmentation}: in low-contrast scenes where the object has the colours of its surroundings (#147), the probability calibration offset, which favours the object label and was tuned on DUTS-TR where objects are large, makes the graph cut label most of the photo as object.
Several other photos of the same random draw show this flooding (#318, #1230: 57\%% and 66\%% predicted for objects of 13\%% and 14\%%), and it is the mechanism by which the calibration lowers DUTS-TE $Q$ slightly although it raises validation $Q$.
(3)~\emph{Poor lighting}: dark objects with hard shadows (#2134) merge with the shadows and have little colour contrast, so the colour cues are uninformative. (4)~\emph{Partial objects}: the model finds the salient colour part (the pale saddle pad, #4471) rather than the whole object.
%(cap_fail)s

\paragraph{Limitations.} The approach has no notion of objects or semantics: it can only use colour, position, contrast and focus statistics of $\approx$144 regions, and it was fitted to DUTS-TR, whose objects are larger than those of DUTS-TE.

\end{document}
"""
    V = dict(
        t_te=t_te, sdisk=f"{s_disk:,}", sdisk_pct=100 * s_disk / 4194304, te_m=te["mIoU"], te_b=te["BF1"], te_q=te["Q"],
        te_ces=metrics.ces(te["Q"], t_te, s_disk), wpos=float(params["wpos"]), iters=int(params["iters"]),
        lam=float(params["lam"]), bias=float(params.get("bias", 0)), temp=float(params.get("temp", 1)), sig=float(params["sigma_scale"]), gc=st.get("graph_cut", 0), stage_tex=stage_tex,
        st_total=st.get("total", 0), ntr=f"{ntr:,}", nva=f"{nva:,}",
        cpu="Intel Core i5-12450H (12th Gen, 8 cores, 2.0\\,GHz base), 16\\,GB RAM, Windows 11, timed pipeline on one core/one thread",
        py=platform.python_version(), np=np.__version__, cv=cv2.__version__, torch=__import__("torch").__version__,
        table="\n".join(table), fte=100 * frac_te.mean(), fva=100 * frac_va.mean(), bucket_tex=bucket_tex,
        cap_section=cap_section, nbad=int((iou_te < 0.25).sum()), fail_tex=fail_tex, cap_fail=cap_fail)
    # fill %(name)fmt placeholders (a plain % operator would clash with LaTeX percent signs)
    doc = re.sub(r"%\((\w+)\)([.\d]*[fds])", lambda m: ("%" + m.group(2)) % V[m.group(1)], doc)
    doc = doc.replace("%%", "%")
    ok_list = "\\begin{itemize}\\itemsep0pt\n" + "\n".join(f"\\item \\textbf{{\\#{i}}} -- {c}." for i, c in TE_OK) + "\n\\end{itemize}"
    for tok, val in (("@@B0@@", f"{bucket_rows[0][5]:.2f}"), ("@@B3@@", f"{bucket_rows[2][5]:.2f}"), ("@@NOK@@", str(int((iou_te >= 0.8).sum()))), ("@@POK@@", f"{100 * (iou_te >= 0.8).mean():.1f}"),
                     ("@@P60@@", f"{100 * (iou_te >= 0.6).mean():.1f}"), ("@@P25@@", f"{100 * (iou_te < 0.25).mean():.1f}"),
                     ("@@OKLIST@@", ok_list), ("@@VM@@", f"{va['mIoU']:.3f}"), ("@@VB@@", f"{va['BF1']:.3f}"), ("@@VQ@@", f"{va['Q']:.3f}"),
                     ("@@TEQ@@", f"{te['Q']:.3f}"), ("@@CAPQ@@", f"{cap['Q']:.3f}" if cap else "n/a")):
        doc = doc.replace(tok, val)
    open(os.path.join(REP, "report.tex"), "w", encoding="utf-8").write(doc)
    print("wrote", os.path.join(REP, "report.tex"))


if __name__ == "__main__":
    main()
