"""Metrics (AUROC / AUPRC / TPR@FPR5%) and the diagnostic figures."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

import data
from data import H, PROCESSES, CASES

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")

SCORES = ["S_max", "S_band", "S_mean", "S_gls_pert", "S_gls_emp",
          "S_chi2", "S_maha"]
# validated categorical slots 1-3 (all-pairs, light mode)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})


# ------------------------------------------------------------------ metrics

def metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (proc, model, seed), g in df.groupby(["process", "model", "seed"]):
        neg = g[g.case == "N0"]
        for case in CASES:
            if case == "N0":
                continue
            pos = g[g.case == case]
            if pos.empty:
                continue
            for s in SCORES:
                a, b = neg[s].to_numpy(), pos[s].to_numpy()
                y = np.r_[np.zeros(a.size), np.ones(b.size)]
                v = np.r_[a, b]
                thr = np.quantile(a, 0.95)          # 5% false-positive budget
                out.append(dict(
                    process=proc, model=model, case=case, seed=seed, score=s,
                    auroc=roc_auc_score(y, v),
                    auprc=average_precision_score(y, v),
                    tpr_at_fpr5=float((b > thr).mean()),
                    n0_mean=float(a.mean()), case_mean=float(b.mean())))
    m = pd.DataFrame(out)
    agg = (m.groupby(["process", "model", "case", "score"])
             .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                  auprc=("auprc", "mean"), auprc_sd=("auprc", "std"),
                  tpr5=("tpr_at_fpr5", "mean"), tpr5_sd=("tpr_at_fpr5", "std"),
                  n0_mean=("n0_mean", "mean"), case_mean=("case_mean", "mean"))
             .reset_index())
    return m, agg


# ------------------------------------------------------------------ figures

def fig_pit(fd, models_):
    keys = [k.split("|") for k in fd["pit_keys"]]
    hs = fd["diag_h"]
    for model in models_:
        fig, axes = plt.subplots(len(PROCESSES), len(hs),
                                 figsize=(2.2 * len(hs), 1.9 * len(PROCESSES)),
                                 sharex=True)
        for i, proc in enumerate(PROCESSES):
            sel = [j for j, k in enumerate(keys) if k[0] == model and k[1] == proc]
            v = np.concatenate([fd["pit_vals"][j] for j in sel], 0)
            for j, h in enumerate(hs):
                ax = axes[i, j]
                ax.hist(v[:, j], bins=20, range=(0, 1), color=C1,
                        edgecolor="#fcfcfb", linewidth=0.5)
                ax.axhline(v.shape[0] / 20, color=INK2, lw=1.2, ls="--")
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"h = {h}", color=INK)
                if j == 0:
                    ax.set_ylabel(proc, color=INK, fontweight="bold")
        fig.suptitle(f"{model}: PIT of normal windows (dashed = uniform)",
                     color=INK, y=1.0)
        fig.tight_layout()
        fig.savefig(f"{FIG}/pit_{model}.png", bbox_inches="tight")
        plt.close(fig)


def fig_psi(fd, models_):
    keys = [k.split("|") for k in fd["psi_keys"]]
    fig, axes = plt.subplots(len(models_), len(PROCESSES),
                             figsize=(3.0 * len(PROCESSES), 2.5 * len(models_)),
                             squeeze=False)
    h = np.arange(1, H + 1)
    for r, model in enumerate(models_):
        for c, proc in enumerate(PROCESSES):
            ax = axes[r][c]
            sel = [j for j, k in enumerate(keys) if k[0] == model and k[1] == proc]
            psi = fd["psi_vals"][sel].mean(0)[1:]
            th = data.theoretical_psi(proc)[1:]
            ax.plot(h, th, color=C2, lw=2.4, label="theory")
            ax.plot(h, psi, color=C1, lw=2.0, ls="--", label="perturbation")
            ax.set_title(f"{model} / {proc}", color=INK)
            ax.set_xlabel("horizon h"); ax.set_ylim(-0.25, 1.25)
            if c == 0:
                ax.set_ylabel(r"$\psi_h$")
            if r == 0 and c == 0:
                ax.legend(frameon=False, labelcolor=INK2)
    fig.suptitle(r"Impulse response $\psi_h$: input perturbation vs theory",
                 color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/psi_response.png", bbox_inches="tight")
    plt.close(fig)


def fig_scores(df, models_):
    show = ["S_max", "S_mean", "S_gls_pert"]
    for model in models_:
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2))
        for i, proc in enumerate(["WN", "RW"]):
            for j, s in enumerate(show):
                ax = axes[i, j]
                g = df[(df.model == model) & (df.process == proc)]
                a = g[g.case == "N0"][s]; b = g[g.case == "A1"][s]
                lo, hi = 0, max(a.quantile(.999), b.max()) * 1.05
                bins = np.linspace(lo, hi, 45)
                ax.hist(a, bins=bins, color=C1, alpha=.85, label="N0 normal")
                ax.hist(b, bins=bins, color=C2, alpha=.85, label="A1 offset")
                ax.axvline(a.quantile(.95), color=INK2, ls="--", lw=1.2)
                ax.set_yticks([])
                ax.set_title(f"{proc} / {s}", color=INK)
                if i == 1:
                    ax.set_xlabel("score")
                if i == 0 and j == 0:
                    ax.legend(frameon=False, labelcolor=INK2, fontsize=8)
        fig.suptitle(f"{model}: N0 vs A1 score distributions "
                     f"(dashed = 95th pct of N0)", color=INK)
        fig.tight_layout()
        fig.savefig(f"{FIG}/scores_A1_{model}.png", bbox_inches="tight")
        plt.close(fig)


def fig_neff(diag, models_):
    d = (diag.groupby(["process", "model"])
             .agg(pert=("n_eff_pert", "mean"), emp=("n_eff_emp", "mean"))
             .reset_index())
    th = {}
    for p in PROCESSES:
        import scoring
        R = scoring.r_from_psi(np.broadcast_to(data.theoretical_psi(p), (1, H + 1)))
        th[p] = float(np.ones(H) @ np.linalg.solve(R[0], np.ones(H)))

    fig, axes = plt.subplots(1, len(models_), figsize=(4.6 * len(models_), 3.4),
                             squeeze=False, sharey=True)
    x = np.arange(len(PROCESSES)); w = 0.27
    for k, model in enumerate(models_):
        ax = axes[0][k]
        sub = d[d.model == model].set_index("process").reindex(PROCESSES)
        series = [("theory", [th[p] for p in PROCESSES], C3),
                  ("R_pert", sub["pert"].to_numpy(), C1),
                  ("R_emp", sub["emp"].to_numpy(), C2)]
        for i, (lab, v, col) in enumerate(series):
            ax.bar(x + (i - 1) * w, v, w * 0.9, color=col, label=lab)
            for xi, vi in zip(x + (i - 1) * w, v):      # relief: direct labels
                ax.text(xi, vi + 0.6, f"{vi:.1f}", ha="center", fontsize=7,
                        color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(PROCESSES)
        ax.set_title(model, color=INK)
        if k == 0:
            ax.set_ylabel(r"$n_{eff} = \mathbf{1}^T R^{-1} \mathbf{1}$")
            ax.legend(frameon=False, labelcolor=INK2)
    fig.suptitle(r"Effective number of independent horizons ($H=32$)", color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/n_eff.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(f"{RES}/scores.csv")
    diag = pd.read_csv(f"{RES}/diagnostics.csv")
    fd = np.load(f"{RES}/figdata.npz")
    models_ = list(dict.fromkeys(df.model))

    per_seed, agg = metrics(df)
    per_seed.to_csv(f"{RES}/metrics_per_seed.csv", index=False)
    agg.to_csv(f"{RES}/metrics.csv", index=False)

    fig_pit(fd, models_); fig_psi(fd, models_)
    fig_scores(df, models_); fig_neff(diag, models_)
    print(f"metrics rows {len(agg)};  figures -> {FIG}")


if __name__ == "__main__":
    main()
