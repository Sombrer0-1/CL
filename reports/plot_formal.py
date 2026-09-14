"""Plot ER-context formal means from formal_context_means.csv. Not a verdict."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MEANS = ROOT / "reports" / "formal_context_means.csv"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    r
    for r in csv.DictReader(MEANS.open(encoding="utf-8"))
    if r.get("algorithm_base") == "er"
]
methods = sorted({r["method_id"] for r in rows})
datasets = ["splitcifar10", "splitcifar100", "core50_ni", "core50_nc", "core50_nic"]
present = [d for d in datasets if any(r["dataset"] == d for r in rows)]
colors = {
    "er_static": "#4c78a8",
    "orion_formula": "#f58518",
    "max_p_reconstructed": "#54a24b",
    "lr_reconstructed": "#e45756",
    "max_a_reconstructed": "#b279a2",
}


def _get(dataset, method):
    for r in rows:
        if r["dataset"] == dataset and r["method_id"] == method:
            return r
    return None


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = range(len(present))
width = 0.16
for ax, metric, ylabel in (
    (axes[0], "p_mean", "P_diag mean"),
    (axes[1], "t_mean", "online_total_s mean"),
):
    for i, method in enumerate(methods):
        ys = []
        yerr = []
        for d in present:
            r = _get(d, method)
            if r is None:
                ys.append(float("nan"))
                yerr.append(0.0)
            else:
                ys.append(float(r[metric]))
                std_key = "p_std" if metric == "p_mean" else "t_std"
                yerr.append(float(r[std_key] or 0.0))
        offs = [xi + (i - (len(methods) - 1) / 2) * width for xi in x]
        ax.bar(
            offs,
            ys,
            width=width,
            yerr=yerr,
            label=method,
            color=colors.get(method, "gray"),
            capsize=3,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("ER paper_feedback (incomplete matrix)")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "formal_p_and_time.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5))
for r in rows:
    pstd = float(r["p_std"] or 0.0)
    tstd = float(r["t_std"] or 0.0)
    ax.errorbar(
        float(r["t_mean"]),
        float(r["p_mean"]),
        xerr=tstd,
        yerr=pstd,
        fmt="o",
        label=f"{r['dataset']}/{r['method_id']}",
        color=colors.get(r["method_id"], "gray"),
    )
ax.set_xlabel("online_total_s mean")
ax.set_ylabel("P_diag mean")
ax.set_title("ER Pareto sketch (incomplete)")
ax.legend(fontsize=6, loc="best")
fig.tight_layout()
fig.savefig(OUT / "formal_pareto_p_time.png", dpi=120)
plt.close(fig)
print("wrote", OUT)
