"""light24_v1 paired analysis and figures from current-identity runs.

Inputs: reports/light24/coverage.csv (+ run artifacts under runs/<run_id>/).
Outputs: reports/light24/*.csv tables and reports/light24/figures/*.png.

This script is descriptive only; it does not modify runs or select parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "reports" / "light24"
FIG = IN / "figures"
RUNS = ROOT / "runs"


def load_coverage() -> pd.DataFrame:
    cov = pd.read_csv(IN / "coverage.csv")
    return cov


def run_artifacts(run_id: str) -> dict:
    d = RUNS / str(run_id)
    out = {"run_id": run_id}
    if not d.exists():
        return out
    em = pd.read_csv(d / "experience_metrics.csv") if (d / "experience_metrics.csv").exists() else None
    rt = pd.read_csv(d / "resource_trace.csv") if (d / "resource_trace.csv").exists() else None
    if em is not None and len(em):
        out.update(
            learning_s_sum=em.learning_s.sum(),
            evaluation_s_sum=em.evaluation_s.sum(),
            memory_mib_peak=em.memory_mib.max(),
            prefetch_wait_sum=em.prefetch_wait_s.fillna(0).sum() if em.prefetch_wait_s.notna().any() else 0.0,
            prefetch_batches_sum=em.prefetch_batches.fillna(0).sum() if em.prefetch_batches.notna().any() else 0.0,
            replay_visits_sum=em.replay_visits.fillna(0).sum() if em.replay_visits.notna().any() else 0.0,
            new_visits_sum=em.new_sgd_visits.fillna(0).sum() if em.new_sgd_visits.notna().any() else 0.0,
        )
    if rt is not None and len(rt):
        out.update(
            proc_rss_peak_mib=rt.proc_rss_bytes.max() / 2**20,
            gpu_alloc_peak_mib=rt.gpu_alloc_peak_bytes.max() / 2**20,
            gpu_global_peak_mib=rt.gpu_global_used_bytes.max() / 2**20,
            swap_used_peak_mib=rt.swap_used_bytes.max() / 2**20,
        )
    ct_path = d / "control_trace.jsonl"
    if ct_path.exists():
        ct = pd.DataFrame([json.loads(l) for l in ct_path.open()])
        c = ct.dropna(subset=["applied_new_batch"]) if "applied_new_batch" in ct.columns else pd.DataFrame()
        if len(c):
            out.update(
                urge_max=float(c.urge.max()),
                urge_mean=float(c.urge.mean()),
                applied_batch_min=float(c.applied_new_batch.min()),
                applied_batch_max=float(c.applied_new_batch.max()),
                applied_replay_min=float(c.applied_replay.min()),
                applied_replay_max=float(c.applied_replay.max()),
                n_mode_switches=int((c.applied_optimizer_mode != c.applied_optimizer_mode.shift(1)).sum() - 1)
                if len(c) > 1
                else 0,
                controller_s_sum=float(c.controller_s.sum()),
                reconfigure_s_sum=float(c.reconfigure_s.sum()),
                guard_nonok=int((c.guard_reason != "ok").sum()),
            )
    return out


def paired(cov: pd.DataFrame, exp_a: str, var_a: str, exp_b: str, var_b: str, label: str, dataset: str) -> list[dict]:
    rows = []
    a = cov[(cov.experiment == exp_a) & (cov.variant == var_a)].set_index("seed")
    b = cov[(cov.experiment == exp_b) & (cov.variant == var_b)].set_index("seed")
    for seed in sorted(set(a.index) & set(b.index)):
        ra, rb = a.loc[seed], b.loc[seed]
        if ra.status != "completed" or rb.status != "completed":
            status = f"{ra.status}/{rb.status}"
            rows.append(dict(comparison=label, dataset=dataset, seed=seed, status=status))
            continue
        rows.append(
            dict(
                comparison=label,
                dataset=dataset,
                seed=seed,
                status="completed",
                dP=float(ra.P) - float(rb.P),
                dS=float(ra.S) - float(rb.S),
                d_online_s=float(ra.online_s) - float(rb.online_s),
                P_a=float(ra.P), P_b=float(rb.P),
                S_a=float(ra.S), S_b=float(rb.S),
                t_a=float(ra.online_s), t_b=float(rb.online_s),
            )
        )
    return rows


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cov = load_coverage()
    done = cov[cov.run_id.notna()].copy()

    # ---- per-run artifacts table -------------------------------------------------
    arts = pd.DataFrame([run_artifacts(rid) for rid in done.run_id])
    merged = done.merge(arts, on="run_id", how="left")
    keep = [
        "experiment", "variant", "dataset", "seed", "status", "run_id",
        "P", "S", "online_s", "learning_s_sum", "evaluation_s_sum",
        "controller_s_sum", "reconfigure_s_sum", "prefetch_wait_sum", "prefetch_batches_sum",
        "replay_visits_sum", "new_visits_sum", "memory_mib_peak", "proc_rss_peak_mib",
        "gpu_alloc_peak_mib", "gpu_global_peak_mib", "swap_used_peak_mib",
        "urge_max", "urge_mean", "applied_batch_min", "applied_batch_max",
        "applied_replay_min", "applied_replay_max", "n_mode_switches", "guard_nonok",
    ]
    merged[[c for c in keep if c in merged.columns]].to_csv(IN / "run_artifacts.csv", index=False)

    # ---- paired differences ------------------------------------------------------
    rows: list[dict] = []
    for ds in ("splitcifar100", "core50_nc"):
        base = f"{ds}_er_static"
        rows += paired(cov, "L_core", f"{ds}_orion_formula", "L_core", base, "orion_formula - er_static", ds)
        rows += paired(cov, "L_core", f"{ds}_orion_batch_replay", "L_core", base, "orion_batch_replay - er_static", ds)
        rows += paired(cov, "L_selected", ds, "L_core", base, "selected_static - er_static", ds)
        rows += paired(cov, "L_selected", ds, "L_core", f"{ds}_orion_formula", "selected_static - orion_formula", ds)
    for pref in ("latency", "ps"):
        rows += paired(cov, "L_preference", f"core50_nc_{pref}", "L_core", "core50_nc_orion_formula",
                       f"pref_{pref} - orion_balanced", "core50_nc")
    for thr in ("0.025", "0.1"):
        for ds in ("splitcifar100", "core50_nc"):
            rows += paired(cov, "L_sensitivity", f"{ds}_thr{thr}", "L_core", f"{ds}_orion_formula",
                           f"thr{thr} - thr0.05_orion", ds)
    rows += paired(cov, "L_prefetch", "core50_nc_on", "L_prefetch", "core50_nc_off", "prefetch on - off", "core50_nc")
    for scen in ("ic", "il", "wc"):
        for s in (0, 1, 2):
            sel = cov[(cov.experiment == "L_endless") & (cov.seed == s)]
            a = sel[sel.variant.str.startswith(f"endless_{scen}_orion_formula")]
            b = sel[sel.variant.str.startswith(f"endless_{scen}_er_static")]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                if a.status == "completed" and b.status == "completed":
                    rows.append(dict(comparison="endless orion - er", dataset=f"endless_{scen}", seed=s,
                                     status="completed", dP=float(a.P) - float(b.P), dS=float(a.S) - float(b.S),
                                     d_online_s=float(a.online_s) - float(b.online_s)))
    pd.DataFrame(rows).to_csv(IN / "paired_diffs.csv", index=False)

    # ---- budget summary ----------------------------------------------------------
    bud = cov[cov.experiment == "L_budget"].copy()
    bud["quota_mib"] = bud.variant.str.extract(r"(\d+)mib").astype(float)
    bud["method"] = bud.variant.str.extract(r"(er_static|orion_formula)", expand=False)
    if len(bud):
        agg = bud.groupby(["quota_mib", "method"]).agg(
            n=("status", "size"),
            n_completed=("status", lambda s: (s == "completed").sum()),
            n_oom=("status", lambda s: (s == "cuda_oom").sum()),
            P_mean=("P", lambda s: s.dropna().mean()),
            S_mean=("S", lambda s: s.dropna().mean()),
            online_s_mean=("online_s", lambda s: s.dropna().mean()),
        ).reset_index()
        agg.to_csv(IN / "budget_summary.csv", index=False)

    # ---- figures -----------------------------------------------------------------
    plot_main(cov)
    plot_control(merged)
    if len(bud):
        plot_budget(bud, merged)
    plot_prefetch(cov, merged)
    plot_sens_pref(cov)
    plot_endless(cov)
    print("analysis written to", IN)


def plot_main(cov: pd.DataFrame) -> None:
    groups = [
        ("splitcifar100_er_static", "L_core", "splitcifar100_er_static", "ER static (b16/r200)"),
        ("splitcifar100_orion", "L_core", "splitcifar100_orion_formula", "Orion"),
        ("splitcifar100_selected", "L_selected", "splitcifar100", "ER selected (b16/r2000)"),
        ("core50_er_static", "L_core", "core50_nc_er_static", "ER static (b16/r200)"),
        ("core50_orion", "L_core", "core50_nc_orion_formula", "Orion"),
        ("core50_selected", "L_selected", "core50_nc", "ER selected (b16/r2000)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"splitcifar100": ["#4c78a8", "#f58518", "#54a24b"], "core50_nc": ["#4c78a8", "#f58518", "#54a24b"]}
    for ax, ds in zip(axes, ("splitcifar100", "core50_nc")):
        i = 0
        for _, exp, var, label in groups:
            if not var.startswith(ds):
                continue
            g = cov[(cov.experiment == exp) & (cov.variant == var) & (cov.status == "completed")]
            if not len(g):
                continue
            c = colors[ds][i % 3]
            ax.scatter(g.online_s, g.P, s=45, color=c, label=label)
            ax.errorbar(
                g.online_s.mean(), g.P.mean(),
                xerr=g.online_s.std(ddof=1) if len(g) > 1 else 0,
                yerr=g.P.std(ddof=1) if len(g) > 1 else 0,
                fmt="o", color=c, capsize=3,
            )
            i += 1
        ax.set_title({"splitcifar100": "SplitCIFAR100 (10 exp)", "core50_nc": "CORe50-NC (9 exp)"}[ds])
        ax.set_xlabel("online training time per run (s)")
        ax.set_ylabel("P_diag (plasticity)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Learning quality vs training time (3 seeds, mean±sd; paper-feedback protocol)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_main_pareto.png", dpi=150)
    plt.close(fig)


def plot_control(merged: pd.DataFrame) -> None:
    adap = merged[(merged.experiment == "L_core") & (merged.variant.str.contains("orion_formula"))]
    if not len(adap):
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, ds in enumerate(("splitcifar100", "core50_nc")):
        g = adap[(adap.dataset == ds) & (adap.seed == 0)]
        if not len(g):
            continue
        rid = g.iloc[0].run_id
        ct = pd.DataFrame([json.loads(l) for l in (RUNS / str(rid) / "control_trace.jsonl").open()])
        c = ct.dropna(subset=["applied_new_batch"])
        f = pd.DataFrame(list(c.factors))
        ax = axes[0][j]
        ax.plot(c.t, c.urge, "o-", label="URGE")
        ax.axhline(0.05, color="r", ls="--", label="Thr0=0.05")
        ax.set_yscale("log")
        ax.set_title(f"{ds}: URGE vs threshold (seed0)")
        ax.set_xlabel("experience")
        ax.legend(fontsize=8)
        ax = axes[1][j]
        for col, lbl in (("factor_p", "P factor"), ("factor_s", "S factor"), ("factor_l", "L factor"), ("factor_m", "M factor")):
            ax.plot(c.t, f[col], "o-", label=lbl)
        ax.set_yscale("log")
        ax.set_title(f"{ds}: sigmoid factors (seed0)")
        ax.set_xlabel("experience")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_control_trace.png", dpi=150)
    plt.close(fig)


def plot_budget(bud: pd.DataFrame, merged: pd.DataFrame) -> None:
    arts = merged[merged.experiment == "L_budget"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for method, marker in (("er_static", "s"), ("orion_formula", "o")):
        sub = bud[bud.method == method].groupby("quota_mib").agg(
            comp=("status", lambda s: (s == "completed").sum()), total=("status", "size")).reset_index()
        ax.plot(sub.quota_mib, sub.comp / sub.total, marker + "-", label=f"{method} (n={int(sub.total.sum())})")
    ax.set_xlabel("allocator quota (MiB)")
    ax.set_ylabel("completed fraction")
    ax.set_title("Budget feasibility (3 seeds each)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for method in ("er_static", "orion_formula"):
        sub = arts[arts.variant.str.contains(method) & arts.gpu_alloc_peak_mib.notna()]
        if len(sub):
            sub = sub.copy()
            sub["quota"] = sub.variant.str.extract(r"(\d+)mib").astype(float)
            grp = sub.groupby("quota").gpu_alloc_peak_mib.agg(["mean", "std", "count"])
            ax.errorbar(grp.index, grp["mean"], yerr=grp["std"].fillna(0), fmt="o", label=method)
    ax.set_xlabel("allocator quota (MiB)")
    ax.set_ylabel("peak PyTorch allocated (MiB)")
    ax.set_title("Peak allocator usage when completed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_budget.png", dpi=150)
    plt.close(fig)


def plot_prefetch(cov: pd.DataFrame, merged: pd.DataFrame) -> None:
    pf = cov[cov.experiment == "L_prefetch"]
    if not len(pf):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric, label in ((axes[0], "P", "P_diag"), (axes[1], "S", "S_initial"), (axes[2], "online_s", "online time (s)")):
        data = [pf[pf.variant == v][metric].dropna().values for v in ("core50_nc_off", "core50_nc_on")]
        if any(len(d) == 0 for d in data):
            continue
        ax.boxplot(data, tick_labels=["prefetch off", "prefetch on"])
        for i, d in enumerate(data):
            ax.scatter([i + 1] * len(d), d, color="k", s=12, zorder=3)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    fig.suptitle("Prefetch ablation: CORe50-NC static ER fixed trajectory (3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_prefetch.png", dpi=150)
    plt.close(fig)


def plot_sens_pref(cov: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ("splitcifar100", "core50_nc")):
        pts = {"0.025": [], "0.05": [], "0.1": []}
        sens = cov[(cov.experiment == "L_sensitivity") & (cov.variant.str.startswith(ds))]
        for thr in ("0.025", "0.1"):
            pts[thr] = sens[sens.variant == f"{ds}_thr{thr}"][["P", "S"]].dropna().values
        core = cov[(cov.experiment == "L_core") & (cov.variant == f"{ds}_orion_formula")][["P", "S"]].dropna().values
        pts["0.05"] = core
        for thr, marker in (("0.025", "s"), ("0.05", "o"), ("0.1", "^")):
            if len(pts[thr]):
                ax.scatter(pts[thr][:, 0], pts[thr][:, 1], marker=marker, label=f"Thr0={thr}")
        ax.set_title(f"{ds}: threshold sensitivity (Orion, 3 seeds)")
        ax.set_xlabel("P_diag")
        ax.set_ylabel("S_initial")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensitivity.png", dpi=150)
    plt.close(fig)

    pref = cov[cov.experiment == "L_preference"]
    if len(pref):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bal = cov[(cov.experiment == "L_core") & (cov.variant == "core50_nc_orion_formula")]
        groups = [("balanced (core)", bal)]
        for name in ("latency", "ps"):
            groups.append((name, pref[pref.variant == f"core50_nc_{name}"]))
        width = 0.25
        for k, metric in enumerate(("P", "S", "online_s")):
            means, sds = [], []
            for _, g in groups:
                v = g[metric].dropna()
                means.append(v.mean() if len(v) else float("nan"))
                sds.append(v.std(ddof=1) if len(v) > 1 else 0)
            ax.bar([i + (k - 1) * width for i in range(len(groups))], means, width, yerr=sds, capsize=3,
                   label={"P": "P_diag", "S": "S_initial", "online_s": "online time (s)"}[metric])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([n for n, _ in groups])
        ax.set_title("User preference: CORe50-NC Orion (3 seeds)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIG / "fig_preference.png", dpi=150)
        plt.close(fig)


def plot_endless(cov: pd.DataFrame) -> None:
    en = cov[cov.experiment == "L_endless"]
    if not len(en):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, scen in zip(axes, ("ic", "il", "wc")):
        sub = en[en.variant.str.contains(f"endless_{scen}_")]
        for meth, color in (("er_static", "#4c78a8"), ("orion_formula", "#f58518")):
            g = sub[sub.variant.str.endswith(meth)]
            v = g[["P", "S"]].dropna()
            if len(v):
                ax.scatter(v.P, v.S, color=color, label=meth)
                ax.scatter([v.P.mean()], [v.S.mean()], color=color, s=160, marker="*", edgecolor="k")
        ax.set_title(f"Endless-{scen.upper()} (dev split, 3 seeds)")
        ax.set_xlabel("P_diag")
        ax.set_ylabel("S_initial")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Endless scenarios: star = seed mean")
    fig.tight_layout()
    fig.savefig(FIG / "fig_endless.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
