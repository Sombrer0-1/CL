"""pressure_v2 reports. Never writes into reports/light24."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orion_repro.runner.spec import load_yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pressure_v2"


def write_csv(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _spec(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "resolved_config.yaml.json"
    if path.is_file():
        return json.loads(path.read_text())
    return None


def collect_attempts() -> list[dict[str, Any]]:
    attempts = []
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        spec = _spec(path.parent)
        if not spec or spec.get("study_id") != "pressure_v2":
            continue
        summary = _summary(path.parent)
        attempts.append(
            {
                "run_id": path.parent.name,
                "status": summary.get("status"),
                "experiment": spec.get("experiment_id"),
                "group": spec.get("pressure_group") or spec.get("pressure_role"),
                "dataset": spec["dataset"]["name"],
                "method": spec.get("method_id"),
                "seed": spec["seeds"]["model"],
                "quota_bytes": spec["budget"].get("limit_bytes"),
                "eval_batch": spec["training"]["eval_batch"],
                "P": summary.get("p_diag"),
                "S": summary.get("s_initial"),
                "online_s": summary.get("online_total_s"),
                "failure_phase": summary.get("failure_phase"),
                "unannounced_transition": summary.get("unannounced_transition"),
                "n_trained": summary.get("n_experiences_trained"),
                "n_evaluated": summary.get("n_experiences_evaluated") or summary.get("n_experiences_run"),
            }
        )
    return attempts


def paired_diffs(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = (row.get("group"), row.get("seed"), row.get("quota_bytes"))
        by[key][row["method"]] = row
    out = []
    for key, methods in sorted(by.items()):
        if left in methods and right in methods:
            a, b = methods[left], methods[right]
            item = {"group": key[0], "seed": key[1], "quota_bytes": key[2], "left": left, "right": right}
            for metric in ("P", "S", "online_s"):
                if a.get(metric) is None or b.get(metric) is None:
                    item[f"d_{metric}"] = None
                else:
                    item[f"d_{metric}"] = float(a[metric]) - float(b[metric])
            out.append(item)
    return out


def _means(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    return mean(vals), (stdev(vals) if len(vals) > 1 else None)


def plot_ps(rows: list[dict[str, Any]], path: Path) -> None:
    completed = [r for r in rows if r.get("status") == "completed" and r.get("P") is not None]
    if not completed:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = sorted({r["method"] for r in completed})
    for method in methods:
        xs = [float(r["S"]) for r in completed if r["method"] == method]
        ys = [float(r["P"]) for r in completed if r["method"] == method]
        ax.scatter(xs, ys, label=method)
    ax.set_xlabel("Stability S_initial")
    ax.set_ylabel("Plasticity P_diag")
    ax.set_title("pressure_v2 P vs S by method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_time(rows: list[dict[str, Any]], path: Path) -> None:
    completed = [r for r in rows if r.get("status") == "completed" and r.get("online_s") is not None]
    if not completed:
        return
    methods = sorted({r["method"] for r in completed})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [[float(r["online_s"]) for r in completed if r["method"] == m] for m in methods]
    ax.boxplot(data, vert=True)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Online time (s)")
    ax.set_title("pressure_v2 online time by method")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_results(coverage: dict[str, Any], attempts: list[dict[str, Any]], means: list[dict[str, Any]]) -> None:
    frozen = ROOT / "experiments/pressure_v2/frozen_protocol.json"
    frozen_payload = json.loads(frozen.read_text()) if frozen.exists() else {}
    scenario = frozen_payload.get("scenario_coverage") or {}
    lines = [
        "# pressure_v2 结果",
        "",
        "本报告只覆盖第二阶段。第一阶段冻结证据见 `reports/light24/`，不被本脚本改写。",
        "",
        "## 场景覆盖",
        "",
        "| 场景 | 状态 |",
        "|---|---|",
    ]
    for key, value in scenario.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## 运行规模",
        "",
        f"- 尝试次数：{len(attempts)}",
        f"- completed：{sum(1 for a in attempts if a.get('status') == 'completed')}",
        f"- cuda_oom：{sum(1 for a in attempts if a.get('status') == 'cuda_oom')}",
        f"- 场景未建立而跳过：{sum(1 for a in attempts if a.get('status') == 'scenario_failed_to_construct')}",
        "",
        "## 均值表",
        "",
        "| group | method | n | P_mean | P_sd | S_mean | S_sd | online_s_mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in means:
        lines.append(
            "| {group} | {method} | {n} | {P_mean} | {P_sd} | {S_mean} | {S_sd} | {online_s_mean} |".format(
                **{k: row.get(k) for k in ("group", "method", "n", "P_mean", "P_sd", "S_mean", "S_sd", "online_s_mean")}
            )
        )
    lines += [
        "",
        "## 限定结论",
        "",
        "逐场景判定以 `docs/pressure_v2_acceptance.csv` 与下文分析为准。",
        "URGE 触发次数不是优越性指标；未触发、静态更优、OOM 更多都是有效结果。",
        "",
        f"冻结 protocol hash：`{frozen_payload.get('frozen_hash', 'not_frozen')}`",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attempts = collect_attempts()
    write_csv(OUT / "attempts.csv", attempts)
    formal = ROOT / "experiments/pressure_v2/formal.yaml"
    coverage_rows = []
    if formal.exists():
        matrix = load_yaml(formal)
        by_id = {(a.get("method"), a.get("seed"), a.get("group")): a for a in attempts}
        for rel in matrix["configs"]:
            spec = load_yaml(ROOT / rel)
            key = (spec["method_id"], spec["seeds"]["model"], spec.get("pressure_group"))
            hit = by_id.get(key) or {}
            coverage_rows.append(
                {
                    "config": rel,
                    "group": spec.get("pressure_group"),
                    "method": spec["method_id"],
                    "seed": spec["seeds"]["model"],
                    "status": hit.get("status", "not_run"),
                    "run_id": hit.get("run_id"),
                    "P": hit.get("P"),
                    "S": hit.get("S"),
                    "online_s": hit.get("online_s"),
                    "failure_phase": hit.get("failure_phase"),
                }
            )
        write_csv(OUT / "coverage.csv", coverage_rows)
    groups = defaultdict(list)
    for row in attempts:
        groups[(row.get("group"), row.get("method"))].append(row)
    means = []
    for (group, method), cells in sorted(groups.items()):
        valid = [
            c
            for c in cells
            if c.get("status") == "completed" and all(c.get(k) is not None for k in ("P", "S", "online_s"))
        ]
        p_m, p_sd = _means([float(c["P"]) for c in valid])
        s_m, s_sd = _means([float(c["S"]) for c in valid])
        t_m, t_sd = _means([float(c["online_s"]) for c in valid])
        means.append(
            {
                "group": group,
                "method": method,
                "n": len(valid),
                "n_attempts": len(cells),
                "P_mean": p_m,
                "P_sd": p_sd,
                "S_mean": s_m,
                "S_sd": s_sd,
                "online_s_mean": t_m,
                "online_s_sd": t_sd,
            }
        )
    write_csv(OUT / "means.csv", means)
    write_csv(OUT / "paired.csv", paired_diffs(attempts, "O11", "S0") + paired_diffs(attempts, "O11", "S_star") + paired_diffs(attempts, "O11", "R11"))
    frozen = ROOT / "experiments/pressure_v2/frozen_protocol.json"
    coverage = json.loads(frozen.read_text()).get("scenario_coverage") if frozen.exists() else {}
    (OUT / "scenario_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    plot_ps(attempts, OUT / "fig_ps.png")
    plot_time(attempts, OUT / "fig_online_s.png")
    write_results(coverage, attempts, means)
    print(json.dumps({"n_attempts": len(attempts), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
