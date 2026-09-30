"""Collect H6 summaries and controller action traces."""
import csv
import json
from pathlib import Path

REPO = Path("/home/zhuzetong/research/cl/Reproduce-Orion")
OUT = REPO / "reports" / "fullmem_v4" / "g2" / "h6_collect.json"
runs = sorted((REPO / "runs").glob("fullmem_v4_g2_*_h6*_seed17_mem16g"))
rows = []
for p in runs:
    s = json.loads((p / "summary.json").read_text()) if (p / "summary.json").is_file() else {}
    metrics_path = p / "experience_metrics.csv"
    ctrl_path = None
    for name in ("control.jsonl", "controller.jsonl", "controls.jsonl"):
        cand = p / name
        if cand.is_file():
            ctrl_path = cand
            break
    if ctrl_path is None:
        hits = list(p.glob("*control*.jsonl"))
        ctrl_path = hits[0] if hits else None
    events_path = p / "events.jsonl"
    traj = []
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                traj.append(
                    {
                        "exp": row.get("experience_index"),
                        "p": row.get("p_diag") or row.get("plasticity"),
                        "s": row.get("s_initial") or row.get("stability"),
                        "learning_s": row.get("learning_s"),
                        "opt": row.get("optimizer_mode"),
                        "new_batch": row.get("new_batch") or row.get("applied_new_batch"),
                        "replay": row.get("replay_max_size") or row.get("replay_capacity"),
                        "urge": row.get("urge"),
                        "thr": row.get("threshold") or row.get("thr"),
                        "fp": row.get("factor_p"),
                        "fs": row.get("factor_s"),
                        "fl": row.get("factor_l"),
                        "fm": row.get("factor_m"),
                    }
                )
    ctrl = []
    if ctrl_path is not None and ctrl_path.is_file():
        for line in ctrl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            factors = rec.get("factors") or {}
            ctrl.append(
                {
                    "t": rec.get("t"),
                    "urge": rec.get("urge"),
                    "thr": rec.get("thr"),
                    "suggested_new_batch": rec.get("suggested_new_batch"),
                    "suggested_replay": rec.get("suggested_replay"),
                    "applied_new_batch": (rec.get("applied") or {}).get("new_batch") if isinstance(rec.get("applied"), dict) else rec.get("applied_new_batch"),
                    "applied_replay": (rec.get("applied") or {}).get("replay_capacity") if isinstance(rec.get("applied"), dict) else rec.get("applied_replay"),
                    "mode": rec.get("suggested_optimizer_mode"),
                    "applied_mode": (rec.get("applied") or {}).get("optimizer_mode") if isinstance(rec.get("applied"), dict) else rec.get("applied_optimizer_mode"),
                    "fp": factors.get("factor_p"),
                    "fs": factors.get("factor_s"),
                    "fl": factors.get("factor_l"),
                    "fm": factors.get("factor_m"),
                    "memory_mib": rec.get("memory_mib"),
                    "reason": rec.get("reason"),
                }
            )
    batches = sorted({int(x["new_batch"]) for x in traj if x.get("new_batch") not in (None, "")})
    replays = sorted({int(float(x["replay"])) for x in traj if x.get("replay") not in (None, "")})
    modes = [x.get("opt") for x in traj]
    sug_b = sorted({int(x["suggested_new_batch"]) for x in ctrl if x.get("suggested_new_batch") not in (None, "")})
    sug_r = sorted({int(x["suggested_replay"]) for x in ctrl if x.get("suggested_replay") not in (None, "")})
    fms = [x.get("fm") for x in ctrl if x.get("fm") is not None]
    rows.append(
        {
            "run_id": p.name,
            "status": s.get("status"),
            "p_diag": s.get("p_diag"),
            "s_initial": s.get("s_initial"),
            "run_total_s": s.get("run_total_s"),
            "n_traj": len(traj),
            "n_ctrl": len(ctrl),
            "unique_new_batch": batches,
            "unique_replay": replays,
            "suggested_new_batch": sug_b,
            "suggested_replay": sug_r,
            "modes": modes,
            "fm_min": min(fms) if fms else None,
            "fm_max": max(fms) if fms else None,
            "ctrl_head": ctrl[:2],
            "ctrl_tail": ctrl[-2:],
        }
    )

occ = {}
op = REPO / "reports" / "fullmem_v4" / "g2" / "occupancy.json"
if op.is_file():
    raw = json.loads(op.read_text())
    occ = {
        "pressure_scene": raw.get("pressure_scene"),
        "min_available_bytes": raw.get("min_available_bytes"),
        "n_members": len(raw.get("members") or []),
        "note": raw.get("note"),
    }

payload = {
    "n_runs": len(rows),
    "occupancy": occ,
    "runs": rows,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"n_runs": len(rows), "occupancy": occ, "run_ids": [r["run_id"] for r in rows]}, indent=2))
for r in rows:
    print(
        f"{r['run_id']:72} {r['status']:10} P={r['p_diag']} S={r['s_initial']} wall={r['run_total_s']} "
        f"batch={r['unique_new_batch']} replay={r['unique_replay']} sugB={r['suggested_new_batch']} sugR={r['suggested_replay']} fm={r['fm_min']}-{r['fm_max']} modes={r['modes']}"
    )
