"""Generate G2 development yaml/task files from the G1 CIFAR100 template."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "configs" / "fullmem_v4" / "g1_cifar100_exp1.yaml").read_text(encoding="utf-8")

CIFAR100_ORDER = """  fixed_class_order:
  - 37
  - 69
  - 71
  - 44
  - 19
  - 53
  - 90
  - 91
  - 93
  - 95
  - 40
  - 42
  - 74
  - 76
  - 80
  - 81
  - 82
  - 85
  - 56
  - 63
  - 97
  - 66
  - 8
  - 41
  - 10
  - 16
  - 89
  - 26
  - 92
  - 57
  - 65
  - 98
  - 35
  - 7
  - 20
  - 23
  - 24
  - 29
  - 30
  - 31
  - 4
  - 5
  - 46
  - 78
  - 15
  - 52
  - 54
  - 59
  - 61
  - 94
  - 2
  - 3
  - 68
  - 70
  - 6
  - 72
  - 28
  - 49
  - 55
  - 60
  - 32
  - 1
  - 34
  - 0
  - 11
  - 12
  - 45
  - 77
  - 79
  - 22
  - 96
  - 36
  - 9
  - 47
  - 50
  - 83
  - 84
  - 87
  - 25
  - 62
  - 64
  - 67
  - 39
  - 75
  - 13
  - 48
  - 17
  - 18
  - 21
  - 88
  - 33
  - 99
  - 38
  - 73
  - 43
  - 14
  - 51
  - 86
  - 58
  - 27
"""

COMMON_TAIL = """replay:
  capacity: {replay}
  policy: {replay_policy}
  representation: {replay_repr}
controller:
  enabled: {controller_enabled}
  units:
    accuracy: ratio
    latency: seconds
    memory: MiB
  metric_definition: diag_initial
  feedback_source: development_val_seen
  coefficients:
    kp: 0.25
    ks: 0.25
    kl: 0.25
    km: 0.25
  thresholds:
    p: 0.5
    s: 0.5
    latency_s: {latency_th_s}
    m_max_mib: 64293.0
  thr0: 0.05
  delta: 0.0
  updates:
    alpha: 0.1
    beta: 0.2
  mb0: {mb0}
  mr0: {mr0}
  m_batch: {m_batch}
  m_frame: {m_frame}
  min_batch: 1
  max_batch: 1024
  equal_uses_gt: true
  plugin_policy: {plugin_policy}
budget:
  enforcement: observed_only
  controlled_resource: {controlled_resource}
  limit_bytes: null
  guard_mode: stop
  cost_model_path: null
prefetch:
  enabled: {prefetch_enabled}
  queue_depth: {queue_depth}
  pin_memory: false
  num_workers: 0
measurement:
  sample_interval_ms: 100
  checkpoint_policy: none
study_id: fullmem_v4
reuse_version: 4
pressure_role: g2_dev_calibration_mem64g
v3_role: not_v3
v3_dataset: {v3_dataset}
board_capacity_id: mem64g
notes: {notes}
"""


def header(
    run_id: str,
    experiment_id: str,
    method_id: str,
    num_classes: int,
    plugins: str,
    new_batch: int,
    replay_batch: int,
    *,
    start_enabled: bool = False,
    algo_base: str = "er",
    extra_algo: str | None = None,
    eval_batch: int = 32,
    seed: int = 0,
) -> str:
    plugin_policy = "fixed_advanced" if start_enabled else "fixed_default"
    start_line = f"  optional_start_enabled: {str(start_enabled).lower()}"
    if extra_algo is None:
        extra_algo = ""
        if plugins in {"gem", "gem_ewc"}:
            extra_algo = "  patterns_per_exp: 20\n  memory_strength: 0.5\n"
        if plugins in {"ewc", "gem_ewc"}:
            extra_algo += "  ewc_lambda: 100.0\n"
    elif extra_algo and not extra_algo.endswith("\n"):
        extra_algo = extra_algo + "\n"
    return f"""schema_version: 1
run_id: {run_id}
phase: development
claim_ids:
- {experiment_id}
experiment_id: {experiment_id}
protocol_id: fullmem_v4_g2_dev_calibration
method_id: {method_id}
dataset_manifest_sha256: null
code_revision_or_snapshot: null
environment_lock_sha256: null
alignment_version: fullmem_v4
seeds:
  model: {seed}
  stream: {seed}
  replay: {seed}
  augmentation: {seed}
model:
  name: cifar_resnet20
  initialization: random
  precision: fp32
  num_classes: {num_classes}
algorithm:
  base: {algo_base}
  optional_plugins: {plugins}
{start_line}
{extra_algo}training:
  new_epochs: 1
  new_batch: {new_batch}
  replay_batch: {replay_batch}
  eval_batch: {eval_batch}
  device: cuda:0
  allow_tf32: false
  cudnn_benchmark: false
  optimizer:
    name: sgd
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0
    scheduler: none
"""


def cifar100_dataset() -> str:
    return """dataset:
  name: splitcifar100
  root: data/raw/cifar100
  split_manifest: data/manifests/cifar100_dev_split_seed17.json
  n_experiences: 10
  experience_limit: 10
  shuffle: false
""" + CIFAR100_ORDER


def cifar10_dataset() -> str:
    return """dataset:
  name: splitcifar10
  root: data/raw/cifar10
  split_manifest: data/manifests/cifar10_dev_split_seed17.json
  n_experiences: 10
  experience_limit: 10
  shuffle: false
  fixed_class_order:
  - 4
  - 1
  - 7
  - 5
  - 3
  - 9
  - 0
  - 8
  - 6
  - 2
"""


def core50_dataset() -> str:
    return """dataset:
  name: core50_nc
  root: data/raw/core50
  split_dir: data/processed/core50_nc_run0_dev_split_seed17
  split_manifest: data/manifests/core50_nc_run0_dev_split_seed17.json
  n_experiences: 9
  experience_limit: 9
  shuffle: false
  scenario: nc
  run: 0
  object_level: true
"""


def core50_ni_dataset() -> str:
    return """dataset:
  name: core50_ni
  root: data/raw/core50
  split_dir: reports/fullmem_v4/data/core50_ni_run0_dev_split_seed17
  split_manifest: data/manifests/core50_ni_run0_dev_split_seed17.json
  n_experiences: 8
  experience_limit: 8
  shuffle: false
  scenario: ni
  run: 0
  object_level: true
"""


def core50_nic_dataset() -> str:
    return """dataset:
  name: core50_nic
  root: data/raw/core50
  split_dir: reports/fullmem_v4/data/core50_nicv2_79_run0_dev_split_seed17
  split_manifest: data/manifests/core50_nicv2_79_run0_dev_split_seed17.json
  n_experiences: 79
  experience_limit: 79
  shuffle: false
  scenario: nicv2_79
  run: 0
  object_level: true
"""


def endless_dataset(name: str, n_exp: int, scenario: str) -> str:
    return f"""dataset:
  name: {name}
  root: data/raw/endless_cl_sim
  n_experiences: {n_exp}
  experience_limit: {n_exp}
  shuffle: false
  scenario: {scenario}
  patch_size: 32
  split_policy: class_ordered_holdout_embargo_v2
  split_seed: 17
  val_fraction: 0.2
  embargo_patches: 32
"""


CONFIGS = [
    {
        "stem": "g2_cifar100_s0",
        "run_id": "fullmem_v4_g2_cifar100_s0",
        "experiment_id": "V4_g2_dev_cifar100_s0",
        "method_id": "S0",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 full-stream S0 development calibration on mem64g; observed_only; not a freeze.",
        "task_id": "g2-dev-cifar100-s0",
        "seq": 24,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_s0.yaml"],
    },
    {
        "stem": "g2_cifar100_batch256",
        "run_id": "fullmem_v4_g2_cifar100_batch256",
        "experiment_id": "V4_g2_dev_cifar100_batch256",
        "method_id": "static_high_batch",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 256,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 high-resource batch candidate for pressure identifiability, not Orion success.",
        "task_id": "g2-dev-cifar100-batch256",
        "seq": 25,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_batch256.yaml"],
    },
    {
        "stem": "g2_cifar100_replay2000",
        "run_id": "fullmem_v4_g2_cifar100_replay2000",
        "experiment_id": "V4_g2_dev_cifar100_replay2000",
        "method_id": "static_high_replay",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 2000,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 high-resource replay candidate for pressure identifiability.",
        "task_id": "g2-dev-cifar100-replay2000",
        "seq": 26,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_replay2000.yaml"],
    },
    {
        "stem": "g2_cifar100_er_gem",
        "run_id": "fullmem_v4_g2_cifar100_er_gem",
        "experiment_id": "V4_g2_dev_cifar100_er_gem",
        "method_id": "ER+GEM",
        "num_classes": 100,
        "plugins": "gem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 plugin-on development calibration; pause-keep-state default.",
        "task_id": "g2-dev-cifar100-er-gem",
        "seq": 27,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_er_gem.yaml"],
    },
    {
        "stem": "g2_cifar100_er_ewc",
        "run_id": "fullmem_v4_g2_cifar100_er_ewc",
        "experiment_id": "V4_g2_dev_cifar100_er_ewc",
        "method_id": "ER+EWC",
        "num_classes": 100,
        "plugins": "ewc",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 plugin-on development calibration; pause-keep-state default.",
        "task_id": "g2-dev-cifar100-er-ewc",
        "seq": 28,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_er_ewc.yaml"],
    },
    {
        "stem": "g2_cifar100_er_gem_ewc",
        "run_id": "fullmem_v4_g2_cifar100_er_gem_ewc",
        "experiment_id": "V4_g2_dev_cifar100_er_gem_ewc",
        "method_id": "ER+GEM+EWC",
        "num_classes": 100,
        "plugins": "gem_ewc",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 stacked optional plugins development calibration.",
        "task_id": "g2-dev-cifar100-er-gem-ewc",
        "seq": 29,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_er_gem_ewc.yaml"],
    },
    {
        "stem": "g2_core50_nc_s0",
        "run_id": "fullmem_v4_g2_core50_nc_s0",
        "experiment_id": "V4_g2_dev_core50_nc_s0",
        "method_id": "S0",
        "num_classes": 50,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 CORe50-NC full 9-experience S0 development calibration.",
        "task_id": "g2-dev-core50-nc-s0",
        "seq": 30,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nc_s0.yaml"],
    },
    {
        "stem": "g2_cifar10_s0",
        "run_id": "fullmem_v4_g2_cifar10_s0",
        "experiment_id": "V4_g2_dev_cifar10_s0",
        "method_id": "S0",
        "num_classes": 10,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar10_dataset,
        "v3_dataset": "splitcifar10",
        "notes": "G2 SplitCIFAR10 10-experience S0; class order frozen 4 1 7 5 3 9 0 8 6 2.",
        "task_id": "g2-dev-cifar10-s0",
        "seq": 23,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar10_s0.yaml"],
    },
    {
        "stem": "g2_cifar100_prefetch",
        "run_id": "fullmem_v4_g2_cifar100_prefetch",
        "experiment_id": "V4_g2_dev_cifar100_prefetch",
        "method_id": "S0_prefetch_on",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 H7 isolate: same S0 with prefetch on. Not a freeze.",
        "task_id": "g2-dev-cifar100-prefetch",
        "seq": 41,
        "kind": "probe",
        "prefetch_enabled": True,
        "queue_depth": 2,
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_prefetch.yaml"],
    },
    {
        "stem": "g2_core50_ni_s0",
        "run_id": "fullmem_v4_g2_core50_ni_s0",
        "experiment_id": "V4_g2_dev_core50_ni_s0",
        "method_id": "S0",
        "num_classes": 50,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_ni_dataset,
        "v3_dataset": "core50_ni",
        "notes": "G2 CORe50-NI 8-experience S0; processed split written under data/processed if missing.",
        "task_id": "g2-dev-core50-ni-s0",
        "seq": 42,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_ni_s0.yaml"],
    },
    {
        "stem": "g2_core50_nic_s0",
        "run_id": "fullmem_v4_g2_core50_nic_s0",
        "experiment_id": "V4_g2_dev_core50_nic_s0",
        "method_id": "S0",
        "num_classes": 50,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_nic_dataset,
        "v3_dataset": "core50_nic",
        "notes": "G2 CORe50-NIC Avalanche nicv2_79 reconstruction (A14), 79 experiences.",
        "task_id": "g2-dev-core50-nic-s0",
        "seq": 43,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nic_s0.yaml"],
    },
    {
        "stem": "g2_endless_ic_s0",
        "run_id": "fullmem_v4_g2_endless_ic_s0",
        "experiment_id": "V4_g2_dev_endless_ic_s0",
        "method_id": "S0",
        "num_classes": 5,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": lambda: endless_dataset("endless_ic", 4, "Classes"),
        "v3_dataset": "endless_ic",
        "notes": "G2 Endless IC 4-exp S0; A22 grouped holdout is development split, not official robot protocol.",
        "task_id": "g2-dev-endless-ic-s0",
        "seq": 44,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_endless_ic_s0.yaml"],
    },
    {
        "stem": "g2_endless_il_s0",
        "run_id": "fullmem_v4_g2_endless_il_s0",
        "experiment_id": "V4_g2_dev_endless_il_s0",
        "method_id": "S0",
        "num_classes": 5,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": lambda: endless_dataset("endless_il", 5, "Illumination"),
        "v3_dataset": "endless_il",
        "notes": "G2 Endless IL 5-exp S0; A22 grouped holdout is development split, not official robot protocol.",
        "task_id": "g2-dev-endless-il-s0",
        "seq": 45,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_endless_il_s0.yaml"],
    },
    {
        "stem": "g2_endless_wc_s0",
        "run_id": "fullmem_v4_g2_endless_wc_s0",
        "experiment_id": "V4_g2_dev_endless_wc_s0",
        "method_id": "S0",
        "num_classes": 5,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": lambda: endless_dataset("endless_wc", 5, "Weather"),
        "v3_dataset": "endless_wc",
        "notes": "G2 Endless WC 5-exp S0; A22 grouped holdout is development split, not official robot protocol.",
        "task_id": "g2-dev-endless-wc-s0",
        "seq": 46,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_endless_wc_s0.yaml"],
    },
    {
        "stem": "g2_cifar100_sstar_b64_r200",
        "run_id": "fullmem_v4_g2_cifar100_sstar_b64_r200",
        "experiment_id": "V4_g2_dev_cifar100_sstar",
        "method_id": "Sstar_candidate",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 64,
        "replay_batch": 64,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 S* remaining cell batch 64 replay 200. Selection not frozen.",
        "task_id": "g2-dev-cifar100-sstar-b64-r200",
        "seq": 62,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_sstar_b64_r200.yaml"],
    },
    {
        "stem": "g2_cifar100_sstar_b64_r2000",
        "run_id": "fullmem_v4_g2_cifar100_sstar_b64_r2000",
        "experiment_id": "V4_g2_dev_cifar100_sstar",
        "method_id": "Sstar_candidate",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 64,
        "replay_batch": 64,
        "replay": 2000,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 S* remaining cell batch 64 replay 2000. Selection not frozen.",
        "task_id": "g2-dev-cifar100-sstar-b64-r2000",
        "seq": 63,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_sstar_b64_r2000.yaml"],
    },
    {
        "stem": "g2_cifar100_sstar_b256_r2000",
        "run_id": "fullmem_v4_g2_cifar100_sstar_b256_r2000",
        "experiment_id": "V4_g2_dev_cifar100_sstar",
        "method_id": "Sstar_candidate",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 256,
        "replay_batch": 256,
        "replay": 2000,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 S* remaining cell batch 256 replay 2000. Selection not frozen.",
        "task_id": "g2-dev-cifar100-sstar-b256-r2000",
        "seq": 64,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_sstar_b256_r2000.yaml"],
    },
    {
        "stem": "g2_cifar100_s0_eval128",
        "run_id": "fullmem_v4_g2_cifar100_s0_eval128",
        "experiment_id": "V4_g2_dev_cifar100_eval128",
        "method_id": "S0",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "eval_batch": 128,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 common eval-batch probe: S0 with eval_batch 128. Not a freeze.",
        "task_id": "g2-dev-cifar100-s0-eval128",
        "seq": 65,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_s0_eval128.yaml"],
    },
    {
        "stem": "g2_cifar100_orecon",
        "run_id": "fullmem_v4_g2_cifar100_orecon",
        "experiment_id": "V4_g2_dev_cifar100_orecon",
        "method_id": "O-recon",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "controller_enabled": True,
        "latency_th_s": 5.513,
        "prefetch_enabled": True,
        "queue_depth": 2,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 O-recon ER-only; L_cal=CIFAR100 S0 median learning_s 5.513s; prefetch on as H1 base. Not a freeze.",
        "task_id": "g2-dev-cifar100-orecon",
        "seq": 120,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_orecon.yaml"],
    },
    {
        "stem": "g2_cifar100_gss",
        "run_id": "fullmem_v4_g2_cifar100_gss",
        "experiment_id": "V4_g2_dev_cifar100_gss",
        "method_id": "GSS",
        "num_classes": 100,
        "plugins": "none",
        "algo_base": "gss",
        "extra_algo": "  mem_strength: 1\n  input_size:\n  - 3\n  - 32\n  - 32\n",
        "replay_policy": "gss_greedy",
        "replay_repr": "avalanche_gss",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 GSS as base algorithm on CIFAR100. Not a freeze.",
        "task_id": "g2-dev-cifar100-gss",
        "seq": 121,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_gss.yaml"],
    },
    {
        "stem": "g2_cifar100_gem",
        "run_id": "fullmem_v4_g2_cifar100_gem",
        "experiment_id": "V4_g2_dev_cifar100_gem",
        "method_id": "GEM",
        "num_classes": 100,
        "plugins": "none",
        "algo_base": "gem",
        "extra_algo": "  patterns_per_exp: 20\n  memory_strength: 0.5\n",
        "replay_policy": "gem_patterns_per_exp",
        "replay_repr": "avalanche_gem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 GEM as base algorithm on CIFAR100. GEM is not an optional plugin here.",
        "task_id": "g2-dev-cifar100-gem",
        "seq": 122,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_gem.yaml"],
    },
    {
        "stem": "g2_cifar100_agem",
        "run_id": "fullmem_v4_g2_cifar100_agem",
        "experiment_id": "V4_g2_dev_cifar100_agem",
        "method_id": "AGEM",
        "num_classes": 100,
        "plugins": "none",
        "algo_base": "agem",
        "extra_algo": "  patterns_per_exp: 20\n  sample_size: 64\n",
        "replay_policy": "agem_patterns_per_exp",
        "replay_repr": "avalanche_agem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 AGEM as base algorithm on CIFAR100. Not a freeze.",
        "task_id": "g2-dev-cifar100-agem",
        "seq": 123,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_agem.yaml"],
    },
    {
        "stem": "g2_core50_nc_orecon",
        "run_id": "fullmem_v4_g2_core50_nc_orecon",
        "experiment_id": "V4_g2_dev_core50_nc_orecon",
        "method_id": "O-recon",
        "num_classes": 50,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "controller_enabled": True,
        "latency_th_s": 15.706,
        "prefetch_enabled": True,
        "queue_depth": 2,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 O-recon ER-only on CORe50-NC; L_cal=NC S0 median learning_s 15.706s. Not a freeze.",
        "task_id": "g2-dev-core50-nc-orecon",
        "seq": 124,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nc_orecon.yaml"],
    },
    {
        "stem": "g2_core50_nc_gss",
        "run_id": "fullmem_v4_g2_core50_nc_gss",
        "experiment_id": "V4_g2_dev_core50_nc_gss",
        "method_id": "GSS",
        "num_classes": 50,
        "plugins": "none",
        "algo_base": "gss",
        "extra_algo": "  mem_strength: 1\n  input_size:\n  - 3\n  - 32\n  - 32\n",
        "replay_policy": "gss_greedy",
        "replay_repr": "avalanche_gss",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 GSS as base algorithm on CORe50-NC (H2 stream). Not a freeze.",
        "task_id": "g2-dev-core50-nc-gss",
        "seq": 125,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nc_gss.yaml"],
    },
    {
        "stem": "g2_core50_nc_gem",
        "run_id": "fullmem_v4_g2_core50_nc_gem",
        "experiment_id": "V4_g2_dev_core50_nc_gem",
        "method_id": "GEM",
        "num_classes": 50,
        "plugins": "none",
        "algo_base": "gem",
        "extra_algo": "  patterns_per_exp: 50\n  memory_strength: 0.5\n",
        "replay_policy": "gem_patterns_per_exp",
        "replay_repr": "avalanche_gem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 GEM as base algorithm on CORe50-NC; patterns_per_exp 50 matches historical NC static.",
        "task_id": "g2-dev-core50-nc-gem",
        "seq": 126,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nc_gem.yaml"],
    },
    {
        "stem": "g2_core50_nc_agem",
        "run_id": "fullmem_v4_g2_core50_nc_agem",
        "experiment_id": "V4_g2_dev_core50_nc_agem",
        "method_id": "AGEM",
        "num_classes": 50,
        "plugins": "none",
        "algo_base": "agem",
        "extra_algo": "  patterns_per_exp: 50\n  sample_size: 64\n",
        "replay_policy": "agem_patterns_per_exp",
        "replay_repr": "avalanche_agem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 AGEM as base algorithm on CORe50-NC. Not a freeze.",
        "task_id": "g2-dev-core50-nc-agem",
        "seq": 127,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nc_agem.yaml"],
    },
    {
        "stem": "g2_cifar100_orecon_full",
        "run_id": "fullmem_v4_g2_cifar100_orecon_full",
        "experiment_id": "V4_g2_dev_cifar100_orecon_full",
        "method_id": "O-recon",
        "num_classes": 100,
        "plugins": "gem_ewc",
        "start_enabled": False,
        "plugin_policy": "adaptive",
        "controlled_resource": "board",
        "seeds": 17,
        "m_batch": 3072.0,
        "m_frame": 3072.0,
        "mb0": 49152.0,
        "mr0": 614400.0,
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "controller_enabled": True,
        "latency_th_s": 5.513,
        "prefetch_enabled": True,
        "queue_depth": 2,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 complete O-recon wiring: board MemTotal-MemAvailable feedback, byte-space m_batch/m_frame from 32x32 uint8, adaptive GEM+EWC, seed 17. Previous oreconf run is batch/replay-only. Not a freeze.",
        "task_id": "g2-dev-cifar100-orecon-full",
        "seq": 150,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_orecon_full.yaml"],
    },
    {
        "stem": "g2_core50_nc_orecon_full",
        "run_id": "fullmem_v4_g2_core50_nc_orecon_full",
        "experiment_id": "V4_g2_dev_core50_nc_orecon_full",
        "method_id": "O-recon",
        "num_classes": 50,
        "plugins": "gem_ewc",
        "start_enabled": False,
        "plugin_policy": "adaptive",
        "controlled_resource": "board",
        "seeds": 17,
        "m_batch": 3072.0,
        "m_frame": 3072.0,
        "mb0": 49152.0,
        "mr0": 614400.0,
        "extra_algo": "  patterns_per_exp: 50\n  memory_strength: 0.5\n  ewc_lambda: 100.0\n",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "controller_enabled": True,
        "latency_th_s": 15.706,
        "prefetch_enabled": True,
        "queue_depth": 2,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 complete O-recon wiring on CORe50-NC: board feedback, byte-space mapping, adaptive GEM+EWC, seed 17. Previous nc-orecon is batch/replay-only. Not a freeze.",
        "task_id": "g2-dev-core50-nc-orecon-full",
        "seq": 151,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_core50_nc_orecon_full.yaml"],
    },
]


ORACLE_BATCHES = [16, 32, 64, 128, 256, 512, 1024]
ORACLE_REPLAY = [10, 100, 1000, 10000, 100000, 1000000]


def oracle_specs() -> list[dict]:
    specs = []
    seq = 70
    for batch in ORACLE_BATCHES:
        for replay in ORACLE_REPLAY:
            cell = f"b{batch}_r{replay}"
            stem = f"g2_oracle_cifar100_{cell}"
            specs.append(
                {
                    "stem": stem,
                    "run_id": f"fullmem_v4_g2_oracle_cifar100_{cell}",
                    "experiment_id": "V4_g2_oracle_cifar100",
                    "method_id": "oracle_reconstructed",
                    "num_classes": 100,
                    "plugins": "none",
                    "new_batch": batch,
                    "replay_batch": batch,
                    "replay": replay,
                    "dataset": cifar100_dataset,
                    "v3_dataset": "splitcifar100",
                    "notes": f"G2 Oracle 42-cell search on CIFAR100 development stream, cell {cell}. Search cost not a selected-config train cost.",
                    "task_id": f"g2-oracle-cifar100-{cell.replace('_', '-')}",
                    "seq": seq,
                    "kind": "probe",
                    "argv_extra": ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
                }
            )
            seq += 1
    return specs


PROBE_TASKS = [
    {
        "task_id": "g2-inventory",
        "seq": 20,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-inventory"],
    },
    {
        "task_id": "g2-physical",
        "seq": 21,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-physical"],
    },
    {
        "task_id": "g2-prepare-cifar10",
        "seq": 22,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-prepare-cifar10"],
    },
    {
        "task_id": "g2-prepare-endless",
        "seq": 31,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-prepare-endless", "--scenario", "all"],
    },
    {
        "task_id": "g2-prepare-core50-splits",
        "seq": 40,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-prepare-core50-splits"],
    },
    {
        "task_id": "g2-dev-cifar100-prefetch-b",
        "seq": 41,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.run", "--config", "configs/fullmem_v4/g2_cifar100_prefetch.yaml"],
    },
    {
        "task_id": "g2-occupancy",
        "seq": 50,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
    },
    {
        "task_id": "g2-lcal",
        "seq": 60,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-lcal"],
    },
    {
        "task_id": "g2-idle-baseline",
        "seq": 61,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"],
    },
    {
        "task_id": "g2-occupancy-b",
        "seq": 140,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
    },
    {
        "task_id": "g2-cost",
        "seq": 141,
        "kind": "probe",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "g2-cost"],
    },
    {
        "task_id": "g1-impl-error-inject",
        "seq": 19,
        "kind": "dummy",
        "argv_extra": ["-m", "orion_repro.stages.fullmem_v4", "dummy-fail", "--reason", "g1_implementation_error_injection"],
        "batch_id": "g1-fault-inject",
    },
]


def write_task(task_id: str, seq: int, kind: str, argv_extra: list[str], batch_id: str) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": seq,
        "kind": kind,
        "argv": ["/home/zhuzetong/miniconda3/envs/orion/bin/python", *argv_extra],
        "required_capacity": "mem64g",
        "batch_id": batch_id,
    }
    dest = ROOT / "experiments" / "fullmem_v4" / "tasks" / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("task", dest)


def main() -> None:
    cfg_dir = ROOT / "configs" / "fullmem_v4"
    for spec in [*CONFIGS, *oracle_specs()]:
        start_enabled = spec.get("start_enabled", spec["plugins"] != "none")
        plugin_policy = spec.get(
            "plugin_policy",
            "fixed_advanced" if start_enabled else "fixed_default",
        )
        text = (
            header(
                spec["run_id"],
                spec["experiment_id"],
                spec["method_id"],
                spec["num_classes"],
                spec["plugins"],
                spec["new_batch"],
                spec["replay_batch"],
                start_enabled=start_enabled,
                algo_base=spec.get("algo_base", "er"),
                extra_algo=spec.get("extra_algo"),
                eval_batch=int(spec.get("eval_batch", 32)),
                seed=int(spec.get("seeds", 0)),
            )
            + spec["dataset"]()
            + COMMON_TAIL.format(
                replay=spec["replay"],
                v3_dataset=spec["v3_dataset"],
                notes=json.dumps(spec["notes"]),
                plugin_policy=plugin_policy,
                prefetch_enabled=str(bool(spec.get("prefetch_enabled", False))).lower(),
                queue_depth=int(spec.get("queue_depth", 1)),
                replay_policy=spec.get("replay_policy", "experience_balanced"),
                replay_repr=spec.get("replay_repr", "avalanche_buffer"),
                controller_enabled=str(bool(spec.get("controller_enabled", False))).lower(),
                latency_th_s=spec.get("latency_th_s", 30.0),
                mb0=spec.get("mb0", 16.0),
                mr0=spec.get("mr0", 200.0),
                m_batch=spec.get("m_batch", 1.0),
                m_frame=spec.get("m_frame", 1.0),
                controlled_resource=spec.get("controlled_resource", "device"),
            )
        )
        dest = cfg_dir / f"{spec['stem']}.yaml"
        dest.write_text(text, encoding="utf-8", newline="\n")
        print("yaml", dest)
        write_task(spec["task_id"], spec["seq"], spec["kind"], spec["argv_extra"], "g2-dev-mem64g")
    for spec in PROBE_TASKS:
        write_task(
            spec["task_id"],
            spec["seq"],
            spec["kind"],
            spec["argv_extra"],
            spec.get("batch_id", "g2-dev-mem64g"),
        )


if __name__ == "__main__":
    main()
