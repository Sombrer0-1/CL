"""Download EndlessCL-Sim classification zips (E10). Classification only, not video."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[2]

ZENODO = "https://zenodo.org/record/4899267/files/"
SCENARIOS = {
    "ic": (
        "IncrementalClasses_Classification.zip",
        "8f53c18e46de35ba375d6ed8fced5d47",
    ),
    "il": (
        "IncrementalLighting_Classification.zip",
        "61a36070d6aae926ef3d121fd17ec501",
    ),
    "wc": (
        "IncrementalWeather_Classification.zip",
        "60e1a0d50b0091e16424d8e88ae2a2a2",
    ),
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_scenario(name: str, dest: Path) -> dict:
    filename, expected = SCENARIOS[name]
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / filename
    url = ZENODO + filename
    if not zip_path.is_file():
        print(f"downloading {url}")
        urlretrieve(url, zip_path)
    digest = _md5(zip_path)
    if digest != expected:
        raise SystemExit(f"{filename} md5 {digest} != {expected}")
    return {
        "scenario": name,
        "filename": filename,
        "url": url,
        "path": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "md5": digest,
        "paper_input": "32x32 classification patches; Avalanche default patch is 64",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }


def extract_scenario(name: str, dest: Path) -> dict:
    """Expand classification zip with the stdlib; do not require system unzip."""
    filename, _expected = SCENARIOS[name]
    zip_path = dest / filename
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out_dir = dest / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    nested = sorted(out_dir.glob("*.zip"))
    for inner in nested:
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(out_dir)
        inner.unlink()
    train_dirs = [p.name for p in out_dir.iterdir() if p.is_dir() and "train" in p.name.lower()]
    test_dirs = [p.name for p in out_dir.iterdir() if p.is_dir() and "test" in p.name.lower()]
    return {
        "scenario": name,
        "extracted_to": str(out_dir),
        "train_dirs": train_dirs,
        "test_dirs": test_dirs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["ic", "il", "wc", "all"])
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Expand already-downloaded classification zips (Python zipfile, no unzip binary).",
    )
    args = parser.parse_args(argv)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    dest = ROOT / "data" / "raw" / "endless_cl_sim"
    manifests = ROOT / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if args.extract:
        rows = [extract_scenario(n, dest) for n in names]
        print(json.dumps({"extracted": rows}, indent=2))
        return
    rows = [download_scenario(n, dest) for n in names]
    out = manifests / "endless_cl_sim.json"
    payload = {
        "dataset": "endless_cl_sim_classification",
        "source": "zenodo.4899267 classification zips; not video",
        "patch_size_paper": 32,
        "scenarios": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
