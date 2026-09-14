"""Download CORe50 32×32 (mini) on demand. Never keep 128×128 images in RAM."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare_core50_mini(root: Path) -> dict:
    from avalanche.benchmarks.datasets.core50.core50 import CORe50Dataset

    root.mkdir(parents=True, exist_ok=True)
    # Instantiating with mini=True replaces the 128×128 zip with core50_32x32.zip.
    ds = CORe50Dataset(root=root, train=True, download=True, mini=True, object_level=True)
    img_root = root / "core50_32x32"
    payload = {
        "dataset": "core50_mini_32x32",
        "source": "Avalanche CORe50Dataset mini=True; zip from vps.continualai.org",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "n_indexed_train_paths": len(ds),
        "image_root": str(img_root),
        "image_root_exists": img_root.exists(),
        "note": (
            "Paper Table 2 input is 32×32 and lists 164866 images (train+test). "
            "This path downloads official mini 32×32 instead of 128×128. "
            "Images are file-backed; do not np.load a full imgs.npz into RAM. "
            "NI/NC/NIC splits use Avalanche filelists."
        ),
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/core50")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    payload = prepare_core50_mini(root)
    out = ROOT / "data" / "manifests" / "core50_mini.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
