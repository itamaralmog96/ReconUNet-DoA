"""``reconunet-generate`` — materialise train/val/test scene manifests.

This is the deterministic data-pipeline entrypoint.  It reads one YAML
config that defines (manifest paths, sizes, meta, sampling preset) and
writes three ``.npy`` manifests (+ sibling ``.yaml`` metadata files) to
disk.  Each manifest is ~96 bytes × N, so even the full 2 M-sample paper
corpus stays under 200 MB.

Usage::

    reconunet-generate --config configs/data/shared_manifest.yaml
    reconunet-generate --config configs/data/tiny.yaml --overwrite

The RNG seed in the config file is split into three *disjoint* child
seeds for train/val/test so the splits share no scenes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from reconunet.data.scene_manifest import (
    ArrayType,
    ManifestMeta,
    ModulationType,
    SceneManifest,
)

LOG = logging.getLogger("reconunet.generate")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a YAML mapping, got {type(cfg)}")
    for key in ("manifest", "meta", "sampling"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required top-level section '{key}'")
    return cfg


def _meta_from_cfg(meta: dict) -> ManifestMeta:
    array_type_str = str(meta.get("array_type", "ULA")).upper()
    array_type = ArrayType[array_type_str]
    return ManifestMeta(
        M=int(meta["M"]),
        T=int(meta["T"]),
        fs_Hz=float(meta["fs_Hz"]),
        tau=int(meta["tau"]),
        K_max=int(meta.get("K_max", 8)),
        angle_range_deg=tuple(meta["angle_range_deg"]),
        snr_range_db=tuple(meta["snr_range_db"]),
        element_spacing_lambda=float(meta.get("element_spacing_lambda", 0.5)),
        speed_of_light=float(meta.get("speed_of_light", 2.998e8)),
        array_type=array_type,
        version=str(meta.get("version", "1.0")),
        notes=str(meta.get("notes", "")),
    )


# ---------------------------------------------------------------------------
# Split-seed derivation
# ---------------------------------------------------------------------------


def _split_seeds(master_seed: int) -> dict[str, int]:
    """Deterministically derive non-overlapping child seeds.

    Uses NumPy's SeedSequence for cryptographic-grade de-correlation, so the
    same ``master_seed`` always yields the same triplet regardless of
    Python/NumPy version.
    """
    ss = np.random.SeedSequence(master_seed)
    train_ss, val_ss, test_ss = ss.spawn(3)
    return {
        "train": int(train_ss.generate_state(1)[0]),
        "val":   int(val_ss.generate_state(1)[0]),
        "test":  int(test_ss.generate_state(1)[0]),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _resolve_path(p: str, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def generate(cfg: dict, *, root: Path, overwrite: bool = False) -> dict[str, Path]:
    """Emit train/val/test manifests on disk.

    Returns
    -------
    dict[str, Path]
        Mapping ``{"train", "val", "test"} → absolute .npy path``.
    """
    meta_base = _meta_from_cfg(cfg["meta"])
    sampling = cfg.get("sampling", {})
    seeds = _split_seeds(int(sampling.get("rng_seed", 0)))

    array_errors = str(sampling.get("array_errors", "mild"))
    k_choices = tuple(int(k) for k in sampling.get("k_choices", [3]))
    min_sep = float(sampling.get("min_separation_deg", 3.0))

    # Multipath knobs (default off — legacy parity when enabled).  Accepted
    # keys under the YAML ``sampling`` section:
    #   enable_multipath (bool, default False)
    #   max_paths (int, default 3)                — legacy default
    #   num_multipath_components (int or None)     — None = random draw
    #   multipath_distribution ("uniform"|"exponential")
    #   mp_max_delay_factor (float, default 10.0)  — legacy default
    enable_multipath = bool(sampling.get("enable_multipath", False))
    max_paths = int(sampling.get("max_paths", 3))
    raw_nmp = sampling.get("num_multipath_components", None)
    num_mp_components = None if raw_nmp in (None, "none", "None") else int(raw_nmp)
    mp_distribution = str(sampling.get("multipath_distribution", "uniform"))
    mp_max_delay_factor = float(sampling.get("mp_max_delay_factor", 10.0))

    # Allow the config to switch modulation per split if desired, but default
    # to narrowband (Gaussian) which matches the paper's primary scenario.
    modulation = ModulationType[
        str(sampling.get("modulation", "NARROWBAND")).upper()
    ]

    out_paths: dict[str, Path] = {}
    man_cfg = cfg["manifest"]
    for split in ("train", "val", "test"):
        n = int(man_cfg[f"{split}_size"])
        out_path = _resolve_path(man_cfg[f"{split}_path"], root).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"{out_path} already exists — pass --overwrite to regenerate "
                "(this invalidates every checkpoint trained on the old scenes)."
            )

        LOG.info("Generating %s split: n=%d → %s  (seed=%d)",
                 split, n, out_path, seeds[split])
        manifest = SceneManifest.random(
            meta=replace(meta_base, notes=f"{meta_base.notes} [split={split}]"),
            size=n,
            rng=np.random.default_rng(seeds[split]),
            k_choices=k_choices,
            min_separation_deg=min_sep,
            array_errors=array_errors,
            progress_desc=f"{split:>5} scenes",
            enable_multipath=enable_multipath,
            max_paths=max_paths,
            num_multipath_components=num_mp_components,
            multipath_distribution=mp_distribution,
            mp_max_delay_factor=mp_max_delay_factor,
        )
        manifest.save(str(out_path))
        LOG.info("  wrote %s  (%.1f MB + sidecar)",
                 out_path.name, out_path.stat().st_size / 1024 / 1024)
        out_paths[split] = out_path

    return out_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reconunet-generate",
        description="Materialise train/val/test SceneManifests from a YAML config.",
    )
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="Path to shared_manifest YAML (see configs/data/).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing manifest files.  WARNING: "
                             "invalidates every checkpoint trained on them.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    cfg = _load_config(args.config)
    # Anchor relative manifest paths to the config's parent directory's
    # grandparent — i.e. the project root where `configs/` lives — so the
    # same config works from any cwd.
    root = args.config.resolve().parents[2]
    out = generate(cfg, root=root, overwrite=args.overwrite)
    LOG.info("Done.  Manifests: %s",
             {k: str(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
