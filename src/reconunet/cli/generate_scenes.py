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
    mode = str(cfg.get("mode", "random")).lower()
    if mode == "fixed_angles_snr_sweep":
        required = ("manifest", "meta", "scenario")
    else:
        required = ("manifest", "meta", "sampling")
    for key in required:
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
        # Legacy-default 0.05·fs band-limited sources (coherent multipath);
        # set ``source_bw_frac: 0`` in the yaml for white full-band sources.
        source_bw_frac=(
            None if meta.get("source_bw_frac", 0.05) in (None, 0, 0.0)
            else float(meta["source_bw_frac"])
        ),
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


def _generate_fixed_angles_snr_sweep(
    cfg: dict, *, root: Path, overwrite: bool,
) -> dict[str, Path]:
    """Paper §IV "consistent angles across SNR" mode.

    Produces a single test manifest with ``n_angle_configs * len(snr_levels_db)``
    scenes laid out as ``n_angle_configs`` consecutive blocks of
    ``len(snr_levels_db)`` rows.  Within one block, every parameter except
    ``snr_db`` is identical so noise power is the only thing that changes
    across the SNR sweep.

    YAML schema (top-level keys; the ``manifest`` block specifies one
    ``test_path`` and a single ``test_size`` ignored — the size is derived
    from ``n_angle_configs * len(snr_levels_db)``):

        mode: fixed_angles_snr_sweep
        manifest:
          test_path: data/scenes/scenarios/<scenario>/test.npy
        scenario:
          n_angle_configs: 1000
          snr_levels_db: [-20, -15, -10, -5, 0, 5, 10, 15, 20]
          k: 4
          num_multipath_components: 3
          min_separation_deg: 10.0
          enable_multipath: true
          array_errors: mild
          rng_seed: 20260422
        meta:
          ...                   # same as the random-mode meta block
    """
    meta_base = _meta_from_cfg(cfg["meta"])
    scen = cfg.get("scenario", {})
    if not scen:
        raise ValueError(
            "scenario_sweep mode requires a top-level 'scenario' block — "
            "see configs/data/scenarios/*.yaml for an example."
        )

    n_angle_configs = int(scen["n_angle_configs"])
    snr_levels_db = [float(x) for x in scen["snr_levels_db"]]
    k = int(scen.get("k", 1))
    n_mp = int(scen.get("num_multipath_components", 0))
    enable_mp = bool(scen.get("enable_multipath", n_mp > 0))
    min_sep = float(scen.get("min_separation_deg", 10.0))
    array_errors = str(scen.get("array_errors", "mild"))
    mp_dist = str(scen.get("multipath_distribution", "uniform"))
    mp_delay = float(scen.get("mp_max_delay_factor", 10.0))
    rng_seed = int(scen.get("rng_seed", 0))

    test_path = _resolve_path(cfg["manifest"]["test_path"], root).resolve()
    test_path.parent.mkdir(parents=True, exist_ok=True)
    if test_path.exists() and not overwrite:
        raise FileExistsError(
            f"{test_path} already exists — pass --overwrite to regenerate."
        )

    LOG.info(
        "Generating fixed-angles SNR sweep: n_angle_configs=%d  S=%d  "
        "K=%d  M=%d  multipath=%s  → %s  (seed=%d)",
        n_angle_configs, len(snr_levels_db), k, n_mp,
        "on" if enable_mp else "off", test_path, rng_seed,
    )
    manifest = SceneManifest.fixed_angles_snr_sweep(
        meta=replace(meta_base, notes=f"{meta_base.notes} [scenario fixed-angles SNR sweep]"),
        n_angle_configs=n_angle_configs,
        snr_levels_db=snr_levels_db,
        rng=np.random.default_rng(rng_seed),
        k=k,
        min_separation_deg=min_sep,
        array_errors=array_errors,
        progress_desc="scenarios",
        enable_multipath=enable_mp,
        num_multipath_components=n_mp,
        multipath_distribution=mp_dist,
        mp_max_delay_factor=mp_delay,
    )
    manifest.save(str(test_path))
    LOG.info("  wrote %s  (%.1f MB + sidecar)  size=%d",
             test_path.name, test_path.stat().st_size / 1024 / 1024,
             len(manifest))
    return {"test": test_path}


def generate(cfg: dict, *, root: Path, overwrite: bool = False) -> dict[str, Path]:
    """Emit train/val/test manifests on disk.

    Returns
    -------
    dict[str, Path]
        Mapping ``{"train", "val", "test"} → absolute .npy path``.
    """
    if str(cfg.get("mode", "random")).lower() == "fixed_angles_snr_sweep":
        return _generate_fixed_angles_snr_sweep(cfg, root=root, overwrite=overwrite)

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
    # Anchor relative manifest paths to the project root, defined as the
    # nearest ancestor of the config that contains ``pyproject.toml``.  This
    # is robust to where exactly the YAML lives under ``configs/`` (depth-3
    # for legacy ``configs/data/*.yaml`` and depth-4 for the new
    # ``configs/data/scenarios/*.yaml`` both work).
    root = args.config.resolve().parent
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    if not (root / "pyproject.toml").exists():
        # Fall back to the legacy depth-3 assumption so behaviour is
        # unchanged for users running outside an installed project tree.
        root = args.config.resolve().parents[2]
    out = generate(cfg, root=root, overwrite=args.overwrite)
    LOG.info("Done.  Manifests: %s",
             {k: str(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
