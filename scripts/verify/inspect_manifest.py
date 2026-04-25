#!/usr/bin/env python3
"""``inspect_manifest.py`` — sanity-check a SceneManifest ``.npy`` file.

Reports:
  * row count, meta YAML, fingerprint
  * distribution of K (n_sources), SNR, angles, array_type, modulation
  * imperfection preset (by checking the four error columns)
  * seed uniqueness (a critical correctness invariant)
  * optional histogram PNG via ``--plot``

Usage
-----
    python scripts/verify/inspect_manifest.py data/scenes/verify/tiny/train.npy
    python scripts/verify/inspect_manifest.py data/scenes/train.npy --plot \\
        --out experiments/verify/train_hist.png
    python scripts/verify/inspect_manifest.py \\
        data/scenes/verify/tiny/train.npy \\
        data/scenes/verify/tiny/val.npy \\
        data/scenes/verify/tiny/test.npy \\
        --check-split-disjoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `reconunet` importable whether the user invokes this from the repo root
# or from scripts/verify/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from reconunet.data.scene_manifest import (  # noqa: E402
    ArrayType,
    ModulationType,
    SceneManifest,
)


def _preset_name(gain: float, phase: float, coupling: float, pos: float) -> str:
    """Best-effort label for one row's (gain, phase, coupling, position) tuple."""
    if np.allclose([gain, phase, coupling, pos], 0.0, atol=1e-6):
        return "none"
    if np.allclose([gain, phase, coupling, pos], [0.1, 1.0, 0.02, 1.0], atol=0.05):
        return "mild"
    if np.allclose([gain, phase, coupling, pos], [0.5, 5.0, 0.10, 5.0], atol=0.2):
        return "harsh"
    return "custom"


def _summarise(man: SceneManifest, label: str) -> dict:
    rows = man.raw
    n = len(man)
    K = rows["n_sources"].astype(int)
    snr = rows["snr_db"].astype(float)
    scene_ids = rows["scene_id"].astype(np.uint64)
    seeds = rows["seed"].astype(np.uint64)

    gain = rows["gain_err_dB"].astype(float)
    phase = rows["phase_err_deg"].astype(float)
    coupling = rows["mutual_coupling"].astype(float)
    pos = rows["position_err_pct"].astype(float)

    angles = rows["angles_deg"].astype(np.float64)  # [n, K_MAX]
    valid_mask = np.isfinite(angles)
    valid_angles = angles[valid_mask]

    array_types = np.unique(rows["array_type"])
    modulations = np.unique(rows["modulation"])

    print(f"\n=== {label}: {n} rows ===")
    print(f"  meta.M={man.meta.M}, T={man.meta.T}, tau={man.meta.tau}, "
          f"array={man.meta.array_type}, d/λ={man.meta.element_spacing_lambda}")
    print(f"  meta.angle_range_deg={man.meta.angle_range_deg}, "
          f"snr_range_db={man.meta.snr_range_db}")
    print(f"  meta.version={man.meta.version!r}, notes={man.meta.notes!r}")
    print(f"  fingerprint={man.fingerprint()[:16]}…")

    print("\n  K distribution:")
    for k, c in zip(*np.unique(K, return_counts=True)):
        print(f"    K={int(k):>2d}   n={int(c):>7d}   ({100*c/n:5.1f}%)")

    print("\n  SNR (dB):")
    print(f"    min={snr.min():6.2f}  max={snr.max():6.2f}  "
          f"mean={snr.mean():6.2f}  std={snr.std():5.2f}")

    print("\n  Angles (deg, valid only, broadside=0):")
    if valid_angles.size:
        print(f"    count={valid_angles.size}  min={valid_angles.min():7.2f}  "
              f"max={valid_angles.max():7.2f}  "
              f"mean={valid_angles.mean():7.3f}  std={valid_angles.std():6.2f}")
    else:
        print("    (no valid angles — this is a bug)")

    print("\n  Array types present (IntEnum values):")
    for a in array_types:
        try:
            name = ArrayType(int(a)).name
        except ValueError:
            name = f"?({a})"
        c = int((rows["array_type"] == a).sum())
        print(f"    {name:<10s} n={c:>7d}")

    print("\n  Modulations present:")
    for m in modulations:
        try:
            name = ModulationType(int(m)).name
        except ValueError:
            name = f"?({m})"
        c = int((rows["modulation"] == m).sum())
        print(f"    {name:<10s} n={c:>7d}")

    print("\n  Imperfection preset (inferred from row 0):")
    p0 = _preset_name(
        float(gain[0]), float(phase[0]), float(coupling[0]), float(pos[0])
    )
    print(f"    row[0] preset = {p0}")
    print(f"    gain_err_dB       min/max = {gain.min():.4f} / {gain.max():.4f}")
    print(f"    phase_err_deg     min/max = {phase.min():.4f} / {phase.max():.4f}")
    print(f"    mutual_coupling   min/max = {coupling.min():.4f} / {coupling.max():.4f}")
    print(f"    position_err_pct  min/max = {pos.min():.4f} / {pos.max():.4f}")

    n_unique_ids = np.unique(scene_ids).size
    n_unique_seeds = np.unique(seeds).size
    print("\n  Uniqueness:")
    print(f"    unique scene_ids : {n_unique_ids:>7d} / {n}  "
          f"{'OK' if n_unique_ids == n else '** DUPLICATES **'}")
    print(f"    unique seeds     : {n_unique_seeds:>7d} / {n}  "
          f"{'OK' if n_unique_seeds == n else '** DUPLICATES **'}")

    return {
        "label": label,
        "seeds": seeds,
        "scene_ids": scene_ids,
        "K": K,
        "snr": snr,
        "valid_angles": valid_angles,
    }


def _cross_check(summaries: list[dict]) -> None:
    if len(summaries) < 2:
        return
    print("\n=== Cross-split disjointness checks ===")
    for i, a in enumerate(summaries):
        for b in summaries[i + 1:]:
            seed_overlap = np.intersect1d(a["seeds"], b["seeds"]).size
            id_overlap = np.intersect1d(a["scene_ids"], b["scene_ids"]).size
            status_s = "OK" if seed_overlap == 0 else "** OVERLAP **"
            status_i = "OK" if id_overlap == 0 else "** OVERLAP **"
            print(f"  {a['label']}  vs  {b['label']}:  "
                  f"seed overlap = {seed_overlap:>6d} {status_s}   "
                  f"scene_id overlap = {id_overlap:>6d} {status_i}")


def _plot_histograms(summaries: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(summaries)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3 * n_rows), squeeze=False)
    for r, s in enumerate(summaries):
        axes[r, 0].hist(s["K"], bins=np.arange(s["K"].min(), s["K"].max() + 2) - 0.5,
                        edgecolor="black")
        axes[r, 0].set_title(f"{s['label']}: K"); axes[r, 0].set_xlabel("n_sources")
        axes[r, 0].set_ylabel("count")

        axes[r, 1].hist(s["snr"], bins=30, edgecolor="black")
        axes[r, 1].set_title(f"{s['label']}: SNR (dB)"); axes[r, 1].set_xlabel("snr_db")

        axes[r, 2].hist(s["valid_angles"], bins=60, edgecolor="black")
        axes[r, 2].set_title(f"{s['label']}: angles (deg)")
        axes[r, 2].set_xlabel("angle_deg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nWrote histograms → {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+", type=Path,
                    help="One or more manifest .npy paths (sibling .yaml auto-detected).")
    ap.add_argument("--plot", action="store_true", help="Emit histogram PNG.")
    ap.add_argument("--out", type=Path, default=Path("experiments/verify/manifest_hist.png"),
                    help="Histogram output path when --plot is set.")
    ap.add_argument("--check-split-disjoint", action="store_true",
                    help="Verify seeds and scene_ids don't overlap across manifests.")
    args = ap.parse_args(argv)

    summaries: list[dict] = []
    for p in args.manifests:
        man = SceneManifest.load(p)
        summaries.append(_summarise(man, label=p.stem))

    if args.check_split_disjoint:
        _cross_check(summaries)

    if args.plot:
        _plot_histograms(summaries, args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
