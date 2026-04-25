#!/usr/bin/env python3
"""``run_classic_on_manifest.py`` — sweep classical DOA algorithms on a manifest.

For each randomly-selected scene, renders it, feeds the resulting
covariance (or snapshots) to the chosen classical estimators, and
reports permutation-invariant RMSPE (degrees, broadside-0 convention)
aggregated by SNR bucket (or by K, modulation, preset, etc.).

Angle-convention translation
----------------------------
The manifest stores angles as broadside-0° degrees (range e.g. [-60°, 60°]),
rendered with steering ``a_m(θ_b) = exp(-j·2π·m·(d/λ)·sin(θ_b))`` (−j sign).
The classical ``ArrayModel`` linear array lies on the x-axis with steering
``a_m(ψ_c) = exp(+j·2π·m·(d/λ)·cos(ψ_c))`` (+j sign, endfire-0°).  Because
the j-signs are opposite, matching the two steering vectors requires
``cos(ψ_c) = -sin(θ_b)`` → ``ψ_c = 90° + θ_b`` →
``θ_b = ψ_c − 90°`` (the conversion used when returning results).
The scan grid is restricted to [0°, 180°] to avoid the ULA's front/back
image.

Calibration policy
------------------
The classic model's ``ArrayModel`` is configured with imperfections
*disabled* so that its steering matrix is nominal.  The rendered
snapshots include the per-scene imperfection, so feeding them to a
nominal estimator measures the effect of mis-calibration — which is the
whole point of the imperfection sweep.

Usage
-----
    python scripts/verify/run_classic_on_manifest.py \\
        data/scenes/verify/tiny/val.npy --algo music --n 200 --true-k

    python scripts/verify/run_classic_on_manifest.py \\
        data/scenes/verify/snr_sweep/test.npy \\
        --algo music,rootmusic,esprit --n 2000 --true-k \\
        --group-by snr_db --snr-bins "-10,-5,0,5,10,15,20" \\
        --out experiments/verify/snr_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
# Both paths are needed: one for `import reconunet`, one so the legacy
# absolute import `from signalgen.signal_generator import SignalConfig`
# inside reconunet.signalgen.array_processing resolves cleanly.
for p in (_SRC, _SRC / "reconunet"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import reconunet bits AFTER path manipulation.
from reconunet.data.scene_manifest import SceneManifest           # noqa: E402
from reconunet.data.scene_renderer import SceneRenderer            # noqa: E402
from reconunet.signalgen.array_processing import (                 # noqa: E402
    ArrayConfig,
    ArrayModel,
)
from reconunet.models.classic import (                             # noqa: E402
    Beamformer,
    MVDR,
    MUSIC,
    RootMUSIC,
    ESPRIT,
    UnitaryESPRIT,
)

_ALGO_MAP = {
    "beamformer":     Beamformer,
    "mvdr":           MVDR,
    "music":          MUSIC,
    "rootmusic":      RootMUSIC,
    "esprit":         ESPRIT,
    "unitaryesprit":  UnitaryESPRIT,
}


def _build_array(meta) -> ArrayModel:
    """Build a NOMINAL linear ArrayModel matching the manifest geometry.

    Imperfections are disabled here: we want to measure the cost of model
    mismatch against the perturbed data.
    """
    cfg = ArrayConfig(
        array_type="linear",
        num_elements=int(meta.M),
        carrier_freq=2.45e9,                   # any consistent choice
        element_spacing=float(meta.element_spacing_lambda),
        enable_gain_phase_errors=False,
        enable_mutual_coupling=False,
        position_error_std=0.0,
    )
    return ArrayModel(cfg, seed=0)


def _instantiate(algo: str, array: ArrayModel, K: Optional[int]):
    """Construct a classic model with a half-plane scan grid [0°, 180°]."""
    cls = _ALGO_MAP[algo]
    scan = np.linspace(0.0, 180.0, 721)         # 0.25° resolution
    # All BaseDOAModel subclasses accept (array_model, array_manifold=None,
    # scan_angles_deg=None, num_sources=None) at minimum.
    return cls(array_model=array, scan_angles_deg=scan, num_sources=K)


def _rmspe_deg(true_deg: np.ndarray, est_deg: np.ndarray) -> float:
    """Permutation-invariant RMSE for sorted 1D angle vectors (degrees)."""
    K_true, K_est = true_deg.size, est_deg.size
    K = min(K_true, K_est)
    if K == 0:
        return float("nan")
    t = np.sort(true_deg)[:K]
    e = np.sort(est_deg)[:K]
    return float(np.sqrt(np.mean((t - e) ** 2)))


def _pick_bucket(value: float, edges: np.ndarray) -> int:
    """Return bucket index of ``value`` inside a sorted edges array."""
    idx = int(np.argmin(np.abs(edges - value)))
    return idx


def _run_one(scene, renderer, array, algo: str, K: Optional[int],
             use_snapshots: bool) -> tuple[np.ndarray, float]:
    """Render + estimate; return (est_deg_broadside0, elapsed_sec)."""
    res = renderer.render(scene)
    model = _instantiate(algo, array, K)
    t0 = time.perf_counter()
    if use_snapshots:
        model.set_received_data(res.snapshots)
    else:
        model.set_received_covariance(res.covariance)
    est_deg_classic, _spectrum = model.estimate_doa()
    dt = time.perf_counter() - t0

    est_deg_classic = np.asarray(est_deg_classic, dtype=np.float64)
    # Trim to K peaks if more were returned.
    K_est = int(K if K is not None else est_deg_classic.size)
    est_deg_classic = est_deg_classic[:K_est]
    # θ_classic (endfire=0°, from x-axis, +j·cos(ψ) steering) →
    # θ_manifest (broadside=0°, -j·sin(θ_b) steering in the new SceneRenderer).
    # The renderer's steering sign is opposite to the legacy ArrayModel's,
    # so matching requires cos(ψ) = -sin(θ_b), i.e. ψ = θ_b + 90°, hence:
    est_deg_bs = est_deg_classic - 90.0
    return est_deg_bs, dt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--algo", default="music",
                    help="Comma-separated list from: "
                         + ",".join(sorted(_ALGO_MAP.keys())))
    ap.add_argument("--n", type=int, default=500,
                    help="Number of scenes to sample from the manifest.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    ap.add_argument("--true-k", action="store_true",
                    help="Pass the manifest K per scene (recommended for QA).")
    ap.add_argument("--fixed-k", type=int, default=None,
                    help="Override K with a single value for every scene.")
    ap.add_argument("--use-snapshots", action="store_true",
                    help="Feed raw snapshots via set_received_data (vs covariance).")
    ap.add_argument("--group-by", choices=("snr_db", "K", "preset", "modulation"),
                    default="snr_db")
    ap.add_argument("--snr-bins", type=str, default=None,
                    help="Comma-separated bucket centres (dB), e.g. -10,-5,0,5,10,15,20.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write per-scene results as CSV.")
    args = ap.parse_args(argv)

    algos = [a.strip().lower() for a in args.algo.split(",") if a.strip()]
    for a in algos:
        if a not in _ALGO_MAP:
            raise SystemExit(f"Unknown algo {a!r}.  Choose from: {sorted(_ALGO_MAP)}")

    man = SceneManifest.load(args.manifest)
    renderer = SceneRenderer(man.meta)
    array = _build_array(man.meta)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(man), size=min(args.n, len(man)), replace=False)

    # Collect rows for CSV + aggregation.
    rows: List[Dict] = []
    for scene_idx in idx:
        scene = man[int(scene_idx)]
        K_true = int(scene.n_sources)
        K = args.fixed_k if args.fixed_k is not None else (K_true if args.true_k else None)
        true_deg = np.sort(scene.angles_deg[:K_true].astype(np.float64))

        for algo in algos:
            try:
                est_deg, dt = _run_one(scene, renderer, array, algo,
                                       K=K, use_snapshots=args.use_snapshots)
                err = _rmspe_deg(true_deg, est_deg)
            except Exception as exc:                # noqa: BLE001
                est_deg = np.array([])
                err = float("nan")
                dt = float("nan")
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  [!] scene {scene_idx} algo={algo}: {msg}", file=sys.stderr)
            rows.append(dict(
                scene_idx=int(scene_idx),
                scene_id=int(scene.scene_id),
                algo=algo,
                snr_db=float(scene.snr_db),
                K=K_true,
                modulation=int(scene.modulation),
                gain_err_dB=float(scene.gain_err_dB),
                phase_err_deg=float(scene.phase_err_deg),
                mutual_coupling=float(scene.mutual_coupling),
                position_err_pct=float(scene.position_err_pct),
                rmspe_deg=err,
                elapsed_s=dt,
                true_deg=";".join(f"{x:+.4f}" for x in true_deg),
                est_deg=";".join(f"{x:+.4f}" for x in np.sort(est_deg)),
            ))

    # CSV output (optional).
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out}  ({len(rows)} rows)")

    # Aggregation by --group-by.
    def _bucket_label(r: dict) -> str:
        if args.group_by == "snr_db":
            if args.snr_bins:
                edges = np.array([float(x) for x in args.snr_bins.split(",")])
                return f"{edges[_pick_bucket(r['snr_db'], edges)]:+g}"
            return f"{r['snr_db']:+.1f}"
        if args.group_by == "K":
            return f"K={r['K']}"
        if args.group_by == "modulation":
            return f"mod={r['modulation']}"
        # preset: infer from imperfection magnitudes
        g, p, c, pos = r["gain_err_dB"], r["phase_err_deg"], r["mutual_coupling"], r["position_err_pct"]
        if np.allclose([g, p, c, pos], 0.0, atol=1e-4):
            return "none"
        if max(g, p, c, pos) < 0.3 * 5:       # loose bound
            return "mild"
        return "harsh"

    # Group → {algo: [err, ...]}
    groups: Dict[str, Dict[str, list[float]]] = {}
    for r in rows:
        lbl = _bucket_label(r)
        groups.setdefault(lbl, {}).setdefault(r["algo"], []).append(r["rmspe_deg"])

    # Pretty print.
    header = f"{args.group_by:>10s}  |  " + "  ".join(f"{a:>14s}" for a in algos)
    print("\n" + header)
    print("-" * len(header))
    for lbl in sorted(groups.keys(),
                      key=lambda s: float(s) if _is_number(s) else s):
        cells = []
        for a in algos:
            vals = np.array(groups[lbl].get(a, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                cells.append("       —      ")
            else:
                cells.append(f"{vals.mean():7.3f}° (n={vals.size:>3d})")
        print(f"{lbl:>10s}  |  " + "  ".join(cells))

    return 0


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
