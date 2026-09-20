#!/usr/bin/env python3
"""Bootstrap 95 % confidence intervals for every cell of the Table II, scenario-sweep
and paper-test-split tables, from the per-scene squared errors dumped by

    reproduce_table2.py / scenario_sweep.py / compare_paper_testset.py  --dump-errors X.npz

Each npz key is ``scenario|snr|method`` (or ``K=k||method`` for the test split) holding an
``[n_scenes, K]`` array of per-source squared errors in deg².  Resampling is over
*scenes* (with replacement, ``--resamples`` draws, default 1000, seed 20260920), and for
each draw we recompute

  * pooled RMSE   sqrt(sum of resampled scene error-sums / sum of resampled source counts)  (paper eq. 31)
  * median RMSPE  median over resampled scenes of sqrt(mean_k e_k)

The 2.5 / 97.5 percentiles of the draws are the reported interval (percentile bootstrap).
For the test split an ``all K`` row pools the K buckets (mixed K handled through the
sums/counts formulation).

Usage::

    python scripts/analysis/bootstrap_ci.py errors.npz [-o out_ci.csv] [--resamples 1000]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

COLS = ["table", "scenario", "snr_db", "method", "n_scenes", "rmse_deg", "rmse_ci_lo", "rmse_ci_hi",
        "median_rmspe_deg", "median_ci_lo", "median_ci_hi", "resamples"]


def boot(E: np.ndarray, B: int, rng: np.random.Generator):
    """E: [n, K] squared errors (deg^2), NaN-free.  Returns point estimates and CIs."""
    E = np.asarray(E, dtype=np.float64)
    n = E.shape[0]
    sums = E.sum(axis=1); cnt = np.full(n, E.shape[1], dtype=np.float64)
    ps = np.sqrt(E.mean(axis=1))
    return _boot_sums(sums, cnt, ps, B, rng)


def _boot_sums(sums, cnt, ps, B, rng):
    n = sums.size
    point_rmse = float(np.sqrt(sums.sum() / cnt.sum())); point_med = float(np.median(ps))
    idx = rng.integers(0, n, size=(B, n))
    rmse_b = np.sqrt(sums[idx].sum(axis=1) / cnt[idx].sum(axis=1))
    med_b = np.median(ps[idx], axis=1)
    lo, hi = np.percentile(rmse_b, [2.5, 97.5]); mlo, mhi = np.percentile(med_b, [2.5, 97.5])
    return dict(n_scenes=int(n), rmse_deg=point_rmse, rmse_ci_lo=float(lo), rmse_ci_hi=float(hi),
                median_rmspe_deg=point_med, median_ci_lo=float(mlo), median_ci_hi=float(mhi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--output", "-o", type=Path, default=None)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--table", default=None, help="label for the 'table' column (default: npz stem)")
    a = ap.parse_args()
    out = a.output or a.npz.with_name(a.npz.stem.replace("errors", "").strip("_") + "_ci.csv" if "errors" in a.npz.stem
                                      else a.npz.stem + "_ci.csv")
    table = a.table or a.npz.parent.name
    rng = np.random.default_rng(a.seed)
    z = np.load(a.npz)
    rows = []
    pooled_by_method: dict[str, list] = {}
    for key in z.files:
        E = z[key]
        if E.ndim == 1: E = E[:, None]
        scen, snr, method = key.split("|")
        r = boot(E, a.resamples, rng)
        rows.append({"table": table, "scenario": scen, "snr_db": snr, "method": method, "resamples": a.resamples, **r})
        if scen.startswith("K="):
            pooled_by_method.setdefault(method, []).append(E)
    for method, Es in pooled_by_method.items():            # test split: all-K row (mixed K)
        sums = np.concatenate([E.sum(1) for E in Es]); cnt = np.concatenate([np.full(E.shape[0], E.shape[1], float) for E in Es])
        ps = np.concatenate([np.sqrt(E.mean(1)) for E in Es])
        r = _boot_sums(sums.astype(float), cnt, ps, a.resamples, rng)
        rows.append({"table": table, "scenario": "all K", "snr_db": "", "method": method, "resamples": a.resamples, **r})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"[ci] {len(rows)} cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
