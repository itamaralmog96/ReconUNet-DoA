#!/usr/bin/env python3
"""Forward-backward spatial smoothing (FBSS) Root-MUSIC baseline for the coherent
scenarios (reviewer R2-5c), on the Moderate (K=2 + 1) and Crowded (K=4 + 3)
scenario sets with mild imperfections, all SNR levels (0 dB is the Table II row).

FBSS: with sub-array length L and P = M - L + 1 forward sub-arrays,
    R_f  = (1/P) sum_p J_p R J_p^T ,   R_fb = (R_f + J conj(R_f) J) / 2   (J = exchange matrix)
then Root-MUSIC on the L x L smoothed covariance with the true number of direct
sources K.  Decorrelating C coherent arrivals needs 2P >= C and resolving K needs
L >= K + 1; L in {5, 6, 7} covers the sensible range for M = 8.  Plain Root-MUSIC
and ReconUNet + Root-MUSIC on the same scenes are included for reference.

Output: experiments/runs/sweeps_20260920/fbss_baseline.csv with columns
scenario, snr_db, method, rmse_deg, median_rmspe_deg, n.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, Models, chunks, root_music_rad, scm, sq_err_deg2  # noqa: E402

from reconunet.data.scene_dataset import SceneDataset  # noqa: E402
from reconunet.data.scene_manifest import SceneManifest  # noqa: E402

SCEN = [("moderate", 2), ("advanced2_crowded", 4)]


def fbss(R: torch.Tensor, L: int) -> torch.Tensor:
    """Forward-backward spatially smoothed covariance [g, L, L] from R [g, M, M]."""
    M = R.shape[-1]; P = M - L + 1
    Rf = torch.stack([R[:, p:p + L, p:p + L] for p in range(P)], 0).mean(0)
    J = torch.flip(torch.eye(L, dtype=R.dtype), dims=[0])
    return 0.5 * (Rf + J @ Rf.conj() @ J)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subarrays", type=int, nargs="*", default=[5, 6, 7])
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--snr-only", type=float, default=None, help="evaluate a single SNR level")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--output-dir", "-o", type=Path, default=REPO / "experiments/runs/sweeps_20260920")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = Models(dev, with_damusic=False)
    rows = []
    for scen, K in SCEN:
        man = SceneManifest.load(str(REPO / f"data/scenes/scenarios/{scen}/test.npy")); ds = SceneDataset(man)
        snr_all = man.raw["snr_db"]
        for lvl in sorted(np.unique(np.round(snr_all))):
            if a.snr_only is not None and abs(lvl - a.snr_only) > 0.5: continue
            idx = np.where(np.abs(snr_all - lvl) < 1.0)[0]
            if a.max_scenes: idx = idx[: a.max_scenes]
            errs = {}
            for ch in chunks(idx.tolist(), a.batch):
                samples = [ds[int(i)] for i in ch]
                X = torch.stack([s.snapshots for s in samples], 0); true = np.stack([s.angles_rad.numpy()[:K] for s in samples])
                R = scm(X)
                errs.setdefault("Root-MUSIC", []).append(sq_err_deg2(root_music_rad(R, K), true))
                errs.setdefault("ReconUNet+Root-MUSIC", []).append(sq_err_deg2(mdl.reconunet(X, K), true))
                for L in a.subarrays:
                    if L < K + 1: continue
                    errs.setdefault(f"FBSS(L={L})+Root-MUSIC", []).append(sq_err_deg2(root_music_rad(fbss(R, L), K), true))
            for m, e in errs.items():
                e = np.concatenate(e)
                rows.append({"scenario": scen, "snr_db": float(lvl), "method": m, "rmse_deg": float(np.sqrt(e.mean())),
                             "median_rmspe_deg": float(np.median(np.sqrt(e.mean(1)))), "n": int(e.shape[0])})
            print(f"[fbss] {scen:18s} {lvl:+.0f} dB  " + "  ".join(f"{r['method']}={r['rmse_deg']:.2f}" for r in rows[-len(errs):]))
    out = a.output_dir / "fbss_baseline.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "snr_db", "method", "rmse_deg", "median_rmspe_deg", "n"]); w.writeheader(); w.writerows(rows)
    print(f"[fbss] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
