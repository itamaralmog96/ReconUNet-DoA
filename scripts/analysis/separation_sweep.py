#!/usr/bin/env python3
"""Separation sweep: two equal-power direct sources at angular separation
dtheta in {2,4,6,8,10,15} deg, centre uniformly in [-45, 45] deg, mild imperfections,
no multipath, T=512, at 0 and -5 dB.  1000 angle configurations per separation; the
same configuration (seed, imperfections, sources) is rendered at both SNRs.

Reports pooled RMSE, median RMSPE and the resolution probability
P(both |theta_hat_i - theta_i| < dtheta/2) for Root-MUSIC, ESPRIT, ReconUNet(+Root-MUSIC),
SubspaceNet and DA-MUSIC (K=2 model), plus the stochastic CRLB.

Output: experiments/runs/sweeps_20260920/separation_sweep.csv with columns
sep_deg, snr_db, method, rmse_deg, median_rmspe_deg, resolution_prob, n.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, Models, chunks, scm, sq_err_deg2  # noqa: E402

from reconunet.data.scene_dataset import SceneDataset  # noqa: E402
from reconunet.data.scene_manifest import SceneManifest  # noqa: E402
from reconunet.evaluation.unified_harness import stochastic_crlb_deg  # noqa: E402
from reconunet.models.deep_learning.subspace_models import esprit  # noqa: E402

SEPS = (2.0, 4.0, 6.0, 8.0, 10.0, 15.0)
SNRS = (0.0, -5.0)


def build_manifest(meta, sep: float, n_cfg: int, seed: int) -> SceneManifest:
    rng = np.random.default_rng(seed)
    man = SceneManifest.fixed_angles_snr_sweep(meta, n_angle_configs=n_cfg, snr_levels_db=list(SNRS), rng=rng,
                                              k=2, min_separation_deg=1.0, array_errors="mild")
    centres = rng.uniform(-45.0, 45.0, size=n_cfg)
    rows = man.raw
    for i in range(n_cfg):
        for j in range(len(SNRS)):
            r = rows[i * len(SNRS) + j]
            ang = np.full(r["angles_deg"].shape, np.nan, dtype=np.float32)
            ang[0], ang[1] = centres[i] - sep / 2, centres[i] + sep / 2
            r["angles_deg"] = ang
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-configs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output-dir", "-o", type=Path, default=REPO / "experiments/runs/sweeps_20260920")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = Models(dev)
    meta = SceneManifest.load(str(REPO / "data/scenes/scenarios/moderate/test.npy")).meta
    M = int(meta.M); T = int(meta.T); K = 2
    rows = []
    for si, sep in enumerate(SEPS):
        man = build_manifest(meta, sep, a.n_configs, a.seed + si)
        ds = SceneDataset(man)
        for snr in SNRS:
            idx = np.where(np.abs(man.raw["snr_db"] - snr) < 0.5)[0]
            samples = [ds[int(i)] for i in idx]
            X = torch.stack([s.snapshots for s in samples], 0)
            true = np.stack([s.angles_rad.numpy()[:K] for s in samples])
            preds = {"Root-MUSIC": [], "ESPRIT": [], "ReconUNet": [], "SubspaceNet": [], "DA-MUSIC": []}
            for ch in chunks(list(range(X.shape[0])), a.batch):
                xb = X[ch]
                preds["Root-MUSIC"].append(Models.root_music(xb, K))
                with torch.no_grad():
                    preds["ESPRIT"].append(esprit(scm(xb), K, xb.shape[0]).numpy()[:, :K])
                preds["ReconUNet"].append(mdl.reconunet(xb, K))
                preds["SubspaceNet"].append(mdl.subspacenet(xb, K))
                d = mdl.damusic(xb, K)
                if d is not None: preds["DA-MUSIC"].append(d)
            for m, pl in preds.items():
                if not pl: continue
                p = np.concatenate(pl); e = sq_err_deg2(p, true)
                err_deg = np.abs(np.rad2deg(np.sort(p, -1)[:, :K] - np.sort(true, -1)))
                resolved = float((err_deg < sep / 2).all(axis=1).mean())
                rows.append({"sep_deg": sep, "snr_db": snr, "method": m, "rmse_deg": float(np.sqrt(e.mean())),
                             "median_rmspe_deg": float(np.median(np.sqrt(e.mean(1)))), "resolution_prob": resolved, "n": int(e.shape[0])})
            crlb = np.sqrt(np.nanmean([stochastic_crlb_deg(M=M, T=T, angles_rad=r, snr_db=snr,
                                                            element_spacing_lambda=meta.element_spacing_lambda) ** 2 for r in true]))
            rows.append({"sep_deg": sep, "snr_db": snr, "method": "CRLB", "rmse_deg": float(crlb),
                         "median_rmspe_deg": float("nan"), "resolution_prob": float("nan"), "n": int(true.shape[0])})
            print(f"[sep] {sep:4.1f} deg @ {snr:+.0f} dB  " + "  ".join(
                f"{r['method']}={r['rmse_deg']:.2f}/{r['resolution_prob']:.2f}" for r in rows[-6:-1]))
    out = a.output_dir / "separation_sweep.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sep_deg", "snr_db", "method", "rmse_deg", "median_rmspe_deg", "resolution_prob", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"[sep] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
