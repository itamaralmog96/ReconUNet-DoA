#!/usr/bin/env python3
"""Snapshot sweep T in {8,16,32,64,128,256,512} on the Moderate scenario (K=2 + 1
coherent replica, mild imperfections) at 0 dB for ReconUNet(+Root-MUSIC),
Root-MUSIC and SubspaceNet, plus the stochastic CRLB.

The models were trained at T=512 and consume the tau=8 lag stack, whose shape does
not depend on T, so no retraining is involved.  Two ways of forming a T-snapshot
observation from the rendered 512-sample scene are reported (column ``sampling``):

  window     the first T consecutive samples (band-limited sources, coherence time
             ~20 samples -> consecutive snapshots are strongly correlated)
  decimated  T samples spaced 512/T apart (approximately independent snapshots,
             the textbook "T i.i.d. snapshots" assumption)

At T = 8 the tau = 8 lag stack is undefined (lag 7 would use a single sample), so ReconUNet and
SubspaceNet are reported as NaN there (n = 0); Root-MUSIC and the CRLB are still evaluated.

Output: experiments/runs/sweeps_20260920/snapshot_sweep.csv with columns
T, sampling, method, rmse_deg, median_rmspe_deg, n.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, Models, chunks, sq_err_deg2  # noqa: E402

from reconunet.data.scene_dataset import SceneDataset  # noqa: E402
from reconunet.data.scene_manifest import SceneManifest  # noqa: E402
from reconunet.evaluation.unified_harness import stochastic_crlb_deg  # noqa: E402

TS = (8, 16, 32, 64, 128, 256, 512)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="moderate")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--output-dir", "-o", type=Path, default=REPO / "experiments/runs/sweeps_20260920")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = Models(dev, with_damusic=False)

    man = SceneManifest.load(str(REPO / f"data/scenes/scenarios/{a.scenario}/test.npy"))
    ds = SceneDataset(man); meta = man.meta; M = int(meta.M)
    idx = np.where(np.abs(man.raw["snr_db"] - a.snr) < 1.0)[0]
    if a.max_scenes: idx = idx[: a.max_scenes]
    K = int(np.bincount(man.raw["n_sources"].astype(int)[idx]).argmax())
    print(f"[T-sweep] {a.scenario} @ {a.snr:g} dB: {idx.size} scenes, K={K}, device={dev}")

    rows = []
    samples = [ds[int(i)] for i in idx]
    X = torch.stack([s.snapshots for s in samples], 0)                      # [n, M, 512]
    true = np.stack([s.angles_rad.numpy()[:K] for s in samples])
    T0 = X.shape[-1]
    for T in TS:
        for sampling in ("window", "decimated"):
            Xt = X[:, :, :T] if sampling == "window" else X[:, :, :: max(1, T0 // T)][:, :, :T]
            errs = {"Root-MUSIC": [], "ReconUNet": [], "SubspaceNet": []}
            lag_ok = T > mdl.tau                      # the tau-lag stack needs T > tau (lag tau-1 uses T-tau+1 samples)
            for ch in chunks(list(range(Xt.shape[0])), a.batch):
                xb = Xt[ch]
                errs["Root-MUSIC"].append(sq_err_deg2(Models.root_music(xb, K), true[ch]))
                if lag_ok:
                    errs["ReconUNet"].append(sq_err_deg2(mdl.reconunet(xb, K), true[ch]))
                    errs["SubspaceNet"].append(sq_err_deg2(mdl.subspacenet(xb, K), true[ch]))
            for m, e in errs.items():
                if not e:                               # lag-stack model not applicable at this T
                    rows.append({"T": T, "sampling": sampling, "method": m, "rmse_deg": float("nan"),
                                 "median_rmspe_deg": float("nan"), "n": 0}); continue
                e = np.concatenate(e)
                rows.append({"T": T, "sampling": sampling, "method": m, "rmse_deg": float(np.sqrt(e.mean())),
                             "median_rmspe_deg": float(np.median(np.sqrt(e.mean(1)))), "n": int(e.shape[0])})
            if sampling == "window":
                crlb = np.sqrt(np.nanmean([stochastic_crlb_deg(M=M, T=T, angles_rad=r, snr_db=a.snr,
                                                                element_spacing_lambda=meta.element_spacing_lambda) ** 2
                                           for r in true]))
                rows.append({"T": T, "sampling": "both", "method": "CRLB", "rmse_deg": float(crlb),
                             "median_rmspe_deg": float("nan"), "n": int(true.shape[0])})
            print(f"[T-sweep] T={T:3d} {sampling:9s} " + "  ".join(f"{r['method']}={r['rmse_deg']:.2f}" for r in rows[-3:]))
    out = a.output_dir / "snapshot_sweep.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["T", "sampling", "method", "rmse_deg", "median_rmspe_deg", "n"]); w.writeheader(); w.writerows(rows)
    print(f"[T-sweep] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
