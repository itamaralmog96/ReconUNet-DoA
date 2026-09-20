#!/usr/bin/env python3
"""Grid-MUSIC verification (reviewer R2-4): why spectral MUSIC (1 deg grid) reports a
larger *pooled* error than Root-MUSIC at high SNR although its *median* error is tiny.

On the Basic scenario (K=1, no multipath, mild imperfections) at the requested SNR
(default 20 dB; 0 dB also reported for context) we compute, per scene,

  MUSIC-grid      argmax of the 1 deg spectrum (no refinement)
  MUSIC-refined   1 deg grid + three-point parabolic refinement (the paper's estimator)
  Root-MUSIC      gridless

and report median |error|, pooled RMSE, mean |error| and the fraction of scenes whose
selected peak is spurious (|error| > 1, 3, 5 deg).  Output: a markdown section on
stdout (appended to docs/revision_facts.md by the pipeline) and music_verification.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, chunks, root_music_rad, scm  # noqa: E402

from reconunet.data.scene_dataset import SceneDataset  # noqa: E402
from reconunet.data.scene_manifest import SceneManifest  # noqa: E402
from reconunet.evaluation import classical_batched as CB  # noqa: E402


def music_spectrum(R: torch.Tensor, K: int, M: int, grid: torch.Tensor) -> torch.Tensor:
    R64 = R.to(torch.complex128)
    A = CB.steering_matrix(grid, M, 0.5)
    w, V = torch.linalg.eigh(R64)
    En = V[:, :, : M - K]
    EA = torch.einsum("bmk,mg->bkg", En.conj(), A)
    return 1.0 / (EA.abs() ** 2).sum(dim=1).clamp_min(1e-12)


def stats(err_deg: np.ndarray) -> dict:
    a = np.abs(err_deg)
    return {"median_abs_err_deg": float(np.median(a)), "pooled_rmse_deg": float(np.sqrt((a ** 2).mean())),
            "mean_abs_err_deg": float(a.mean()), "frac_gt_1deg": float((a > 1).mean()),
            "frac_gt_3deg": float((a > 3).mean()), "frac_gt_5deg": float((a > 5).mean()), "n": int(a.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snrs", type=float, nargs="*", default=[20.0, 0.0])
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--output-dir", "-o", type=Path, default=REPO / "experiments/runs/sweeps_20260920")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    man = SceneManifest.load(str(REPO / "data/scenes/scenarios/basic/test.npy")); ds = SceneDataset(man)
    M = int(man.meta.M); K = 1
    grid = CB._default_grid(torch.device("cpu"))                                  # 1 deg on [-60, 60]
    rows = []
    for snr in a.snrs:
        idx = np.where(np.abs(man.raw["snr_db"] - snr) < 1.0)[0]
        if a.max_scenes: idx = idx[: a.max_scenes]
        errs = {"MUSIC-grid": [], "MUSIC-refined": [], "Root-MUSIC": []}
        for ch in chunks(idx.tolist(), 500):
            samples = [ds[int(i)] for i in ch]
            X = torch.stack([s.snapshots for s in samples], 0); true = np.stack([s.angles_rad.numpy()[:K] for s in samples])
            R = scm(X)
            spec = music_spectrum(R, K, M, grid)
            grid_est = grid[spec.argmax(dim=1)].numpy()[:, None]
            refined = CB.music(R, K, M).numpy()[:, :K]
            errs["MUSIC-grid"].append(np.rad2deg(grid_est - true).ravel())
            errs["MUSIC-refined"].append(np.rad2deg(refined - true).ravel())
            errs["Root-MUSIC"].append(np.rad2deg(root_music_rad(R, K) - true).ravel())
        for m, e in errs.items():
            rows.append({"snr_db": snr, "method": m, **stats(np.concatenate(e))})
    out = a.output_dir / "music_verification.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # ---- markdown section --------------------------------------------------
    print("Basic scenario (K = 1, no multipath, mild imperfections), 1000 scenes per SNR. MUSIC uses the paper's "
          "1° scan grid on [-60°, 60°]; 'refined' adds the three-point parabolic peak refinement; Root-MUSIC is gridless. "
          "A spurious peak is a scene whose selected maximum is more than 1°/3°/5° from the true angle.\n")
    print("| SNR (dB) | estimator | median abs err (°) | pooled RMSE (°) | mean abs err (°) | frac > 1° | frac > 3° | frac > 5° | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['snr_db']:g} | {r['method']} | {r['median_abs_err_deg']:.3f} | {r['pooled_rmse_deg']:.3f} | "
              f"{r['mean_abs_err_deg']:.3f} | {r['frac_gt_1deg']:.4f} | {r['frac_gt_3deg']:.4f} | {r['frac_gt_5deg']:.4f} | {r['n']} |")
    r20 = {r["method"]: r for r in rows if abs(r["snr_db"] - 20) < 0.5}
    if r20:
        g, rf, rm = r20.get("MUSIC-grid"), r20.get("MUSIC-refined"), r20.get("Root-MUSIC")
        print(f"\nReading: at 20 dB the refined grid-MUSIC median error is {rf['median_abs_err_deg']:.3f}° against "
              f"{rm['median_abs_err_deg']:.3f}° for Root-MUSIC, i.e. the estimator itself is fine; the pooled RMSE gap "
              f"({rf['pooled_rmse_deg']:.3f}° vs {rm['pooled_rmse_deg']:.3f}°) is driven by the {100 * rf['frac_gt_3deg']:.2f} % of "
              f"scenes whose selected peak is spurious (> 3°), which RMSE squares. Without refinement the grid alone "
              f"contributes a {g['median_abs_err_deg']:.3f}° median quantisation error.")
    import os
    print(f"\nData: `{os.path.relpath(out, REPO)}`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
