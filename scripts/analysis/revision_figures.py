#!/usr/bin/env python3
"""Revision figures regenerated with the released (corrected-renderer) ReconUNet
checkpoint, as vector PDF with >= 9 pt fonts (+ PNG previews) in docs/figs_revision/.

  angle_sweep_0dB.pdf            RMSE vs true DoA for K = 1 at 0 dB (mild imperfections, no multipath):
                                 Root-MUSIC, SubspaceNet, DA-MUSIC (K=1) and ReconUNet + Root-MUSIC,
                                 200 scenes per angle on a 5 deg grid from -60 to 60 deg (endfire ill-conditioning)
  music_spectrum_moderate_0dB.pdf  MUSIC pseudo-spectrum (0.1 deg grid) of one Moderate scene at 0 dB on the raw
                                 sample covariance and on the ReconUNet reconstruction, true DoAs marked
  rootmusic_roots_moderate_0dB.pdf  Root-MUSIC roots of the same scene on the unit circle (raw vs reconstructed),
                                 selected roots highlighted, true-DoA roots marked

Usage::  python scripts/analysis/revision_figures.py [--scenes-per-angle 200] [--scene-index 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, Models, root_music_rad, scm, sq_err_deg2  # noqa: E402

from reconunet.data.scene_dataset import SceneDataset  # noqa: E402
from reconunet.data.scene_manifest import SceneManifest  # noqa: E402
from reconunet.evaluation import classical_batched as CB  # noqa: E402
from reconunet.models.deep_learning.subspace_models import (  # noqa: E402
    _noise_projector_batched, find_roots_batched, sum_of_diags_batched)

OUT = REPO / "docs/figs_revision"
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10.5, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
C = {"Root-MUSIC": "#4a4a4a", "SubspaceNet": "#7f7f7f", "DA-MUSIC": "#b0b0b0", "ReconUNet + Root-MUSIC": "#1f4e79",
     "raw SCM": "#7f7f7f", "ReconUNet R̂": "#1f4e79"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=170); plt.close(fig)
    print(f"[figs] wrote {OUT / name}.pdf/.png")


def angle_sweep(mdl, meta, per_angle: int, seed: int):
    angles = np.arange(-60.0, 60.1, 5.0)
    rng = np.random.default_rng(seed)
    man = SceneManifest.fixed_angles_snr_sweep(meta, n_angle_configs=len(angles) * per_angle, snr_levels_db=[0.0],
                                              rng=rng, k=1, array_errors="mild")
    rows = man.raw
    for i in range(len(rows)):
        ang = np.full(rows[i]["angles_deg"].shape, np.nan, dtype=np.float32); ang[0] = angles[i // per_angle]
        rows[i]["angles_deg"] = ang
    ds = SceneDataset(man)
    res = {m: [] for m in ("Root-MUSIC", "SubspaceNet", "DA-MUSIC", "ReconUNet + Root-MUSIC")}
    for ai, th in enumerate(angles):
        idx = list(range(ai * per_angle, (ai + 1) * per_angle))
        samples = [ds[i] for i in idx]
        X = torch.stack([s.snapshots for s in samples], 0); true = np.stack([s.angles_rad.numpy()[:1] for s in samples])
        preds = {"Root-MUSIC": Models.root_music(X, 1), "SubspaceNet": mdl.subspacenet(X, 1),
                 "ReconUNet + Root-MUSIC": mdl.reconunet(X, 1)}
        d = mdl.damusic(X, 1)
        if d is not None: preds["DA-MUSIC"] = d
        for m, p in preds.items():
            res[m].append(float(np.sqrt(sq_err_deg2(p, true).mean())))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, ys in res.items():
        if ys: ax.semilogy(angles, ys, marker="o", ms=3.5, lw=1.5, color=C[m], label=m, zorder=3 if "ReconUNet" in m else 2)
    ax.set_xlabel("true DoA (deg, broadside = 0)"); ax.set_ylabel("RMSE (deg)")
    ax.set_title(f"K = 1, 0 dB, mild imperfections, {per_angle} scenes per angle")
    ax.grid(True, which="both", alpha=0.3); ax.legend(frameon=False)
    save(fig, "angle_sweep_0dB")
    return angles, res


def spectrum_and_roots(mdl, scene_index: int):
    man = SceneManifest.load(str(REPO / "data/scenes/scenarios/moderate/test.npy")); ds = SceneDataset(man)
    idx = np.where(np.abs(man.raw["snr_db"]) < 0.5)[0]
    s = ds[int(idx[scene_index])]; K = int(s.n_sources); M = int(man.meta.M)
    X = s.snapshots[None]; R_raw = scm(X); R_hat = mdl.reconunet_cov(X).cpu()
    true_deg = np.rad2deg(s.angles_rad.numpy()[:K])
    grid = torch.deg2rad(torch.arange(-90.0, 90.01, 0.1, dtype=torch.float64))
    A = CB.steering_matrix(grid, M, 0.5)

    def spec(R):
        w, V = torch.linalg.eigh(R.to(torch.complex128)); En = V[:, :, : M - K]
        P = 1.0 / (torch.einsum("bmk,mg->bkg", En.conj(), A).abs() ** 2).sum(1).clamp_min(1e-12)
        return (10 * torch.log10(P / P.max())).numpy()[0]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for lbl, R in (("raw SCM", R_raw), ("ReconUNet R̂", R_hat)):
        ax.plot(np.rad2deg(grid.numpy()), spec(R), lw=1.5, color=C[lbl], label=f"MUSIC on {lbl}")
    for t in true_deg: ax.axvline(t, color="#c0392b", lw=0.9, ls="--")
    ax.plot([], [], color="#c0392b", lw=0.9, ls="--", label="true DoAs")
    ax.set_xlim(-90, 90); ax.set_ylim(-45, 2); ax.set_xlabel("angle (deg)"); ax.set_ylabel("normalised pseudo-spectrum (dB)")
    ax.set_title(f"Moderate scenario scene {scene_index}: K = {K} + 1 coherent replica, 0 dB"); ax.grid(alpha=0.3); ax.legend(frameon=False, loc="lower left")
    save(fig, "music_spectrum_moderate_0dB")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    tt = np.linspace(0, 2 * np.pi, 400)
    for ax, (lbl, R) in zip(axes, (("raw SCM", R_raw), ("ReconUNet R̂", R_hat))):
        F = _noise_projector_batched(R.to(torch.complex64), K)
        roots = find_roots_batched(sum_of_diags_batched(F))[0].numpy()
        est = root_music_rad(R, K)[0]
        ax.plot(np.cos(tt), np.sin(tt), color="#bbbbbb", lw=0.8)
        ax.scatter(roots.real, roots.imag, s=14, color=C[lbl], label="roots")
        inside = roots[np.abs(roots) < 1]; sel = inside[np.argsort(np.abs(np.abs(inside) - 1))[:K]]
        ax.scatter(sel.real, sel.imag, s=60, facecolors="none", edgecolors="#1f4e79", lw=1.4, label="selected (K)")
        zt = np.exp(-1j * np.pi * np.sin(np.deg2rad(true_deg)))
        ax.scatter(zt.real, zt.imag, marker="x", s=50, color="#c0392b", lw=1.4, label="true DoAs")
        ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.grid(alpha=0.3)
        ax.set_title(f"{lbl}: est {np.round(np.rad2deg(np.sort(est)), 1).tolist()}°")
        ax.set_xlabel("Re"); ax.set_ylabel("Im")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Root-MUSIC roots, Moderate scene {scene_index} at 0 dB (true {np.round(np.sort(true_deg), 1).tolist()}°)", fontsize=10)
    fig.tight_layout()
    save(fig, "rootmusic_roots_moderate_0dB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-per-angle", type=int, default=200)
    ap.add_argument("--scene-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = Models(dev)
    meta = SceneManifest.load(str(REPO / "data/scenes/scenarios/basic/test.npy")).meta
    angles, res = angle_sweep(mdl, meta, a.scenes_per_angle, a.seed)
    import csv
    with (OUT / "angle_sweep_0dB.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["angle_deg"] + list(res)); [w.writerow([angles[i]] + [res[m][i] if res[m] else "" for m in res]) for i in range(len(angles))]
    spectrum_and_roots(mdl, a.scene_index)
    print("[figs] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
