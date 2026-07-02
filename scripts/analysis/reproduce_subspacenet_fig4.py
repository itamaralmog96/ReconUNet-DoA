#!/usr/bin/env python3
"""Reproduce SubspaceNet paper Fig. 4 (Shmuel et al. 2023, §IV-B-2):
MSPE vs SNR for M=2 non-coherent sources, T=200, calibrated ULA (N=8),
SNR sweep [-5, 0] dB.

Curves (all on the SAME test scenes, bucketed by SNR):
  * Root-MUSIC / ESPRIT on the raw empirical covariance  (classical baselines)
  * SubspaceNet + Root-MUSIC  (the trained surrogate-covariance model)
  * Stochastic CRLB           (reference lower bound; paper overlays ZZB)

Classical and SubspaceNet angles are produced through the SAME differentiable
``root_music`` / ``esprit`` (subspace_models) so the angle convention is
identical across every curve — the only difference is the covariance fed in
(raw empirical SCM vs the network's surrogate R̂).

Usage::

    python scripts/analysis/reproduce_subspacenet_fig4.py \
        --checkpoint experiments/runs/subspacenet_fig4/checkpoints/best.pt \
        --manifest   data/scenes/fig4_sweep/test.npy \
        --output-dir experiments/runs/subspacenet_fig4/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_dataset import SceneDataset
from reconunet.data.scene_renderer import lag_stack
from reconunet.models.deep_learning.subspace_models import esprit, root_music
from reconunet.models.third_party.subspacenet_adapter import SubspaceNetAdapter
from reconunet.evaluation.unified_harness import stochastic_crlb_deg

SNR_TARGETS = [-5, -4, -3, -2, -1, 0]
M_SOURCES = 2


def _wrapped_err_rad(pred_rad: np.ndarray, true_rad: np.ndarray) -> np.ndarray:
    """Permutation-invariant periodic angle error [B, K] in radians."""
    p = np.sort(pred_rad, axis=-1)
    t = np.sort(true_rad, axis=-1)
    d = p - t
    return (d + np.pi / 2.0) % np.pi - np.pi / 2.0


def _empirical_cov(snaps: torch.Tensor) -> torch.Tensor:
    """[B, M, T] complex → [B, M, M] empirical covariance X X^H / T."""
    T = snaps.shape[-1]
    return (snaps @ snaps.conj().transpose(-1, -2)) / float(T)


def _angles_classic_rootmusic(R: torch.Tensor, K: int) -> np.ndarray:
    deg, _, _ = root_music(R, K, R.shape[0])      # [0,180]
    return np.deg2rad(deg.detach().numpy() - 90.0)


def _angles_classic_esprit(R: torch.Tensor, K: int) -> np.ndarray:
    return esprit(R, K, R.shape[0]).detach().numpy()   # broadside radians


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "-c", required=True, type=Path)
    ap.add_argument("--manifest", "-m", required=True, type=Path)
    ap.add_argument("--output-dir", "-o", required=True, type=Path)
    ap.add_argument("--tau", type=int, default=8)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = SceneManifest.load(str(args.manifest))
    meta = manifest.meta
    ds = SceneDataset(manifest)
    snr_all = manifest.raw["snr_db"]

    # Build + load the trained SubspaceNet.
    adapter = SubspaceNetAdapter(M=M_SOURCES, tau=args.tau, diff_method="root_music")
    model = adapter.build_model({"M": M_SOURCES, "tau": args.tau, "diff_method": "root_music"})
    adapter.load_checkpoint(model, str(args.checkpoint))
    model.eval()

    rows = []
    for target in SNR_TARGETS:
        idx = np.where(np.abs(snr_all - target) < 0.5)[0]
        if idx.size == 0:
            continue
        snaps = torch.stack([ds[int(i)].snapshots for i in idx], dim=0)        # [B,M,T] complex
        true = np.stack([ds[int(i)].angles_rad.numpy()[:M_SOURCES] for i in idx])  # [B,K]

        R_raw = _empirical_cov(snaps)
        with torch.no_grad():
            a_rm = _angles_classic_rootmusic(R_raw, M_SOURCES)
            a_es = _angles_classic_esprit(R_raw, M_SOURCES)
            prepped = lag_stack(snaps, tau=args.tau)
            a_sn = adapter.forward(model, prepped, meta={"tau": args.tau}).angles_pred.detach().numpy()

        crlb_deg2 = np.nanmean([
            stochastic_crlb_deg(M=meta.M, T=meta.T, angles_rad=true[b],
                                snr_db=float(target),
                                element_spacing_lambda=meta.element_spacing_lambda)
            for b in range(true.shape[0])
        ])

        def per_sample_deg(pred):
            """Per-sample RMSPE [deg]; robust to outliers via median."""
            e = _wrapped_err_rad(pred, true)
            return np.rad2deg(np.sqrt(np.mean(e ** 2, axis=-1)))   # [B]

        def stats(pred):
            ps = per_sample_deg(pred)
            mspe_db = 10.0 * np.log10(np.mean((np.deg2rad(ps)) ** 2))  # pooled MSPE [dB]
            return {
                "mean": float(ps.mean()),
                "median": float(np.median(ps)),
                "p90": float(np.percentile(ps, 90)),
                "frac_gt5": float(np.mean(ps > 5.0)),
                "mspe_db": float(mspe_db),
            }

        s_rm, s_es, s_sn = stats(a_rm), stats(a_es), stats(a_sn)
        crlb_mspe_db = 10.0 * np.log10(crlb_deg2 * (np.pi / 180.0) ** 2)
        rows.append({
            "snr_db": target, "n": int(idx.size),
            "rm_median": s_rm["median"], "esprit_median": s_es["median"], "subnet_median": s_sn["median"],
            "rm_mean": s_rm["mean"], "esprit_mean": s_es["mean"], "subnet_mean": s_sn["mean"],
            "subnet_p90": s_sn["p90"], "subnet_frac_gt5": s_sn["frac_gt5"],
            "rm_db": s_rm["mspe_db"], "esprit_db": s_es["mspe_db"], "subnet_rm_db": s_sn["mspe_db"],
            "crlb_db": crlb_mspe_db, "crlb_deg": float(np.sqrt(crlb_deg2)),
        })

    # ---- table to stdout ----
    print(f"\nMedian RMSPE [deg] (robust) | mean in (parens)")
    print(f"{'SNR':>4} {'n':>5} | {'R-MUSIC':>14} {'ESPRIT':>14} {'SubNet+RM':>16} {'CRLB':>7}")
    for r in rows:
        print(f"{r['snr_db']:>4} {r['n']:>5} | "
              f"{r['rm_median']:>6.2f} ({r['rm_mean']:>5.2f}) "
              f"{r['esprit_median']:>6.2f} ({r['esprit_mean']:>5.2f}) "
              f"{r['subnet_median']:>6.2f} ({r['subnet_mean']:>5.2f})  "
              f"{r['crlb_deg']:>6.2f}  [SubNet p90={r['subnet_p90']:.1f} >5deg={r['subnet_frac_gt5']*100:.1f}%]")

    # ---- CSV ----
    import csv
    csv_path = args.output_dir / "fig4_repro.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- plot (MSPE [dB] vs SNR, like paper Fig 4) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snr = [r["snr_db"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(snr, [r["rm_db"] for r in rows], "s-", color="green", label="R-MUSIC (raw SCM)")
    ax.plot(snr, [r["esprit_db"] for r in rows], "v-", color="purple", label="ESPRIT (raw SCM)")
    ax.plot(snr, [r["subnet_rm_db"] for r in rows], "*--", color="C2", lw=2, label="SubspaceNet+R-MUSIC")
    ax.plot(snr, [r["crlb_db"] for r in rows], "k:", label="CRLB")
    ax.set_xlabel("SNR [dB]"); ax.set_ylabel("MSPE [dB]")
    ax.set_title("Reproduction of SubspaceNet Fig. 4\n(M=2 non-coherent, T=200, calibrated ULA N=8)")
    ax.grid(True, ls=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout()
    png = args.output_dir / "fig4_repro.png"
    fig.savefig(png, dpi=150)
    print(f"\nwrote {csv_path}\nwrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
