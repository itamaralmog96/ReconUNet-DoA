#!/usr/bin/env python3
"""Paper-style RMSE-vs-SNR sweep across the four §IV scenarios for the 3-way
comparison + classical Root-MUSIC, with the stochastic CRLB floor.

For each scenario (Basic / Moderate / OOD / Crowded) and each of the 9 SNR
levels, computes mean direct-path RMSPE (deg) for:

  * classical Root-MUSIC (empirical SCM)
  * SubspaceNet + Root-MUSIC
  * SubViT (grid spectral peak-pick)
  * ReconUNet (EVD-UNet) + Root-MUSIC
  * CRLB (stochastic, theoretical floor)

Writes a long-form CSV and a 2x2 RMSE-vs-SNR grid PNG (log-y), matching the
visual style of the Almog & Weiss paper figures.

Usage::
    python scripts/analysis/scenario_sweep.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_dataset import SceneDataset
from reconunet.data.scene_renderer import lag_stack
from reconunet.models.deep_learning.subspace_models import root_music
from reconunet.models.third_party.subspacenet_adapter import SubspaceNetAdapter
from reconunet.models.third_party.subvit_adapter import SubViTAdapter
from reconunet.models.third_party.damusic_adapter import DAMUSICEnsemble
from reconunet.evaluation.unified_harness import stochastic_crlb_deg

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = [
    ("basic",             "Basic (K=1, no multipath)"),
    ("moderate",          "Moderate (K=2, +1 coherent)"),
    ("advanced1_ood",     "OOD (K=1, +6 coherent)"),
    ("advanced2_crowded", "Crowded (K=4, +3 coherent)"),
]
# "DA-MUSIC" is dropped at runtime when no per-K checkpoints exist yet.
METHODS = ["Root-MUSIC", "SubspaceNet", "SubViT", "DA-MUSIC", "ReconUNet"]
COLORS = {"Root-MUSIC": "#888888", "SubspaceNet": "#1f77b4",
          "SubViT": "#2ca02c", "DA-MUSIC": "#9467bd", "ReconUNet": "#d62728"}


def _sq_err_deg2(pred_rad, true_rad):
    """Per-source squared errors (deg²), paper eq. (31): sorted pairing,
    no wrap.  Pool with sqrt(mean) for the paper's RMSE."""
    p = np.sort(pred_rad, axis=-1); t = np.sort(true_rad, axis=-1)
    return (np.rad2deg(p - t) ** 2).ravel()                          # [g*K]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--subspacenet", type=Path, default=REPO / "experiments/runs/subspacenet_paper/checkpoints/best.pt")
    ap.add_argument("--subvit", type=Path, default=REPO / "experiments/runs/subvit_paper/checkpoints/best.pt")
    ap.add_argument("--reconunet", type=Path, default=REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt")
    ap.add_argument("--damusic-dir", type=Path, default=REPO / "experiments/runs/damusic_paper",
                    help="root holding k<K>/checkpoints/best.pt per source count")
    ap.add_argument("--output-dir", "-o", type=Path, default=REPO / "experiments/runs/scenario_sweep")
    ap.add_argument("--dump-errors", type=Path, default=None,
                    help="also save per-scene per-source squared errors (deg^2) to this .npz for bootstrap CIs")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] device={dev}")

    # one set of models for all scenarios
    sn_ad = SubspaceNetAdapter(M=4, tau=args.tau, diff_method="root_music")
    sn = sn_ad.build_model({"M": 4, "tau": args.tau, "diff_method": "root_music"})
    sn_ad.load_checkpoint(sn, str(args.subspacenet)); sn.to(dev).eval()
    # Self-configure from the checkpoint's saved training config (see
    # compare_paper_testset.py — avoids stale hard-coded architecture).
    sv_init = dict(torch.load(args.subvit, map_location="cpu",
                              weights_only=False)["cfg"]["model"]["init"])
    sv_ad = SubViTAdapter(**sv_init)
    sv = sv_ad.build_model(sv_init); sv_ad.load_checkpoint(sv, str(args.subvit)); sv.to(dev).eval()
    from reconunet.cli.evaluate import _NativeEVDUNetAdapter
    rn_ad = _NativeEVDUNetAdapter("reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet")
    rn = rn_ad.build_model({"M": 8, "tau": args.tau, "activation_type": "anti_rectifier", "use_dropout": True})
    rn_ad.load_checkpoint(rn, str(args.reconunet)); rn.to(dev).eval()
    dm = DAMUSICEnsemble.from_run_dir(args.damusic_dir, device=dev)     # per-K models or None
    if dm is None:
        METHODS[:] = [m for m in METHODS if m != "DA-MUSIC"]

    rows = []; dump = {}
    for scen, _label in SCENARIOS:
        man = SceneManifest.load(str(REPO / f"data/scenes/scenarios/{scen}/test.npy"))
        ds = SceneDataset(man); meta = man.meta; M = int(meta.M); T = int(meta.T)
        snr_all = man.raw["snr_db"]; ns_all = man.raw["n_sources"].astype(int)
        K = int(np.bincount(ns_all).argmax())                      # scenario's (constant) K
        for lvl in sorted(np.unique(np.round(snr_all))):
            idx = np.where(np.abs(snr_all - lvl) < 1.0)[0]
            if idx.size == 0:
                continue
            acc = {m: [] for m in METHODS}; crlbs = []
            for chunk in _chunks(idx.tolist(), args.batch):
                snaps = torch.stack([ds[int(i)].snapshots for i in chunk], dim=0)
                true = np.stack([ds[int(i)].angles_rad.numpy()[:K] for i in chunk])
                g = len(chunk); nsrc = torch.full((g,), K)
                R = (snaps @ snaps.conj().transpose(-1, -2)) / snaps.shape[-1]
                lag = lag_stack(snaps, tau=args.tau).to(dev)
                with torch.no_grad():
                    rm = np.deg2rad(root_music(R, K, g)[0].numpy() - 90.0)[:, :K]
                    sn_p = sn_ad.forward(sn, lag, meta={"tau": args.tau, "n_sources": nsrc}).angles_pred.cpu().numpy()[:, :K]
                    rn_p = rn_ad.forward(rn, lag, meta={"tau": args.tau, "n_sources": nsrc}).angles_pred.cpu().numpy()[:, :K]
                    sv_in = sv_ad.prepare_input(snaps, {"M": M}).to(dev)
                    sv_p = sv_ad.forward(sv, sv_in, meta={"n_sources": nsrc}).angles_pred.cpu().numpy()[:, :K]
                    preds = [("Root-MUSIC", rm), ("SubspaceNet", sn_p), ("SubViT", sv_p), ("ReconUNet", rn_p)]
                    if dm is not None:
                        preds.append(("DA-MUSIC", dm.predict(snaps.to(dev), nsrc).cpu().numpy()[:, :K]))
                for m, pr in preds:
                    acc[m].append(_sq_err_deg2(pr, true))
                for r in true:
                    crlbs.append(stochastic_crlb_deg(M=M, T=T, angles_rad=r[:K],
                                                     snr_db=float(lvl),
                                                     element_spacing_lambda=meta.element_spacing_lambda))
            rec = {"scenario": scen, "snr_db": float(lvl), "n": int(idx.size),
                   "crlb_deg": float(np.sqrt(np.nanmean(np.array(crlbs))))}
            for m in METHODS:
                e_all = np.concatenate(acc[m])
                rec[m] = float(np.sqrt(e_all.mean()))   # pooled RMSE (eq 31)
                if args.dump_errors is not None:
                    dump[f"{scen}|{float(lvl):g}|{m}"] = e_all.astype(np.float32)      # [n, K] deg^2
            rows.append(rec)
            print(f"[sweep] {scen:18s} SNR={lvl:+.0f}  " +
                  "  ".join(f"{m}={rec[m]:.2f}" for m in METHODS) + f"  CRLB={rec['crlb_deg']:.3f}")

    csv_path = args.output_dir / "scenario_sweep.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")
    if args.dump_errors is not None:
        args.dump_errors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump_errors, **dump); print(f"wrote {args.dump_errors}")

    # ---- 2x2 grid plot ----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for ax, (scen, label) in zip(axes.ravel(), SCENARIOS):
        sub = [r for r in rows if r["scenario"] == scen]
        sub.sort(key=lambda r: r["snr_db"])
        xs = [r["snr_db"] for r in sub]
        for m in METHODS:
            ax.plot(xs, [r[m] for r in sub], marker="o", ms=4, lw=1.8,
                    color=COLORS[m], label=m)
        ax.plot(xs, [r["crlb_deg"] for r in sub], "k--", lw=1.2, label="CRLB")
        ax.set_title(label, fontsize=11)
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("RMSPE (deg)")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(METHODS) + 1, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("RMSE vs SNR across paper §IV scenarios — ReconUNet vs learned baselines vs classical",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    png = args.output_dir / "scenario_sweep_grid.png"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
