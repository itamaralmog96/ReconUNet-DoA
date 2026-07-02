#!/usr/bin/env python3
"""Evaluate the generalised (variable-K) SubspaceNet on the shared variable-K +
multipath corpus, broken down by true source count K.

For each K in the test set it reports RMSPE (median and mean over the direct-path
angles) for:
  * classical Root-MUSIC / ESPRIT on the raw empirical covariance, and
  * the trained SubspaceNet + Root-MUSIC (per-sample-K head).

This is the apples-to-apples baseline view that mirrors the ReconUNet setup:
one model, all source counts, multipath present (the coherent replicas are what
degrade the classical subspace estimate).

Usage::

    python scripts/analysis/eval_subnet_genk.py \
        --checkpoint experiments/runs/subspacenet_genk/checkpoints/best.pt \
        --manifest   data/scenes/subnet_genk/test.npy \
        --output-dir experiments/runs/subspacenet_genk/
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


def _wrapped_err_deg(pred_rad, true_rad):
    p = np.sort(pred_rad, axis=-1)
    t = np.sort(true_rad, axis=-1)
    d = (p - t + np.pi / 2) % np.pi - np.pi / 2
    return np.rad2deg(np.sqrt(np.mean(d ** 2, axis=-1)))      # per-sample [g]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "-c", required=True, type=Path)
    ap.add_argument("--manifest", "-m", required=True, type=Path)
    ap.add_argument("--output-dir", "-o", required=True, type=Path)
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--max-per-k", type=int, default=3000, help="cap samples per K bucket")
    ap.add_argument("--reconunet-checkpoint", type=Path, default=None,
                    help="if set, also evaluate ReconUNet (EVDUNet) on the same scenes")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = SceneManifest.load(str(args.manifest))
    ds = SceneDataset(manifest)
    n_src_all = manifest.raw["n_sources"].astype(int)

    adapter = SubspaceNetAdapter(M=4, tau=args.tau, diff_method="root_music")
    model = adapter.build_model({"M": 4, "tau": args.tau, "diff_method": "root_music"})
    adapter.load_checkpoint(model, str(args.checkpoint))
    model.eval()

    # Optional: ReconUNet (native EVDUNet) on the SAME scenes for a head-to-head.
    rn_adapter = rn_model = None
    if args.reconunet_checkpoint is not None:
        from reconunet.cli.evaluate import _NativeEVDUNetAdapter
        rn_adapter = _NativeEVDUNetAdapter(
            "reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet")
        rn_model = rn_adapter.build_model(
            {"M": 8, "tau": args.tau, "activation_type": "anti_rectifier", "use_dropout": True})
        rn_adapter.load_checkpoint(rn_model, str(args.reconunet_checkpoint))
        rn_model.eval()

    rows = []
    for K in sorted(np.unique(n_src_all)):
        if K < 1:
            continue
        idx = np.where(n_src_all == K)[0][: args.max_per_k]
        if idx.size == 0:
            continue
        snaps = torch.stack([ds[int(i)].snapshots for i in idx], dim=0)        # [g,M,T]
        true = np.stack([ds[int(i)].angles_rad.numpy()[:K] for i in idx])      # [g,K]
        R = (snaps @ snaps.conj().transpose(-1, -2)) / snaps.shape[-1]
        with torch.no_grad():
            rm = np.deg2rad(root_music(R, int(K), len(idx))[0].numpy() - 90.0)[:, :K]
            es = esprit(R, int(K), len(idx)).numpy()[:, :K]
            ns = torch.full((len(idx),), int(K))
            sn = adapter.forward(model, lag_stack(snaps, tau=args.tau),
                                 meta={"tau": args.tau, "n_sources": ns}).angles_pred.numpy()[:, :K]
            rn = None
            if rn_adapter is not None:
                rn = rn_adapter.forward(rn_model, lag_stack(snaps, tau=args.tau),
                                        meta={"tau": args.tau, "n_sources": ns}).angles_pred.numpy()[:, :K]
        e_rm, e_es, e_sn = (_wrapped_err_deg(x, true) for x in (rm, es, sn))
        row = {
            "K": int(K), "n": int(idx.size),
            "rm_med": float(np.median(e_rm)), "rm_mean": float(e_rm.mean()),
            "es_med": float(np.median(e_es)), "es_mean": float(e_es.mean()),
            "sn_med": float(np.median(e_sn)), "sn_mean": float(e_sn.mean()),
            "sn_p90": float(np.percentile(e_sn, 90)),
        }
        if rn is not None:
            e_rn = _wrapped_err_deg(rn, true)
            row["rn_med"] = float(np.median(e_rn)); row["rn_mean"] = float(e_rn.mean())
            row["rn_p90"] = float(np.percentile(e_rn, 90))
        rows.append(row)

    has_rn = "rn_med" in rows[0]
    print(f"\nVariable-K: SubspaceNet vs ReconUNet vs classical (multipath corpus) | RMSPE deg: median (mean)")
    hdr = f"{'K':>2} {'n':>5} | {'R-MUSIC':>15} {'ESPRIT':>15} {'SubNet+RM':>15}"
    if has_rn:
        hdr += f" {'ReconUNet':>15}"
    print(hdr)
    for r in rows:
        line = (f"{r['K']:>2} {r['n']:>5} | "
                f"{r['rm_med']:>6.2f} ({r['rm_mean']:>5.2f}) "
                f"{r['es_med']:>6.2f} ({r['es_mean']:>5.2f}) "
                f"{r['sn_med']:>6.2f} ({r['sn_mean']:>5.2f})")
        if has_rn:
            line += f" {r['rn_med']:>6.2f} ({r['rn_mean']:>5.2f})"
        print(line)

    import csv
    csv_path = args.output_dir / "genk_by_K.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
