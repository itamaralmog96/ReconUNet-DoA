#!/usr/bin/env python3
"""Head-to-head comparison on the paper test split (data/scenes/paper/test.npy).

Evaluates, per true source count K and pooled over all K, the direct-path RMSPE
(median per-sample RMSPE + pooled RMSE per paper eq. 31, degrees) of five
methods on the SAME scenes:

  * classical Root-MUSIC      (subspace, on the empirical SCM)
  * classical ESPRIT          (subspace, on the empirical SCM)
  * SubspaceNet + Root-MUSIC  (per-sample-K differentiable head)
  * SubViT (DOA-ViT)          (grid spatial-spectrum, per-sample-K peak-pick)
  * ReconUNet (EVD-UNet)      (covariance reconstruction + Root-MUSIC)

All DL methods see the identical paired scenes; classical methods operate on the
raw empirical covariance of the same snapshots.  Angle conventions match the
trainer / eval harness (broadside radians, NaN-aware periodic RMSPE).

Usage::

    python scripts/analysis/compare_paper_testset.py            # all defaults
    python scripts/analysis/compare_paper_testset.py --max-per-k 5000
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
from reconunet.models.deep_learning.subspace_models import esprit, root_music
from reconunet.models.third_party.subspacenet_adapter import SubspaceNetAdapter
from reconunet.models.third_party.subvit_adapter import SubViTAdapter

REPO = Path(__file__).resolve().parents[2]


def _sq_err_deg2(pred_rad, true_rad):
    """Per-source squared errors in deg², shape [g, K]; inputs already sliced.

    Sorted pairing = optimal permutation for scalars (paper eq. 31); no
    angular wrap (broadside sinθ is injective on [-90°,90°]).  Callers derive
    the median column as median(sqrt(row-mean)) and the paper's pooled RMSE
    as sqrt(mean over all entries).
    """
    p = np.sort(pred_rad, axis=-1)
    t = np.sort(true_rad, axis=-1)
    return np.rad2deg(p - t) ** 2                                  # [g, K]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", "-m", type=Path,
                    default=REPO / "data/scenes/paper/test.npy")
    ap.add_argument("--subspacenet", type=Path,
                    default=REPO / "experiments/runs/subspacenet_paper/checkpoints/best.pt")
    ap.add_argument("--subvit", type=Path,
                    default=REPO / "experiments/runs/subvit_paper/checkpoints/best.pt")
    ap.add_argument("--reconunet", type=Path,
                    default=REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt")
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--max-per-k", type=int, default=3000, help="cap samples per K bucket")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--output-dir", "-o", type=Path,
                    default=REPO / "experiments/runs/eval_paper_3way")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cmp] device={dev}  manifest={args.manifest.relative_to(REPO)}  max_per_k={args.max_per_k}")

    manifest = SceneManifest.load(str(args.manifest))
    ds = SceneDataset(manifest)
    n_src_all = manifest.raw["n_sources"].astype(int)
    M = int(manifest.meta.M)

    # ---- build + load the three trained models ----------------------------
    sn_adapter = SubspaceNetAdapter(M=4, tau=args.tau, diff_method="root_music")
    sn_model = sn_adapter.build_model({"M": 4, "tau": args.tau, "diff_method": "root_music"})
    sn_adapter.load_checkpoint(sn_model, str(args.subspacenet)); sn_model.to(dev).eval()

    sv_init = dict(M=M, K_max=4, grid_size=121, angle_range_deg=(-60.0, 60.0),
                   embed_dim=256, depth=6, num_heads=8)
    sv_adapter = SubViTAdapter(**sv_init)
    sv_model = sv_adapter.build_model(sv_init)
    sv_adapter.load_checkpoint(sv_model, str(args.subvit)); sv_model.to(dev).eval()

    from reconunet.cli.evaluate import _NativeEVDUNetAdapter
    rn_adapter = _NativeEVDUNetAdapter(
        "reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet")
    rn_model = rn_adapter.build_model(
        {"M": M, "tau": args.tau, "activation_type": "anti_rectifier", "use_dropout": True})
    rn_adapter.load_checkpoint(rn_model, str(args.reconunet)); rn_model.to(dev).eval()

    METHODS = ["R-MUSIC", "ESPRIT", "SubspaceNet", "SubViT", "ReconUNet"]
    pooled = {m: [] for m in METHODS}     # per-sample errors across all K
    rows = []

    for K in sorted(np.unique(n_src_all)):
        if K < 1:
            continue
        idx = np.where(n_src_all == K)[0][: args.max_per_k]
        if idx.size == 0:
            continue
        errs = {m: [] for m in METHODS}
        for chunk in _chunks(idx.tolist(), args.batch):
            snaps = torch.stack([ds[int(i)].snapshots for i in chunk], dim=0)      # [g,M,T] complex
            true = np.stack([ds[int(i)].angles_rad.numpy()[:K] for i in chunk])    # [g,K]
            g = len(chunk)
            ns = torch.full((g,), int(K))
            R = (snaps @ snaps.conj().transpose(-1, -2)) / snaps.shape[-1]          # [g,M,M]
            lag = lag_stack(snaps, tau=args.tau).to(dev)
            with torch.no_grad():
                rm = np.deg2rad(root_music(R, int(K), g)[0].numpy() - 90.0)[:, :K]
                es = esprit(R, int(K), g).numpy()[:, :K]
                sn = sn_adapter.forward(sn_model, lag,
                                        meta={"tau": args.tau, "n_sources": ns}).angles_pred.cpu().numpy()[:, :K]
                rn = rn_adapter.forward(rn_model, lag,
                                        meta={"tau": args.tau, "n_sources": ns}).angles_pred.cpu().numpy()[:, :K]
                sv_in = sv_adapter.prepare_input(snaps, {"M": M}).to(dev)
                sv = sv_adapter.forward(sv_model, sv_in,
                                        meta={"n_sources": ns}).angles_pred.cpu().numpy()[:, :K]
            for name, pred in [("R-MUSIC", rm), ("ESPRIT", es), ("SubspaceNet", sn),
                               ("SubViT", sv), ("ReconUNet", rn)]:
                errs[name].append(_sq_err_deg2(pred, true))
        row = {"K": int(K), "n": int(idx.size)}
        for m in METHODS:
            e = np.concatenate(errs[m], axis=0); pooled[m].append(e)
            # median of per-sample RMSPE (robust) + the paper's pooled RMSE
            row[f"{m}_med"] = float(np.median(np.sqrt(e.mean(axis=-1))))
            row[f"{m}_mean"] = float(np.sqrt(e.mean()))
        rows.append(row)
        print(f"[cmp] K={K} done (n={idx.size})")

    # ---- overall (pooled over all K) --------------------------------------
    # Buckets have different K, so flatten per bucket: per-sample RMSPE rows
    # for the median, raw per-source squared errors for the pooled RMSE.
    overall = {"K": "all", "n": sum(r["n"] for r in rows)}
    for m in METHODS:
        rmspe_all = np.concatenate([np.sqrt(e.mean(axis=-1)) for e in pooled[m]])
        sq_all = np.concatenate([e.ravel() for e in pooled[m]])
        overall[f"{m}_med"] = float(np.median(rmspe_all))
        overall[f"{m}_mean"] = float(np.sqrt(sq_all.mean()))
    rows.append(overall)

    # ---- print table ------------------------------------------------------
    print(f"\nPaper test set — deg: median-RMSPE (pooled RMSE, paper eq 31), direct-path angles, true per-sample K")
    hdr = f"{'K':>3} {'n':>5} | " + " ".join(f"{m:>16}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        line = f"{str(r['K']):>3} {r['n']:>5} | " + " ".join(
            f"{r[f'{m}_med']:>6.2f} ({r[f'{m}_mean']:>6.2f})" for m in METHODS)
        print(line)

    csv_path = args.output_dir / "paper_testset_by_K.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
