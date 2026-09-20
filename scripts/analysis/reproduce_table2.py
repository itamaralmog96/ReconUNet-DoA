#!/usr/bin/env python3
"""Faithful reproduction of Almog & Weiss Table II + broader 3-way comparison.

For each §IV scenario and SNR level, computes **pooled RMSE** (deg) =
sqrt(mean of all squared per-source wrapped errors) — the paper's metric — for:

  classical on raw SCM:   Bartlett, MVDR, MUSIC, Root-MUSIC, ESPRIT, Unitary-ESPRIT
  ReconUNet-aided (R-hat): ReconUNet+{Root-MUSIC, MUSIC, ESPRIT, Unitary-ESPRIT}
  deep-learning baselines: SubspaceNet, SubViT
  + CRLB floor

Writes a long-form CSV and prints the 0-dB Table-II view.

    python scripts/analysis/reproduce_table2.py
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch

from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_dataset import SceneDataset
from reconunet.data.scene_renderer import lag_stack
from reconunet.evaluation import classical_batched as CB
from reconunet.evaluation.unified_harness import stochastic_crlb_deg
from reconunet.models.third_party.subspacenet_adapter import SubspaceNetAdapter
from reconunet.models.third_party.subvit_adapter import SubViTAdapter
from reconunet.models.third_party.damusic_adapter import DAMUSICEnsemble
from reconunet.cli.evaluate import _NativeEVDUNetAdapter

REPO = Path(__file__).resolve().parents[2]
SCEN = [("basic","Basic",1),("moderate","Moderate",2),
        ("advanced1_ood","OOD",1),("advanced2_crowded","Crowded",4)]
CLASSICAL = ["Bartlett","MVDR","MUSIC","Root-MUSIC","ESPRIT","Unitary-ESPRIT"]
RECON_BACKENDS = ["Root-MUSIC","MUSIC","ESPRIT","Unitary-ESPRIT"]


def sq_errs(pred, true):
    """Per-source squared errors (deg²) per paper eq. (31): sorted pairing =
    optimal permutation for scalars; NO angular wrap (sinθ injective on
    [-90°,90°] in the broadside convention — a former mod-π wrap shrank
    gross errors by up to 5×)."""
    p = np.sort(pred, -1); t = np.sort(true, -1)
    return (np.rad2deg(p - t)**2).ravel()                   # flattened per-source sq err (deg^2)


def _chunks(s, n):
    for i in range(0, len(s), n): yield s[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--output-dir", "-o", type=Path, default=REPO/"experiments/runs/table2")
    ap.add_argument("--scenarios-root", type=Path, default=REPO/"data/scenes/scenarios")
    ap.add_argument("--damusic-dir", type=Path, default=REPO/"experiments/runs/damusic_paper",
                    help="root holding k<K>/checkpoints/best.pt per source count (skipped if absent)")
    ap.add_argument("--dump-errors", type=Path, default=None,
                    help="also save per-scene per-source squared errors (deg^2) to this .npz for bootstrap CIs")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t2] device={dev}")

    sn_ad = SubspaceNetAdapter(M=4, tau=args.tau, diff_method="root_music")
    sn = sn_ad.build_model({"M":4,"tau":args.tau,"diff_method":"root_music"})
    sn_ad.load_checkpoint(sn, str(REPO/"experiments/runs/subspacenet_paper/checkpoints/best.pt")); sn.to(dev).eval()
    # Self-configure from the checkpoint's saved training config (avoids stale
    # hard-coded architecture; SubViT moved 256-dim -> published 768-dim).
    svi = dict(torch.load(REPO/"experiments/runs/subvit_paper/checkpoints/best.pt",
                          map_location="cpu", weights_only=False)["cfg"]["model"]["init"])
    sv_ad = SubViTAdapter(**svi); sv = sv_ad.build_model(svi)
    sv_ad.load_checkpoint(sv, str(REPO/"experiments/runs/subvit_paper/checkpoints/best.pt")); sv.to(dev).eval()
    rn_ad = _NativeEVDUNetAdapter("reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet")
    rn = rn_ad.build_model({"M":8,"tau":args.tau,"activation_type":"anti_rectifier","use_dropout":True})
    rn_ad.load_checkpoint(rn, str(REPO/"experiments/runs/reconunet_paper/checkpoints/best.pt")); rn.to(dev).eval()
    dm = DAMUSICEnsemble.from_run_dir(args.damusic_dir, device=dev)      # per-K models or None

    rows = []; dump = {}
    for scen, label, K in SCEN:
        man = SceneManifest.load(str(args.scenarios_root/f"{scen}/test.npy"))
        ds = SceneDataset(man); meta = man.meta; M = int(meta.M); T = int(meta.T)
        snr_all = man.raw["snr_db"]
        for lvl in sorted(np.unique(np.round(snr_all))):
            idx = np.where(np.abs(snr_all - lvl) < 1.0)[0]
            if idx.size == 0: continue
            acc = {}; crlb_sum = 0.0; crlb_n = 0; per_scene = {}
            def add(method, e):
                acc.setdefault(method, [0.0, 0]); acc[method][0] += e.sum(); acc[method][1] += e.size
                if args.dump_errors is not None:
                    per_scene.setdefault(method, []).append(e.reshape(-1, K).astype(np.float32))
            for chunk in _chunks(idx.tolist(), args.batch):
                snaps = torch.stack([ds[int(i)].snapshots for i in chunk], 0)
                true = np.stack([ds[int(i)].angles_rad.numpy()[:K] for i in chunk])
                g = len(chunk); nsrc = torch.full((g,), K)
                R_raw = ((snaps @ snaps.conj().transpose(-1,-2)) / snaps.shape[-1]).to(dev)
                lag = lag_stack(snaps, tau=args.tau).to(dev)
                with torch.no_grad():
                    _, _, R_hat = rn(lag)                                   # reconstructed covariance
                    R_hat = R_hat.detach()
                    for name in CLASSICAL:
                        fn = CB.ESTIMATORS[name]
                        add(name, sq_errs(fn(R_raw, K, M).cpu().numpy(), true))
                    for name in RECON_BACKENDS:
                        fn = CB.ESTIMATORS[name]
                        add(f"ReconUNet+{name}", sq_errs(fn(R_hat, K, M).cpu().numpy(), true))
                    add("SubspaceNet", sq_errs(sn_ad.forward(sn, lag, meta={"tau":args.tau,"n_sources":nsrc}).angles_pred.cpu().numpy()[:, :K], true))
                    sv_in = sv_ad.prepare_input(snaps, {"M":M}).to(dev)
                    add("SubViT", sq_errs(sv_ad.forward(sv, sv_in, meta={"n_sources":nsrc}).angles_pred.cpu().numpy()[:, :K], true))
                    if dm is not None:
                        add("DA-MUSIC", sq_errs(dm.predict(snaps.to(dev), nsrc).cpu().numpy()[:, :K], true))
                for r in true:
                    crlb_sum += stochastic_crlb_deg(M=M, T=T, angles_rad=r[:K], snr_db=float(lvl),
                                                    element_spacing_lambda=meta.element_spacing_lambda)
                    crlb_n += 1
            for method, (ssum, n) in acc.items():
                rows.append({"scenario": scen, "snr_db": float(lvl), "method": method,
                             "rmse_deg": float(np.sqrt(ssum / n)), "n": int(idx.size)})
            rows.append({"scenario": scen, "snr_db": float(lvl), "method": "CRLB",
                         "rmse_deg": float(np.sqrt(crlb_sum / crlb_n)), "n": int(idx.size)})
            for method, chunks_ in per_scene.items():
                dump[f"{scen}|{float(lvl):g}|{method}"] = np.concatenate(chunks_, 0)        # [n, K] deg^2
            print(f"[t2] {scen:18s} SNR={lvl:+.0f} done")

    csv_path = args.output_dir / "table2_full.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario","snr_db","method","rmse_deg","n"]); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")
    if args.dump_errors is not None:
        args.dump_errors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump_errors, **dump); print(f"wrote {args.dump_errors}")

    # ---- 0 dB Table-II view ----------------------------------------------
    order = (CLASSICAL + [f"ReconUNet+{b}" for b in RECON_BACKENDS] + ["SubspaceNet","SubViT"]
             + (["DA-MUSIC"] if dm is not None else []) + ["CRLB"])
    def get(scen, method):
        r = [x for x in rows if x["scenario"]==scen and x["method"]==method and abs(x["snr_db"])<0.5]
        return r[0]["rmse_deg"] if r else float("nan")
    print("\nTABLE II reproduction — pooled RMSE at 0 dB (deg)")
    hdr = f"{'method':24s}" + "".join(f"{lbl:>10s}" for _,lbl,_ in SCEN); print(hdr); print("-"*len(hdr))
    for m in order:
        print(f"{m:24s}" + "".join(f"{get(s,m):>10.2f}" for s,_,_ in SCEN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
