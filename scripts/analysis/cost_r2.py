#!/usr/bin/env python3
"""R2 cost numbers for ReconUNet-C next to the full ReconUNet, measured exactly as the
cost table of docs/revision_facts.md (item 8, ``revision_facts.py generate``): same helpers
(``count_macs`` forward hooks on Conv/Linear, ``timed`` = median of GPU-synchronised repeats
after warm-up), same inputs (the first 1024 K = 1 scenes of the paper test split), same
pipelines (network forward only; network + Root-MUSIC with CPU eigh/roots), plus the
CPU forward at batch 1.  Both models are measured in the same process for a like-for-like
comparison.

Note: in eval mode ``CovarianceOnlyReconstructionUNet.forward`` also runs an fp64 CPU
``eigh(R_hat)`` to return the same (eigvals, eigvecs, R_hat) triple as the full model; the
Root-MUSIC back end does not use those eigenpairs, so the U-Net-only time (``base_unet``) is
reported as well.

Outputs (--output-dir): cost_latency_r2.csv (long format, columns as the facts table),
cost_summary_r2.csv (one row per model: params, MMACs, GPU b1 / b1024, CPU b1, training time).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, load_reconunet  # noqa: E402
from revision_facts import count_macs, params_of, status_elapsed, timed  # noqa: E402

from reconunet.data.scene_dataset import SceneDataset  # noqa: E402
from reconunet.data.scene_manifest import SceneManifest  # noqa: E402
from reconunet.data.scene_renderer import lag_stack  # noqa: E402
from reconunet.models.deep_learning.subspace_models import root_music  # noqa: E402


def train_info(ck_dir: Path, status_log: Path, stage: str) -> dict:
    h = ck_dir / "history.json"
    out = {"epochs": "", "best_epoch": "", "train_wallclock_min": ""}
    if h.exists():
        rows = json.load(h.open()); best = min(rows, key=lambda r: r["val"])
        out.update(epochs=len(rows), best_epoch=best["epoch"])
    el = status_elapsed(status_log).get(stage, "")
    out["train_wallclock_min"] = el.replace("min", "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconunet", type=Path, default=REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt")
    ap.add_argument("--reconunet-c", type=Path, required=True)
    ap.add_argument("--status-log", type=Path, required=True, help="R2 pipeline status.log (training wall-clock)")
    ap.add_argument("--reconunet-cb", type=Path, default=None, help="optional R2b ReconUNet-CB checkpoint (third row)")
    ap.add_argument("--status-log-cb", type=Path, default=None, help="R2b pipeline status.log (ReconUNet-CB wall-clock)")
    ap.add_argument("--output-dir", "-o", type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = SceneManifest.load(str(REPO / "data/scenes/paper/test.npy")); ds = SceneDataset(man)
    idx = np.where(man.raw["n_sources"] == 1)[0][:1024]
    X = torch.stack([ds[int(i)].snapshots for i in idx], 0)

    models = {"ReconUNet": a.reconunet, "ReconUNet-C": a.reconunet_c}
    if a.reconunet_cb is not None:
        models["ReconUNet-CB"] = a.reconunet_cb
    lat, summ = [], []
    for name, ck in models.items():
        _, net, _ = load_reconunet(ck, dev)
        cpu_net = load_reconunet(ck, torch.device("cpu"))[1]
        rec = {"model": name, "class": type(net).__name__, "trainable parameters": params_of(ck)}
        rec["MMACs per scene"] = count_macs(net, lambda: net(lag_stack(X[:1], tau=8).to(dev))) / 1e6
        for B in (1, 1024):
            xb = X[:B]; lag = lag_stack(xb, tau=8); lag_d = lag.to(dev)
            def fwd():
                with torch.no_grad(): return net(lag_d)
            def unet_only():
                with torch.no_grad(): return net.base_unet(lag_d)
            def full():
                with torch.no_grad(): _, _, Rh = net(lag_stack(xb, tau=8).to(dev)); return root_music(Rh.cpu(), 1, B)
            def cpu_fwd():
                with torch.no_grad(): return cpu_net(lag)
            cands = [(f"{name} forward only (GPU)", fwd, dev), (f"{name} U-Net only, base_unet (GPU)", unet_only, dev),
                     (f"{name} + Root-MUSIC (GPU net, CPU eigh/roots)", full, dev)]
            if B == 1:
                cands.append((f"{name} forward only (CPU, {torch.get_num_threads()} threads)", cpu_fwd, torch.device("cpu")))
            for label, fn, d in cands:
                if d.type == "cuda": torch.cuda.reset_peak_memory_stats()
                t = timed(fn, d, reps=5 if B == 1 else 3)
                mem = torch.cuda.max_memory_allocated() / 2**20 if d.type == "cuda" else float("nan")
                lat.append({"pipeline": label, "batch": B, "ms per scene": 1e3 * t / B, "ms per batch": 1e3 * t, "peak GPU MiB": mem})
                key = {f"{name} forward only (GPU)": "GPU fwd", f"{name} + Root-MUSIC (GPU net, CPU eigh/roots)": "GPU fwd+Root-MUSIC"}.get(label)
                if key: rec[f"{key} b{B} ms/scene"] = 1e3 * t / B
                if d.type == "cpu": rec["CPU fwd b1 ms"] = 1e3 * t
        if name == "ReconUNet":
            rec.update(train_info(REPO / "experiments/runs/reconunet_paper/checkpoints",
                                  REPO / "experiments/runs/revision_retrain_20260906/status.log", "reconunet"))
        elif name == "ReconUNet-CB":
            rec.update(train_info(Path(ck).parent, a.status_log_cb or a.status_log, "train_reconunet_cb"))
        else:
            rec.update(train_info(Path(ck).parent, a.status_log, "train_reconunet_c"))
        summ.append(rec)
        print(f"[cost] {name}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    for fname, rows in (("cost_latency_r2.csv", lat), ("cost_summary_r2.csv", summ)):
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with (a.output_dir / fname).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[cost] device={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}; wrote {a.output_dir}/cost_latency_r2.csv, cost_summary_r2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
