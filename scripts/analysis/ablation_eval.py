#!/usr/bin/env python3
"""Evaluate the ReconUNet ablation variants (experiments/runs/ablation_20260920).

For every ``<ablation-dir>/<variant>/checkpoints/best.pt`` the model is rebuilt
from the checkpoint's saved config (class_path, tau, activation, meta
overrides) and evaluated with the Root-MUSIC back end on

  * ``paper_test``            paper test split, first ``--max-per-k`` scenes of every K (default 3000 -> 12k)
  * ``moderate_0dB``          Section IV Moderate scenario (K=2 + 1 coherent replica), 0 dB rows
  * ``crowded_0dB``           Section IV Crowded scenario (K=4 + 3 coherent replicas), 0 dB rows
  * ``paper_test_fixed_imperf``  (01_full and 09_fixed_imperf only) the paper test split rendered
                              with ONE array-error realisation (seed 20260920 = 09's training draw)

Metric definitions (all per scene, then aggregated; K = true source count):

  pooled_rmse_deg      sqrt(mean over all scenes and sources of squared angle error), paper eq. (31)
  median_rmspe_deg     median over scenes of sqrt(mean_k squared error)
  cov_err_frob         mean ||R_hat - R*||_F / ||R*||_F   (R* = clean A A^H target of the loss)
  subspace_dist_projF  mean ||P_K(R_hat) - P_K(R*)||_F, P_K = V_K V_K^H from eigh (leading K)
  eigengap_err         mean |gap_hat - gap*| / gap*,  gap = lambda_K - lambda_{K+1} from eigh (gap* = lambda*_K)
  n_scenes             number of scenes in the eval set

Writes ``ablation_results.csv`` (exact column set above) and, for 01_full and
08_no_evd_heads, ``route_comparison.csv``:

  route = covariance  eigh(R_hat) -> noise subspace -> Root-MUSIC   (what every table in the paper uses)
  route = subspace    the model's OWN eigenvector output -> noise subspace -> Root-MUSIC (no eigh)
                      For 08 there are no heads, so its "subspace" row uses eigh(R_hat) and is
                      identical to the covariance route by construction (see the note column).
  latency_ms_per_scene  end-to-end (lag stack on device -> network -> angles) at the eval batch
                        size and at batch 1, GPU-synchronised, median of repeated timings
  orthogonality_residual  mean ||V^H V - I||_F of the eigenvector matrix fed to the route

Usage::

    python scripts/analysis/ablation_eval.py                       # all variants, default sets
    python scripts/analysis/ablation_eval.py --variants 01_full 08_no_evd_heads --max-per-k 500
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import time
from pathlib import Path

import numpy as np
import torch

from reconunet.data.scene_dataset import SceneDataset
from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_renderer import SceneRenderer, lag_stack
from reconunet.models.deep_learning.subspace_models import (
    find_roots_batched, root_music, sum_of_diags_batched)

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "experiments/runs/ablation_20260920"
FIXED_SEED = 20260920
ROUTE_VARIANTS = ("01_full", "08_no_evd_heads")
COLS = ["variant", "eval_set", "pooled_rmse_deg", "median_rmspe_deg", "cov_err_frob",
        "subspace_dist_projF", "eigengap_err", "n_scenes"]
ROUTE_COLS = ["variant", "eval_set", "route", "batch_size", "rmse_deg", "median_rmspe_deg",
              "latency_ms_per_scene", "orthogonality_residual", "n_scenes", "note"]


def _import(dotted):
    mod, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def load_variant(vdir: Path, dev):
    ck = torch.load(vdir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cls = _import(cfg["model"]["class_path"])
    init = dict(cfg["model"]["init"])
    model = cls(**init)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    overrides = dict((cfg.get("data") or {}).get("meta_overrides") or {})
    return model, int(init.get("tau", 8)), overrides, int(ck.get("epoch", -1))


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _angles_from_noise_projector(F: torch.Tensor, K: int) -> torch.Tensor:
    """Root-MUSIC from a batched noise projector F=[B,N,N] -> broadside radians [B,K]."""
    coeffs = sum_of_diags_batched(F)
    roots = find_roots_batched(coeffs)
    k_d = 0.5 * 2 * np.pi
    ang = torch.clamp(torch.angle(roots) / k_d, -1 + 1e-6, 1 - 1e-6)
    doa_all = torch.acos(-ang)                                   # radians, pi/2 = broadside
    mag = roots.abs()
    score = torch.where(mag < 1.0, (mag - 1.0).abs(), torch.full_like(mag, float("inf")))
    sel = torch.argsort(score, dim=-1)[:, :K]
    return torch.gather(doa_all, 1, sel) - np.pi / 2


def angles_covariance_route(R_hat: torch.Tensor, K: int) -> torch.Tensor:
    return torch.deg2rad(root_music(R_hat.cpu(), K, R_hat.shape[0])[0] - 90.0)


def angles_subspace_route(V_hat: torch.Tensor, K: int) -> torch.Tensor:
    Vn = V_hat[:, :, K:].cpu().to(torch.complex128)
    F = Vn @ Vn.conj().transpose(-1, -2)
    return _angles_from_noise_subspace_projector(F, K)


def _angles_from_noise_subspace_projector(F, K):
    return _angles_from_noise_projector(F.to(torch.complex64), K)


def sq_err_deg2(pred_rad: np.ndarray, true_rad: np.ndarray) -> np.ndarray:
    p = np.sort(pred_rad, axis=-1)[:, : true_rad.shape[1]]
    t = np.sort(true_rad, axis=-1)
    return np.rad2deg(p - t) ** 2


def eigh_desc(R: torch.Tensor):
    w, V = torch.linalg.eigh(R.detach().to("cpu", torch.complex128))
    return torch.flip(w, dims=[-1]), torch.flip(V, dims=[-1])


def structural_metrics(R_hat: torch.Tensor, R_clean: torch.Tensor, K: int):
    R_hat = R_hat.detach().to("cpu", torch.complex128); R_clean = R_clean.to(torch.complex128)
    cov_rel = (torch.linalg.norm(R_hat - R_clean, dim=(-2, -1)) /
               torch.linalg.norm(R_clean, dim=(-2, -1)).clamp_min(1e-12))
    w_h, V_h = eigh_desc(R_hat); w_c, V_c = eigh_desc(R_clean)
    Ph = V_h[:, :, :K] @ V_h[:, :, :K].conj().transpose(-1, -2)
    Pc = V_c[:, :, :K] @ V_c[:, :, :K].conj().transpose(-1, -2)
    proj = torch.linalg.norm(Ph - Pc, dim=(-2, -1))
    gap_h = w_h[:, K - 1] - (w_h[:, K] if K < w_h.shape[1] else 0.0)
    gap_c = w_c[:, K - 1] - (w_c[:, K] if K < w_c.shape[1] else 0.0)
    gap_err = (gap_h - gap_c).abs() / gap_c.abs().clamp_min(1e-9)
    return cov_rel.numpy(), proj.numpy(), gap_err.numpy()


def eval_sets(max_per_k: int, want_fixed: bool):
    """Yield (name, dataset, list of (K, indices))."""
    man = SceneManifest.load(str(REPO / "data/scenes/paper/test.npy"))
    ns = man.raw["n_sources"].astype(int)
    buckets = [(int(K), np.where(ns == K)[0][:max_per_k].tolist()) for K in sorted(np.unique(ns)) if K >= 1]
    yield "paper_test", SceneDataset(man), buckets
    for name, scen, K in (("moderate_0dB", "moderate", 2), ("crowded_0dB", "advanced2_crowded", 4)):
        m = SceneManifest.load(str(REPO / f"data/scenes/scenarios/{scen}/test.npy"))
        idx = np.where(np.abs(m.raw["snr_db"]) < 1.0)[0].tolist()
        yield name, SceneDataset(m), [(K, idx)]
    if want_fixed:
        meta_fix = dataclasses.replace(man.meta, fixed_imperfection_seed=FIXED_SEED)
        yield "paper_test_fixed_imperf", SceneDataset(man, renderer=SceneRenderer(meta_fix)), buckets


def timed(fn, dev, reps=3):
    ts = []
    for _ in range(reps):
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", type=Path, default=ABL)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--max-per-k", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--latency-scenes", type=int, default=256, help="scenes used for the batch-1 latency loop")
    ap.add_argument("--output-dir", "-o", type=Path, default=None)
    a = ap.parse_args()
    out = a.output_dir or a.ablation_dir
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = sorted(d.name for d in a.ablation_dir.iterdir()
                      if d.is_dir() and (d / "checkpoints" / "best.pt").exists())
    if a.variants:
        missing = [v for v in a.variants if v not in variants]
        if missing: print(f"[abl] WARNING no best.pt for: {missing}")
        variants = [v for v in a.variants if v in variants]
    print(f"[abl] device={dev} variants={variants}")

    rows, route_rows = [], []
    for v in variants:
        model, tau, overrides, epoch = load_variant(a.ablation_dir / v, dev)
        print(f"[abl] {v}: tau={tau} overrides={overrides} best-epoch={epoch}")
        want_fixed = v in ("01_full", "09_fixed_imperf")
        for set_name, ds, buckets in eval_sets(a.max_per_k, want_fixed):
            sq, per_scene, cov, proj, gap = [], [], [], [], []
            route_acc = {r: {"sq": [], "orth": [], "t": 0.0, "n": 0} for r in ("covariance", "subspace")}
            do_routes = v in ROUTE_VARIANTS and set_name in ("paper_test", "moderate_0dB")
            for K, idx in buckets:
                for chunk in _chunks(idx, a.batch):
                    samples = [ds[int(i)] for i in chunk]
                    snaps = torch.stack([s.snapshots for s in samples], 0)
                    R_clean = torch.stack([s.covariance_clean for s in samples], 0)
                    true = np.stack([s.angles_rad.numpy()[:K] for s in samples])
                    lag = lag_stack(snaps, tau=tau).to(dev)
                    with torch.no_grad():
                        w_hat, V_hat, R_hat = model(lag)
                        ang = angles_covariance_route(R_hat, K).numpy()
                    e = sq_err_deg2(ang, true); sq.append(e); per_scene.append(np.sqrt(e.mean(-1)))
                    c, p_, g = structural_metrics(R_hat, R_clean, K); cov.append(c); proj.append(p_); gap.append(g)
                    if do_routes:
                        def cov_route():
                            with torch.no_grad():
                                _, _, Rh = model(lag_stack(snaps, tau=tau).to(dev))
                                return angles_covariance_route(Rh, K)
                        def sub_route():
                            with torch.no_grad():
                                _, Vh, _ = model(lag_stack(snaps, tau=tau).to(dev))
                                return angles_subspace_route(Vh, K)
                        for rname, fn, Vsrc in (("covariance", cov_route, None), ("subspace", sub_route, V_hat)):
                            t = timed(fn, dev); pred = fn().numpy()
                            e_r = sq_err_deg2(pred, true)
                            acc = route_acc[rname]; acc["sq"].append(e_r.ravel()); acc.setdefault("ps", []).append(np.sqrt(e_r.mean(-1)))
                            acc["t"] += t; acc["n"] += len(chunk)
                            Vo = (eigh_desc(R_hat)[1] if Vsrc is None else Vsrc.detach().cpu().to(torch.complex128))
                            I = torch.eye(Vo.shape[-1], dtype=Vo.dtype)
                            acc["orth"].append(torch.linalg.norm(Vo.conj().transpose(-1, -2) @ Vo - I, dim=(-2, -1)).numpy())
            sq_all = np.concatenate([x.ravel() for x in sq]); ps = np.concatenate(per_scene)
            rows.append({"variant": v, "eval_set": set_name,
                         "pooled_rmse_deg": float(np.sqrt(sq_all.mean())),
                         "median_rmspe_deg": float(np.median(ps)),
                         "cov_err_frob": float(np.concatenate(cov).mean()),
                         "subspace_dist_projF": float(np.concatenate(proj).mean()),
                         "eigengap_err": float(np.concatenate(gap).mean()),
                         "n_scenes": int(ps.size)})
            print(f"[abl]   {set_name:24s} rmse={rows[-1]['pooled_rmse_deg']:.3f} med={rows[-1]['median_rmspe_deg']:.3f} "
                  f"cov={rows[-1]['cov_err_frob']:.3f} proj={rows[-1]['subspace_dist_projF']:.3f} gap={rows[-1]['eigengap_err']:.3f} n={ps.size}")
            if do_routes:
                note = ("no EVD heads: eigenvectors from eigh(R_hat); routes coincide by construction"
                        if v == "08_no_evd_heads" else "EVD-head eigenvectors used directly (no eigh)")
                for rname, acc in route_acc.items():
                    e = np.concatenate(acc["sq"]); orth = float(np.concatenate(acc["orth"]).mean())
                    route_rows.append({"variant": v, "eval_set": set_name, "route": rname, "batch_size": a.batch,
                                       "rmse_deg": float(np.sqrt(e.mean())), "median_rmspe_deg": float(np.median(np.concatenate(acc["ps"]))),
                                       "latency_ms_per_scene": 1e3 * acc["t"] / max(1, acc["n"]),
                                       "orthogonality_residual": orth, "n_scenes": int(acc["n"]),
                                       "note": note if rname == "subspace" else "eigh(R_hat) -> Root-MUSIC"})
                # batch-1 latency on the first scenes of the set
                K0, idx0 = buckets[0]; idx0 = idx0[: a.latency_scenes]
                one = [ds[int(i)] for i in idx0]
                for rname in ("covariance", "subspace"):
                    tot = 0.0
                    for s in one:
                        lag1 = lag_stack(s.snapshots[None], tau=tau).to(dev)
                        def f1():
                            with torch.no_grad():
                                _, Vh, Rh = model(lag1)
                                return angles_covariance_route(Rh, K0) if rname == "covariance" else angles_subspace_route(Vh, K0)
                        tot += timed(f1, dev, reps=1)
                    route_rows.append({"variant": v, "eval_set": set_name, "route": rname, "batch_size": 1,
                                       "rmse_deg": float("nan"), "median_rmspe_deg": float("nan"),
                                       "latency_ms_per_scene": 1e3 * tot / max(1, len(one)),
                                       "orthogonality_residual": float("nan"), "n_scenes": len(one),
                                       "note": "batch-1 latency only"})
        model = None
        if dev.type == "cuda": torch.cuda.empty_cache()

    with (out / "ablation_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"[abl] wrote {out / 'ablation_results.csv'} ({len(rows)} rows)")
    if route_rows:
        with (out / "route_comparison.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ROUTE_COLS); w.writeheader(); w.writerows(route_rows)
        print(f"[abl] wrote {out / 'route_comparison.csv'} ({len(route_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
