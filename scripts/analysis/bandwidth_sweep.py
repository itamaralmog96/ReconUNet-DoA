#!/usr/bin/env python3
"""Source-bandwidth sweep (Sensors revision round 2, reviewer: "the multi-lag input
depends on the source bandwidth").  Evaluation only — no model is retrained.

Everything is fixed except the source bandwidth ``source_bw_frac`` (fraction of fs):

  scenarios   Moderate  = the paper's Moderate test scenario (data/scenes/scenarios/moderate,
                          K = 2 direct + 1 coherent replica), 0 dB rows (1000 scenes)
              Crowded   = advanced2_crowded (K = 4 direct + 3 replicas), 0 dB rows (1000 scenes)
              Moderate-3+1 = seeded draw with K = 3 direct + 1 replica (1000 scenes, seed 20260930,
                          same generator/settings as the scenario manifests) — added because the
                          revision request describes Moderate as "3 direct + 1 replica" while the
                          paper's scenario manifest is 2 + 1
              all: mild imperfections, T = 512, 0 dB
  bandwidths  source_bw_frac in {0.01, 0.02, 0.05 (= training), 0.10, 0.20, 0.40} and white (filter off)

Every scene keeps its seed, so angles, replica AoAs, gains, imperfection draws, source
draws and noise are identical at every bandwidth (the renderer's rng consumption does not
depend on the bandwidth; band-limiting is a deterministic FFT mask + power renormalisation).

Replica-delay confound: the renderer draws replica delays in [0, 1/(fs·bw)] seconds, so by
default the delays scale with the bandwidth.  ``--mode decoupled`` (primary result) pins the
delay range to the training bandwidth 0.05 via ``ManifestMeta.mp_delay_bw_frac = 0.05`` so
every delay is exactly the one drawn at 0.05 and only the source spectrum changes;
``--mode coupled`` (secondary) uses the unmodified renderer (delays ∝ 1/bw; for white sources
the renderer always uses the 0.05 delay range, so the white row is identical in both modes).

Methods: raw Root-MUSIC, ReconUNet + Root-MUSIC, ReconUNet-C + Root-MUSIC (if a checkpoint is
given), SubspaceNet, DA-MUSIC (v2 ensemble, true-K dispatch), SubViT (zero-lag covariance only —
a control that does not see the lag stack).

Outputs (in --output-dir):
  bandwidth_sweep.csv / bandwidth_sweep_coupled.csv   pooled RMSE, median per-scene RMSPE and
      percentile-bootstrap 95 % CIs (1000 resamples over scenes, seed 20260920) per
      scenario × bandwidth × method
  bandwidth_errors[_coupled].npz                      per-scene per-source squared errors (deg²)
  bandwidth_autocorr.csv / bandwidth_autocorr_coupled.csv   measured normalised source
      autocorrelation |r(l)|, l = 0..7 (direct sources), measured direct/replica correlation
      coefficient |γ| and replica delays, per scenario × bandwidth
  --mode plot: quick-look RMSE-vs-bandwidth PDF/PNG (log x, one panel per scenario)

    python scripts/analysis/bandwidth_sweep.py --mode decoupled -o DIR --reconunet-c CK
    python scripts/analysis/bandwidth_sweep.py --mode coupled   -o DIR --reconunet-c CK
    python scripts/analysis/bandwidth_sweep.py --mode plot      -o DIR --fig-dir docs/figs_revision/r2
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _revision_common import REPO, CK, Models, chunks, sq_err_deg2  # noqa: E402
from bootstrap_ci import boot  # noqa: E402

from reconunet.data.scene_manifest import SceneManifest  # noqa: E402
from reconunet.data.scene_renderer import SceneRenderer  # noqa: E402

BWS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, None]          # None = white (filter off)
TRAIN_BW = 0.05
LAGS = 8
METHODS = ["Root-MUSIC", "ReconUNet", "ReconUNet-C", "ReconUNet-CB", "SubspaceNet", "DA-MUSIC", "SubViT"]
COLS = ["scenario", "K", "replicas", "delay_mode", "bw_frac", "method", "n_scenes", "rmse_deg", "rmse_ci_lo",
        "rmse_ci_hi", "median_rmspe_deg", "median_ci_lo", "median_ci_hi", "resamples"]


def bw_label(bw) -> str:
    return "white" if bw is None else f"{bw:g}"


def load_scenarios(snr: float, max_scenes: int, seed: int, names: list[str]):
    """-> list of (name, manifest, row indices).  Built from the scenario manifests (0 dB rows)
    or, for moderate3, a seeded draw with the scenario generator."""
    out = []
    for name in names:
        if name == "moderate3":
            meta = SceneManifest.load(str(REPO / "data/scenes/scenarios/moderate/test.npy")).meta
            man = SceneManifest.fixed_angles_snr_sweep(meta, n_angle_configs=max_scenes, snr_levels_db=[snr],
                                                      rng=np.random.default_rng(seed), k=3, min_separation_deg=10.0,
                                                      array_errors="mild", enable_multipath=True,
                                                      num_multipath_components=1)
            out.append(("moderate3", man, np.arange(len(man))[:max_scenes]))
        else:
            d = {"moderate": "moderate", "crowded": "advanced2_crowded"}[name]
            man = SceneManifest.load(str(REPO / f"data/scenes/scenarios/{d}/test.npy"))
            idx = np.where(np.abs(man.raw["snr_db"] - snr) < 0.5)[0][:max_scenes]
            out.append((name, man, idx))
    return out


def meta_for(meta, bw, mode: str):
    return dataclasses.replace(meta, source_bw_frac=bw, mp_delay_bw_frac=(TRAIN_BW if mode == "decoupled" else None))


def autocorr_stats(results, K: int) -> dict:
    """Normalised |r(l)| of the direct sources (non-circular estimate), l = 0..LAGS-1, and |γ|."""
    r = np.zeros(LAGS); n = 0; gam = []; dly = []
    for res in results:
        s = res.source_signals.astype(np.complex128); T = s.shape[1]
        for k in range(K):
            x = s[k]; p = np.mean(np.abs(x) ** 2)
            r += np.array([abs(np.vdot(x[:T - l], x[l:])) / (T - l) / p for l in range(LAGS)]); n += 1
        for j in range(K, s.shape[0]):
            gam.append(abs(np.vdot(s[0], s[j])) / (np.linalg.norm(s[0]) * np.linalg.norm(s[j])))
        dly.extend(res.mp_delay_samples.tolist())
    row = {f"r{l}": r[l] / max(n, 1) for l in range(LAGS)}
    row.update(gamma_mean=float(np.mean(gam)) if gam else float("nan"),
               gamma_median=float(np.median(gam)) if gam else float("nan"),
               gamma_min=float(np.min(gam)) if gam else float("nan"),
               delay_mean_samples=float(np.mean(dly)) if dly else float("nan"),
               delay_max_samples=float(np.max(dly)) if dly else float("nan"))
    return row


def run(a, mode: str) -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _ck(path, name):
        if path and not Path(path).exists():
            print(f"[bw] WARNING: {name} checkpoint {path} not found — method skipped"); return None
        return path or None
    mdl = Models(dev, reconunet=a.reconunet, damusic_dir=a.damusic_dir, with_damusic="DA-MUSIC" in a.methods,
                 reconunet_c=_ck(a.reconunet_c, "ReconUNet-C"), reconunet_cb=_ck(a.reconunet_cb, "ReconUNet-CB"))
    sv = None
    if "SubViT" in a.methods:
        from reconunet.models.third_party.subvit_adapter import SubViTAdapter
        svi = dict(torch.load(a.subvit, map_location="cpu", weights_only=False)["cfg"]["model"]["init"])
        sv_ad = SubViTAdapter(**svi); sv = sv_ad.build_model(svi); sv_ad.load_checkpoint(sv, str(a.subvit)); sv.to(dev).eval()

    rng = np.random.default_rng(a.ci_seed)
    rows, ac_rows, dump = [], [], {}
    for scen, man, idx in load_scenarios(a.snr, a.max_scenes, a.draw_seed, a.scenarios):
        K = int(man.raw["n_sources"][idx[0]]); R_ = int(man.raw["num_multipath"][idx[0]])
        assert (man.raw["n_sources"][idx] == K).all() and (man.raw["num_multipath"][idx] == R_).all()
        M = int(man.meta.M)
        for bw in BWS:
            rend = SceneRenderer(meta_for(man.meta, bw, mode))
            results = [rend.render(man[int(i)]) for i in idx]
            X = torch.from_numpy(np.stack([r.snapshots for r in results]))
            true = np.stack([r.angles_rad[:K] for r in results])
            ac_rows.append({"scenario": scen, "K": K, "replicas": R_, "delay_mode": mode, "bw_frac": bw_label(bw),
                            "n_scenes": len(results), **autocorr_stats(results, K)})
            errs = {m: [] for m in METHODS}
            on = set(a.methods)
            for ch in chunks(list(range(X.shape[0])), a.batch):
                xb = X[ch]; tb = true[ch]; g = len(ch); ns = torch.full((g,), K)
                if "Root-MUSIC" in on: errs["Root-MUSIC"].append(sq_err_deg2(Models.root_music(xb, K), tb))
                if "ReconUNet" in on: errs["ReconUNet"].append(sq_err_deg2(mdl.reconunet(xb, K), tb))
                for name in mdl.extra:
                    if name in on: errs[name].append(sq_err_deg2(mdl.extra_pred(name, xb, K), tb))
                if "SubspaceNet" in on: errs["SubspaceNet"].append(sq_err_deg2(mdl.subspacenet(xb, K), tb))
                d = mdl.damusic(xb, K) if "DA-MUSIC" in on else None
                if d is not None: errs["DA-MUSIC"].append(sq_err_deg2(d, tb))
                if sv is not None:
                    with torch.no_grad():
                        p = sv_ad.forward(sv, sv_ad.prepare_input(xb, {"M": M}).to(dev), meta={"n_sources": ns})
                    errs["SubViT"].append(sq_err_deg2(p.angles_pred.cpu().numpy()[:, :K], tb))
            line = []
            for m in METHODS:
                if not errs[m]: continue
                E = np.concatenate(errs[m]); dump[f"{scen}|{bw_label(bw)}|{m}"] = E.astype(np.float32)
                b = boot(E, a.resamples, rng)
                rows.append({"scenario": scen, "K": K, "replicas": R_, "delay_mode": mode, "bw_frac": bw_label(bw),
                             "method": m, "resamples": a.resamples, **b})
                line.append(f"{m}={b['rmse_deg']:.2f}")
            print(f"[bw:{mode}] {scen:10s} bw={bw_label(bw):5s} |γ|={ac_rows[-1]['gamma_mean']:.3f} "
                  f"r1={ac_rows[-1]['r1']:.3f}  " + "  ".join(line), flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    suf = ("" if mode == "decoupled" else "_coupled") + a.tag
    with (a.output_dir / f"bandwidth_sweep{suf}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    with (a.output_dir / f"bandwidth_autocorr{suf}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ac_rows[0].keys())); w.writeheader(); w.writerows(ac_rows)
    np.savez_compressed(a.output_dir / f"bandwidth_errors{suf}.npz", **dump)
    print(f"[bw:{mode}] wrote {a.output_dir}/bandwidth_sweep{suf}.csv, bandwidth_autocorr{suf}.csv, bandwidth_errors{suf}.npz")
    return 0


def plot(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams.update({"font.size": 10, "legend.fontsize": 8.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    df = pd.read_csv(a.output_dir / f"bandwidth_sweep{a.tag}.csv", dtype={"bw_frac": str})
    cp = a.output_dir / f"bandwidth_sweep_coupled{a.tag}.csv"
    dc = pd.read_csv(cp, dtype={"bw_frac": str}) if cp.exists() else None
    WHITE_X = 1.0
    xval = lambda s: WHITE_X if s == "white" else float(s)
    colors = {"Root-MUSIC": "#4a4a4a", "ReconUNet": "#1f4e79", "ReconUNet-C": "#d35400", "ReconUNet-CB": "#c0392b", "SubspaceNet": "#2e86c1",
              "DA-MUSIC": "#8e44ad", "SubViT": "#27ae60"}
    titles = {"moderate": "Moderate (K = 2 + 1 replica)", "crowded": "Crowded (K = 4 + 3 replicas)",
              "moderate3": "Moderate-3+1 (K = 3 + 1 replica)"}
    scens = [s for s in ("moderate", "moderate3", "crowded") if s in set(df.scenario)]
    fig, axes = plt.subplots(1, len(scens), figsize=(4.2 * len(scens), 3.6), squeeze=False)
    for ax, s in zip(axes[0], scens):
        for m in METHODS:
            sub = df[(df.scenario == s) & (df.method == m)]
            if sub.empty or m not in a.methods: continue
            x = sub.bw_frac.map(xval).values; o = np.argsort(x)
            y = sub.rmse_deg.values[o]; lo = y - sub.rmse_ci_lo.values[o]; hi = sub.rmse_ci_hi.values[o] - y
            ax.errorbar(x[o], y, yerr=[lo, hi], marker="o", ms=3.5, lw=1.5, capsize=2, color=colors[m], label=m)
            if dc is not None:
                sc = dc[(dc.scenario == s) & (dc.method == m)]
                if not sc.empty:
                    xc = sc.bw_frac.map(xval).values; oc = np.argsort(xc)
                    ax.plot(xc[oc], sc.rmse_deg.values[oc], ls="--", lw=0.9, color=colors[m], alpha=0.7)
        ax.axvline(TRAIN_BW, color="#999999", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, WHITE_X]
        ax.set_xticks(ticks); ax.set_xticklabels(["0.01", "0.02", "0.05\n(train)", "0.1", "0.2", "0.4", "white"])
        ax.set_xlabel("source bandwidth (fraction of fs)"); ax.set_ylabel("pooled RMSE (deg)")
        ax.set_title(titles.get(s, s)); ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(frameon=False, loc="best")
    fig.suptitle("Source-bandwidth sweep, 0 dB, mild imperfections, T = 512 (solid: replica delays fixed at bw 0.05 "
                 "with 95 % CI; dashed: delays ∝ 1/bw)", fontsize=9.5)
    fig.tight_layout()
    a.fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.fig_dir / f"{a.fig_name}.{ext}", dpi=170)
    print(f"[bw] wrote {a.fig_dir}/{a.fig_name}.pdf/.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["decoupled", "coupled", "plot"], required=True)
    ap.add_argument("--output-dir", "-o", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, default=REPO / "docs/figs_revision/r2")
    ap.add_argument("--scenarios", nargs="+", default=["moderate", "crowded", "moderate3"])
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--max-scenes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--ci-seed", type=int, default=20260920)
    ap.add_argument("--draw-seed", type=int, default=20260930, help="seed of the Moderate-3+1 scene draw")
    ap.add_argument("--reconunet", type=Path, default=CK["reconunet"])
    ap.add_argument("--reconunet-c", type=Path, default=None)
    ap.add_argument("--reconunet-cb", type=Path, default=None, help="R2b randomised-bandwidth ReconUNet-CB checkpoint")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, help="subset of methods to evaluate / plot")
    ap.add_argument("--tag", default="", help="suffix for the output file names (e.g. _m5dB)")
    ap.add_argument("--fig-name", default="bandwidth_sweep_r2")
    ap.add_argument("--subvit", type=Path, default=REPO / "experiments/runs/subvit_paper/checkpoints/best.pt")
    ap.add_argument("--damusic-dir", type=Path, default=REPO / "experiments/runs/damusic_paper/ensemble_v2")
    a = ap.parse_args()
    return plot(a) if a.mode == "plot" else run(a, a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
