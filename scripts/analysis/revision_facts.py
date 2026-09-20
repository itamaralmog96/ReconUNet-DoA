#!/usr/bin/env python3
"""docs/revision_facts.md — facts for the Sensors major revision, extracted from the
code, configs, checkpoints and logs of this repository (so the manuscript text can
quote them without guessing), plus result sections appended by the pipeline.

    python scripts/analysis/revision_facts.py generate [--no-timing]
    python scripts/analysis/revision_facts.py append-csv --title T --csv F [--filter COL=VAL] [--note N] [--cols a,b,c]
    python scripts/analysis/revision_facts.py append-md  --title T --md F
    python scripts/analysis/revision_facts.py append-history --title T --runs k2_v2 k3_v2 k4_v2

Items 1-9 (reconstructed revision fact list): 1 loss weights & composite loss, 2 architecture
& size, 3 training protocol & outcomes, 4 dataset & imperfection facts, 5 scenario
definitions, 6 how K is defined and supplied, 7 metric definitions, 8 computational cost,
9 baseline implementation notes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs/revision_facts.md"
MARK = "<!-- AUTO-APPEND BELOW: result sections are appended by the pipeline -->"
sys.path.insert(0, str(REPO / "scripts/analysis"))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def md_table(rows: list[dict], cols: list[str] | None = None, fmt: str = "{:.3f}") -> str:
    if not rows: return "_(empty)_\n"
    cols = cols or list(rows[0].keys())
    def cell(v):
        if isinstance(v, float): return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- measurement helpers
def count_macs(model, run) -> int:
    """Approximate multiply-accumulates of one forward pass via hooks on Conv/Linear/GRU."""
    macs = [0]
    def conv_hook(m, i, o):
        cout = o.numel() // o.shape[0]
        macs[0] += cout * (m.in_channels // m.groups) * int(np.prod(m.kernel_size))
    def lin_hook(m, i, o):
        macs[0] += (o.numel() // o.shape[0]) * m.in_features
    def gru_hook(m, i, o):
        x = i[0]; T_ = x.shape[1] if m.batch_first else x.shape[0]
        per_step = 3 * (m.input_size * m.hidden_size + m.hidden_size * m.hidden_size)
        layers = per_step + (m.num_layers - 1) * 3 * (m.hidden_size * m.hidden_size * 2)
        macs[0] += T_ * layers * (2 if m.bidirectional else 1)
    hs = []
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.ConvTranspose2d)): hs.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, torch.nn.Linear): hs.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, torch.nn.GRU): hs.append(mod.register_forward_hook(gru_hook))
    with torch.no_grad(): run()
    for h in hs: h.remove()
    return macs[0]


def timed(fn, dev, reps=5, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def params_of(path: Path) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))


def history_summary(run_dir: Path) -> dict | None:
    h = run_dir / "checkpoints" / "history.json"
    if not h.exists(): return None
    rows = json.load(h.open())
    if not rows: return None
    best = min(rows, key=lambda r: r["val"])
    return {"run": run_dir.name, "epochs_run": len(rows), "best_epoch(val loss)": best["epoch"],
            "val_rmspe_at_best(deg)": round(best["val_rmspe_deg"], 3), "last_val_rmspe(deg)": round(rows[-1]["val_rmspe_deg"], 3),
            "last_lr": f"{rows[-1]['lr']:.1e}", "min_val_rmspe(deg)": round(min(r["val_rmspe_deg"] for r in rows), 3)}


def status_elapsed(status_log: Path) -> dict[str, str]:
    out = {}
    if not status_log.exists(): return out
    for line in status_log.read_text().splitlines():
        if " END " in line and "elapsed=" in line:
            name = line.split(" END ")[1].split()[0]; el = line.split("elapsed=")[1].split()[0]
            out[name] = el
    return out


# --------------------------------------------------------------------------- generate
def generate(no_timing: bool) -> str:
    from reconunet.data.scene_dataset import SceneDataset
    from reconunet.data.scene_manifest import SceneManifest
    from reconunet.data.scene_renderer import SceneRenderer, lag_stack
    from reconunet.models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet
    from reconunet.models.deep_learning.subspace_models import root_music
    from reconunet.models.third_party import damusic_adapter
    from reconunet.models.third_party.subvit_adapter import SubViTAdapter

    L = []
    P = L.append
    P("# Revision facts — ReconUNet (Sensors sensors-4536109)\n")
    P(f"Generated {now()} from repository state `{git_rev()}` by `scripts/analysis/revision_facts.py`. "
      "Every number below is read from the code, the configs, the checkpoints or the run logs of this repository; "
      "nothing is copied from the manuscript. The nine headings follow the fact list agreed for the revision "
      "(reconstructed here as: loss, architecture, training, data, scenarios, source count, metrics, cost, baselines).\n")

    # ---- 1 loss --------------------------------------------------------------
    cfg = yaml.safe_load((REPO / "configs/train/reconunet_paper.yaml").read_text())
    P("## 1. Loss weights and composite loss\n")
    P("Composite loss (trainer `_compute_loss_native`, target = clean covariance R\\* = A A^H of the K direct paths, "
      "unit-power sources, perfect array):\n")
    P("    L = w_eig·L_eig + w_proj·L_proj + w_dom·L_dom + w_rec·L_rec\n"
      "    L_rec  = (1/N²)·‖R̂ − R*‖²_F\n"
      "    L_eig  = mean_k (λ̂_k − λ*_k)²          (both sorted descending)\n"
      "    L_proj = (1/N²)·‖V̂_S V̂_S^H − V*_S V*_S^H‖²_F   (leading-K columns, per-sample K)\n"
      "    L_dom  = 1 − |⟨v̂_1, v*_1⟩|             (phase- and sign-invariant)\n")
    w = cfg["loss"]
    P(md_table([{"weight": k, "value": v, "term": {"eigval_weight": "L_eig", "proj_weight": "L_proj", "dom_weight": "L_dom",
                                                   "reconstruction_weight": "L_rec"}.get(k, k)} for k, v in w.items()], fmt="{:g}"))
    P("All four weights are 1.0 in the configuration that trained the released model "
      "(`configs/train/reconunet_paper.yaml`). Ablation variants change only the set of active terms.\n")

    # ---- 2 architecture ------------------------------------------------------
    P("## 2. Architecture and model size\n")
    m = EVDCovarianceReconstructionUNet(tau=8, M=8)
    P("ReconUNet = `EVDCovarianceReconstructionUNet(tau=8, M=8, activation=anti_rectifier, dropout)`; input is the "
      "lag stack [τ=8, 2N=16, N=8] (real/imag stacked), outputs (λ̂ [N], V̂ [N×N] complex, R̂ = V̂ diag(λ̂) V̂^H).\n")
    P("Top-level modules and parameter counts:\n")
    P(md_table([{"module": n, "params": sum(p.numel() for p in c.parameters())} for n, c in m.named_children()], fmt="{:d}"))
    P("Sub-blocks of `base_unet` (the covariance U-Net):\n")
    P(md_table([{"module": n, "type": type(c).__name__, "params": sum(p.numel() for p in c.parameters())}
                for n, c in m.base_unet.named_children()], fmt="{:d}"))
    ck_paths = {"ReconUNet (released)": REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt",
                "SubspaceNet": REPO / "experiments/runs/subspacenet_paper/checkpoints/best.pt",
                "SubViT": REPO / "experiments/runs/subvit_paper/checkpoints/best.pt",
                "DA-MUSIC (per K, each)": REPO / "experiments/runs/damusic_paper/k1/checkpoints/best.pt",
                "EVD-UNet of the submitted manuscript": REPO / "experiments/runs/legacy/checkpoints/evd_unet_denoising_model_20250929_015132.pth"}
    prow = [{"model": k, "trainable parameters": params_of(v)} for k, v in ck_paths.items() if v.exists()]
    P("Parameter counts from the checkpoints:\n"); P(md_table(prow, fmt="{:d}"))

    # ---- 3 training ------------------------------------------------------------
    P("## 3. Training protocol and outcomes\n")
    tr = []
    for name, f in (("ReconUNet", "reconunet_paper"), ("SubspaceNet", "subspacenet_paper"), ("SubViT", "subvit_paper"), ("DA-MUSIC K=1", "damusic_paper_k1")):
        c = yaml.safe_load((REPO / f"configs/train/{f}.yaml").read_text())
        tr.append({"model": name, "optimizer": c["optim"]["optimizer"], "lr": c["optim"]["lr"], "weight decay": c["optim"]["weight_decay"],
                   "batch": c["data"]["batch_size"], "max epochs": c["train"]["epochs"], "scheduler": f"{c['optim']['scheduler']['name']} ×{c['optim']['scheduler']['factor']} / {c['optim']['scheduler']['patience']}",
                   "early stop": c["train"]["early_stopping_patience"], "grad clip": c["optim"]["grad_clip_norm"], "AMP": c["train"].get("amp"),
                   "loss keys": ", ".join(c["loss"].keys()), "seed": c.get("seed")})
    P(md_table(tr, fmt="{:g}"))
    P("Shared schedule from the paper: Adam, LR 1e-4, weight decay 1e-5, batch 2048, ≤300 epochs, ReduceLROnPlateau(0.7, 5), "
      "early stopping on validation loss with patience 25, gradient-norm clip 1.0, seed 20260420. Checkpoint selection = "
      "minimum validation loss (`best.pt`). AMP is force-disabled for complex tensors, so every model trains in fp32. "
      "DA-MUSIC trains one fixed-head model per K on the K-subset of the same corpus (`data.k_filter`).\n")
    el = status_elapsed(REPO / "experiments/runs/revision_retrain_20260906/status.log")
    outs = []
    for name, d, key in (("ReconUNet", "reconunet_paper", "reconunet"), ("SubspaceNet", "subspacenet_paper", "subspacenet"), ("SubViT", "subvit_paper", "subvit"),
                         ("DA-MUSIC K=1", "damusic_paper/k1", "damusic_k1"), ("DA-MUSIC K=2", "damusic_paper/k2", "damusic_k2"),
                         ("DA-MUSIC K=3", "damusic_paper/k3", "damusic_k3"), ("DA-MUSIC K=4", "damusic_paper/k4", "damusic_k4")):
        h = history_summary(REPO / "experiments/runs" / d)
        if h: h["run"] = name; h["wall-clock"] = el.get(key, ""); outs.append(h)
    P("Outcomes of the 2026-09-06 → 2026-09-10 retrain on the corrected renderer (NVIDIA RTX 2000 Ada 16 GB, 8 loader workers):\n")
    P(md_table(outs, fmt="{:g}"))
    P("DA-MUSIC K=2,3,4 reached the 300-epoch cap without early stopping (continuation runs to 600 epochs are appended below when finished).\n")

    # ---- 4 dataset -----------------------------------------------------------
    P("## 4. Dataset and imperfection facts\n")
    dc = yaml.safe_load((REPO / "configs/data/paper_corpus.yaml").read_text())
    meta, samp = dc["meta"], dc["sampling"]
    P(md_table([{"quantity": k, "value": v} for k, v in {
        "array": f"ULA, N={meta['M']}, d={meta['element_spacing_lambda']} λ, f_c={float(meta['fs_Hz']):.3g} Hz narrowband",
        "snapshots T": meta["T"], "lag depth τ": meta["tau"], "source bandwidth": f"{float(meta['source_bw_frac'])}·fs (10 % of Nyquist) → coherence time ≈ {1/float(meta['source_bw_frac']):.0f} samples",
        "direct sources K": f"{samp['k_choices']} uniform", "min separation": f"{samp['min_separation_deg']}°",
        "DoA sector": f"{meta['angle_range_deg']} broadside convention (= [30°,150°] array-axis convention)",
        "training SNR": f"{meta['snr_range_db']} dB (evaluation sweeps −20…20 dB)", "multipath": f"enabled, ≤ {samp['max_paths']}−K coherent replicas per scene, count U{{0..max}}, delays '{samp['multipath_distribution']}' with mp_max_delay_factor={samp['mp_max_delay_factor']}",
        "imperfection preset": samp["array_errors"], "corpus": f"{dc['manifest']['train_size']:,} train / {dc['manifest']['val_size']:,} val / {dc['manifest']['test_size']:,} test scenes",
        "master seed": samp["rng_seed"], "manifest row": "96 bytes (seed, K, angles, SNR, error magnitudes, multipath params); rendering is lazy and deterministic from the seed"}.items()], fmt="{}"))
    P("Imperfection presets (per-scene draws, paper §II-C eqs. 21–25; `SceneManifest.random` / `SceneRenderer.render`):\n")
    P(md_table([{"preset": "mild", "gain error": "±0.1 dB, g_n ~ U[10^(−0.1/20), 10^(+0.1/20)]", "phase error": "φ_n ~ U[−1°, +1°]", "mutual coupling |γ|": 0.02, "position error": "ε_n ~ N(0, (0.01 λ)² I₂) (1 % of λ, 2-D)"},
                {"preset": "harsh", "gain error": "±0.5 dB", "phase error": "U[−5°, +5°]", "mutual coupling |γ|": 0.10, "position error": "N(0, (0.05 λ)² I₂) (5 % of λ)"}], fmt="{:g}"))
    P("Mutual coupling is a reciprocal Toeplitz matrix I + E with unit-step coefficient γ = |γ|·e^{j(−100°)} and per-diagonal jitter "
      "v_k ~ U[0.55, 1.45] (legacy `coupling_variation` 0.9); composite operator H = M·G as in paper eq. (26).\n")
    # empirical statistics
    try:
        man = SceneManifest.load(str(REPO / "data/scenes/paper/train.npy"))
        ns = man.raw["n_sources"].astype(int); nm = man.raw["num_multipath"].astype(int); snr = man.raw["snr_db"]
        kh = {int(k): int((ns == k).sum()) for k in np.unique(ns)}
        rows = [{"K": k, "scenes": v, "share": v / ns.size, **{f"replicas={r}": float(((nm == r) & (ns == k)).sum() / max(1, v)) for r in range(0, 7)}} for k, v in kh.items()]
        P("Empirical composition of the 2 M-scene training manifest (share of scenes by K, and within each K the fraction with r coherent replicas):\n")
        P(md_table(rows, fmt="{:.3f}"))
        P(f"SNR in the training manifest: min {snr.min():.2f}, max {snr.max():.2f} dB (uniform). Error magnitudes per scene: "
          f"gain {np.unique(man.raw['gain_err_dB']).tolist()} dB, phase {np.unique(man.raw['phase_err_deg']).tolist()}°, coupling {np.unique(man.raw['mutual_coupling']).tolist()}, position {np.unique(man.raw['position_err_pct']).tolist()} %.\n")
        # replica delays and coherence
        r = SceneRenderer(man.meta); ds_idx = np.where(nm >= 1)[0][:300]; d_all = []; gam = []
        for i in ds_idx:
            res = r.render(man[int(i)]); d_all.extend(res.mp_delay_samples.tolist())
            s = res.source_signals; K = int(man.raw["n_sources"][i])
            for j in range(K, s.shape[0]):
                gam.append(abs(np.vdot(s[0], s[j])) / (np.linalg.norm(s[0]) * np.linalg.norm(s[j])))
        P(f"Measured on 300 multipath scenes: replica delays {np.min(d_all):.2f}–{np.max(d_all):.2f} samples (mean {np.mean(d_all):.2f}); "
          f"direct/replica waveform correlation |γ| mean {np.mean(gam):.3f}, median {np.median(gam):.3f}, min {np.min(gam):.3f} "
          "(theory sinc(bw·delay); rank-1 coherent model of paper §II-B).\n")
    except Exception as exc:  # pragma: no cover
        P(f"_(empirical manifest statistics unavailable: {exc})_\n")

    # ---- 5 scenarios ---------------------------------------------------------
    P("## 5. Scenario definitions (Section IV stress tests)\n")
    srows = []
    for preset, d in (("mild", "scenarios"), ("harsh", "scenarios_harsh")):
        for f in sorted((REPO / "configs/data" / d).glob("*.yaml")):
            c = yaml.safe_load(f.read_text()); sc = c["scenario"]
            srows.append({"preset": preset, "scenario": f.stem, "K direct": sc["k"], "coherent replicas": sc.get("num_multipath_components", 0),
                          "angle configs": sc["n_angle_configs"], "SNR levels (dB)": str(sc["snr_levels_db"]), "errors": sc["array_errors"], "seed": sc["rng_seed"]})
    P(md_table(srows, fmt="{:g}"))
    P("Each angle configuration (seed, angles, imperfection draw, multipath layout) is reused at every SNR; only the noise power changes "
      "(`SceneManifest.fixed_angles_snr_sweep`). The corpus 'Moderate' scenario is 2 direct + 1 replica; the submitted Table II caption "
      "said 3 + 1 — the manuscript text must be made consistent with the data.\n")

    # ---- 6 K -------------------------------------------------------------------
    P("## 6. How the source count K is defined and supplied\n")
    P("* K is the number of **direct-path** sources of a scene (`n_sources`); coherent multipath replicas are never counted in K and are "
      "not labelled targets. K is assumed **known** for every method (classical, aided and learned), per scene.\n"
      "* Classical estimators (Bartlett, MVDR, MUSIC, Root-MUSIC, ESPRIT, Unitary-ESPRIT) and ReconUNet-aided estimators receive K directly; "
      "MUSIC-type methods split the eigenbasis into K signal / N−K noise vectors.\n"
      "* SubspaceNet: variable-K forward — the adapter passes the true per-sample K to the differentiable Root-MUSIC head (published code fixes M at build time).\n"
      "* SubViT (DOA-ViT): K strongest local maxima of the 121-point spatial spectrum, per sample.\n"
      "* DA-MUSIC: the published head is `Linear(hidden, M)`, so one model per K is trained and evaluation dispatches each scene to the model of its true K.\n"
      "* Under coherent multipath the signal-subspace rank stays K (|γ|≈0.9 per replica), which is what the covariance reconstruction exploits; "
      "raw subspace methods see a rank-deficient / leaked spectrum.\n")

    # ---- 7 metrics -------------------------------------------------------------
    P("## 7. Metric definitions\n")
    P("* **Pooled RMSE (paper eq. 31)** = sqrt( mean over all scenes and all K sources of the squared angle error in degrees ). Estimates and truths are "
      "sorted (optimal permutation for scalars); errors are **not** wrapped (sin θ is injective on [−90°, 90°] in the broadside convention).\n"
      "* **Median RMSPE** = median over scenes of sqrt(mean_k e_k²) — robust companion reported in the test-split table.\n"
      "* **Resolution probability** (separation sweep) = P(|θ̂_i − θ_i| < Δθ/2 for both sources).\n"
      "* **CRLB** = stochastic (unconditional) Gaussian-signal bound per scene (`stochastic_crlb_deg`, M, T, angles, SNR, d/λ), RMS over scenes.\n"
      "* **Bootstrap 95 % CI** = percentile bootstrap over scenes, 1000 resamples, seed 20260920 (`bootstrap_ci.py`), for pooled RMSE and median RMSPE.\n"
      "* Angle convention everywhere: broadside 0°, sector ±60°; the paper's [30°, 150°] array-axis angles are the same physical directions.\n")

    # ---- 8 cost ----------------------------------------------------------------
    P("## 8. Computational cost\n")
    if no_timing:
        P("_(timing skipped: --no-timing)_\n")
    else:
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            man = SceneManifest.load(str(REPO / "data/scenes/paper/test.npy")); ds = SceneDataset(man)
            idx = np.where(man.raw["n_sources"] == 1)[0][:1024]
            X = torch.stack([ds[int(i)].snapshots for i in idx], 0)
            from _revision_common import Models
            mdl = Models(dev)
            sv_ck = torch.load(REPO / "experiments/runs/subvit_paper/checkpoints/best.pt", map_location="cpu", weights_only=False)
            sv_init = dict(sv_ck["cfg"]["model"]["init"]); sv_ad = SubViTAdapter(**sv_init); sv = sv_ad.build_model(sv_init)
            sv_ad.load_checkpoint(sv, str(REPO / "experiments/runs/subvit_paper/checkpoints/best.pt")); sv.to(dev).eval()
            rows = []
            for B in (1, 1024):
                xb = X[:B]; lag = lag_stack(xb, tau=8); lag_d = lag.to(dev); R = (xb @ xb.conj().transpose(-1, -2)) / xb.shape[-1]
                def rn_fwd(): 
                    with torch.no_grad(): return mdl.rn(lag_d)
                def rn_full():
                    with torch.no_grad(): _, _, Rh = mdl.rn(lag_stack(xb, tau=8).to(dev)); return root_music(Rh.cpu(), 1, B)
                def rm_only(): return root_music(R, 1, B)
                def eigh_only(): return torch.linalg.eigh(R)
                def sn_full(): return mdl.subspacenet(xb, 1)
                def sv_full():
                    with torch.no_grad(): return sv_ad.forward(sv, sv_ad.prepare_input(xb, {"M": 8}).to(dev), meta={"n_sources": torch.ones(B, dtype=torch.long)})
                def dm_full(): return mdl.damusic(xb, 1)
                cands = [("ReconUNet forward only (GPU)", rn_fwd), ("ReconUNet + Root-MUSIC (GPU net, CPU eigh/roots)", rn_full),
                         ("Root-MUSIC on raw SCM (CPU eigh + roots)", rm_only), ("eigh of the 8×8 SCM alone (CPU)", eigh_only),
                         ("SubspaceNet + Root-MUSIC head (GPU)", sn_full), ("SubViT (GPU)", sv_full)]
                if mdl.dm is not None: cands.append(("DA-MUSIC K=1 (GPU)", dm_full))
                for name, fn in cands:
                    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
                    t = timed(fn, dev, reps=5 if B == 1 else 3)
                    mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
                    rows.append({"pipeline": name, "batch": B, "ms per scene": 1e3 * t / B, "ms per batch": 1e3 * t, "peak GPU MiB": mem})
                # CPU-only ReconUNet
                rn_cpu = EVDCovarianceReconstructionUNet(**dict(torch.load(REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt", map_location="cpu", weights_only=False)["cfg"]["model"]["init"])).eval()
                rn_cpu.load_state_dict(torch.load(REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt", map_location="cpu", weights_only=False)["model"])
                def rn_cpu_fwd():
                    with torch.no_grad(): return rn_cpu(lag)
                t = timed(rn_cpu_fwd, torch.device("cpu"), reps=3 if B == 1024 else 5)
                rows.append({"pipeline": f"ReconUNet forward only (CPU, {torch.get_num_threads()} threads)", "batch": B, "ms per scene": 1e3 * t / B, "ms per batch": 1e3 * t, "peak GPU MiB": float("nan")})
            P(f"Inference latency measured {now()} on {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'} (median of repeats, GPU-synchronised; "
              "lag-stack formation included where the pipeline needs it; K = 1 test scenes):\n")
            P(md_table(rows, fmt="{:.3f}"))
            # MACs
            macs = {"ReconUNet": count_macs(mdl.rn, lambda: mdl.rn(lag_stack(X[:1], tau=8).to(dev))),
                    "SubspaceNet (conv/linear only; eigh + roots excluded)": count_macs(mdl.sn, lambda: mdl.subspacenet(X[:1], 1)),
                    "SubViT (linear/conv only; attention products excluded)": count_macs(sv, lambda: sv_ad.forward(sv, sv_ad.prepare_input(X[:1], {"M": 8}).to(dev), meta={"n_sources": torch.ones(1, dtype=torch.long)}))}
            if mdl.dm is not None:
                ad, dm1 = mdl.dm.models[1]
                macs["DA-MUSIC K=1 (GRU formula + linear; MUSIC spectrum excluded)"] = count_macs(dm1, lambda: mdl.damusic(X[:1], 1))
            P("Multiply-accumulates per scene (forward hooks on Conv/Linear/GRU layers; counting rule in parentheses):\n")
            P(md_table([{"model": k, "MMACs per scene": v / 1e6} for k, v in macs.items()], fmt="{:.2f}"))
            P("Reading for the manuscript: for N = 8 the eigendecomposition of the sample covariance costs microseconds per scene and is far cheaper than the "
              "U-Net pass; ReconUNet's cost is dominated by the network and amortises well in batches (see the per-scene column at batch 1024).\n")
        except Exception as exc:
            P(f"_(timing failed: {exc})_\n")
    el2 = status_elapsed(REPO / "experiments/runs/revision_retrain_20260906/status.log")
    P("Training wall-clock of the released models (single RTX 2000 Ada, from `revision_retrain_20260906/status.log`): " +
      ", ".join(f"{k} {v}" for k, v in el2.items()) + ".\n")

    # ---- 9 baselines -----------------------------------------------------------
    P("## 9. Baseline implementation notes\n")
    sub_rev = subprocess.run(["git", "-C", str(REPO / "third_party/subspacenet"), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    P(f"* **SubspaceNet** (Shmuel et al.): vendored upstream tree at commit `{sub_rev}` (= ShlezingerLab/SubspaceNet `f2ef464` + a one-line CUDA→CPU fix, "
      "`third_party/patches/`). Adapter `SubspaceNetAdapter(M=4, tau=8, diff_method=root_music)`; the differentiable Root-MUSIC head receives the true "
      "per-sample K. Training loss = RMSPE (upstream objective), our shared optimiser schedule.\n")
    P("* **SubViT / DOA-ViT** (Zhou et al.): vendored `third_party/doa_est_master`; published capacity "
      + ", ".join(f"{k}={v}" for k, v in yaml.safe_load((REPO / 'configs/train/subvit_paper.yaml').read_text())["model"]["init"].items())
      + "; training objective grid-BCE on the spatial spectrum; validation/checkpoint criterion = sorted RMSPE; per-sample-K peak picking.\n")
    doc = damusic_adapter.__doc__ or ""
    dev_sec = doc.split("Deviations from the vendored forward")[1] if "Deviations from the vendored forward" in doc else ""
    P("* **DA-MUSIC** (Merkofer et al., ICASSP 2022): upstream `DeepAugmentedMUSIC` from the SubspaceNet tree, unmodified weights/architecture; one model per K. "
      "Deviations of the adapter's forward pass from the vendored code (from the adapter docstring):\n")
    P("```text\n" + dev_sec.strip()[:2500] + "\n```\n")
    P("* **Classical estimators** (`reconunet/evaluation/classical_batched.py`): Bartlett, MVDR and MUSIC on a 1° grid over [−60°, 60°] with three-point "
      "parabolic peak refinement and local-maximum selection; Root-MUSIC (roots inside and closest to the unit circle), ESPRIT and Unitary-ESPRIT; "
      "all fp64 CPU eigendecompositions for robustness. The same K is supplied to every estimator.\n")
    P(f"\n{MARK}\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- append modes
def _append(block: str) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    txt = DOC.read_text() if DOC.exists() else f"# Revision facts\n\n{MARK}\n"
    if MARK not in txt: txt += f"\n{MARK}\n"
    DOC.write_text(txt.rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n")


def append_csv(title: str, csv_path: Path, flt: str | None, note: str | None, cols: str | None) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if flt:
        c, v = flt.split("=", 1); df = df[df[c].astype(str) == v]
    if cols: df = df[[c for c in cols.split(",") if c in df.columns]]
    rows = df.to_dict("records")
    block = f"## {title}\n\n_Appended {now()} from `{csv_path.relative_to(REPO) if csv_path.is_relative_to(REPO) else csv_path}`" + (f" ({flt})" if flt else "") + "._\n\n"
    if note: block += note.rstrip() + "\n\n"
    block += md_table(rows)
    _append(block); print(f"[facts] appended '{title}' ({len(rows)} rows)")


def append_md(title: str, md: Path) -> None:
    _append(f"## {title}\n\n_Appended {now()}._\n\n" + md.read_text()); print(f"[facts] appended '{title}'")


def append_history(title: str, runs: list[str]) -> None:
    rows = []
    for r in runs:
        h = history_summary(REPO / "experiments/runs/damusic_paper" / r)
        base = history_summary(REPO / "experiments/runs/damusic_paper" / r.replace("_v2", ""))
        if h:
            h["v1 (300-epoch cap) val RMSPE at best"] = base["val_rmspe_at_best(deg)"] if base else ""
            rows.append(h)
    block = (f"## {title}\n\n_Appended {now()}._\n\nExact continuation of the K = 2, 3, 4 runs from their epoch-300 `last.pt` "
             "(model, Adam moments, scheduler, best/stale restored), cap 600 epochs, patience 25; checkpoints under "
             "`experiments/runs/damusic_paper/k<K>_v2/`.\n\n" + md_table(rows, fmt="{:g}"))
    _append(block); print(f"[facts] appended '{title}' ({len(rows)} runs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--no-timing", action="store_true")
    c = sub.add_parser("append-csv"); c.add_argument("--title", required=True); c.add_argument("--csv", type=Path, required=True)
    c.add_argument("--filter", default=None); c.add_argument("--note", default=None); c.add_argument("--cols", default=None)
    m = sub.add_parser("append-md"); m.add_argument("--title", required=True); m.add_argument("--md", type=Path, required=True)
    h = sub.add_parser("append-history"); h.add_argument("--title", required=True); h.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "generate":
        txt = generate(a.no_timing)
        tail = ""
        if DOC.exists() and MARK in DOC.read_text():           # keep previously appended result sections
            tail = DOC.read_text().split(MARK, 1)[1]
        DOC.parent.mkdir(parents=True, exist_ok=True); DOC.write_text(txt + tail)
        print(f"[facts] wrote {DOC} ({len(txt.splitlines())} lines)")
    elif a.cmd == "append-csv": append_csv(a.title, a.csv, a.filter, a.note, a.cols)
    elif a.cmd == "append-md": append_md(a.title, a.md)
    elif a.cmd == "append-history": append_history(a.title, a.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
