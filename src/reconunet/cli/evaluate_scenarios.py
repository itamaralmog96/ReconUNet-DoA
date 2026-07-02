"""``reconunet-evaluate-scenarios`` — paper §IV scenario stress tests.

Drives a single trained ReconUNet checkpoint through one or more *scenario*
manifests built by ``reconunet-generate -c configs/data/scenarios/*.yaml``
(the "consistent angles across SNR" construction from paper §IV) and writes
one CSV per scenario plus a combined Table-II-style summary at 0 dB.

Within each scenario the harness reports per-SNR RMSPE for:

  * Root-MUSIC and ESPRIT on the raw SCM (paper baselines, "thin solid lines"
    in Figs. 8-10), and
  * Root-MUSIC and ESPRIT on the reconstructed covariance R̂ produced by the
    trained ReconUNet (paper "dashed star-marked line").

The two pipelines share the *same* batch and the *same* angle ground truth,
so per-bucket RMSPE numbers are directly comparable.

Usage::

    reconunet-evaluate-scenarios \\
        --checkpoint experiments/runs/reconunet_paper/checkpoints/best.pt \\
        --manifests data/scenes/scenarios/basic/test.npy \\
                    data/scenes/scenarios/moderate/test.npy \\
                    data/scenes/scenarios/advanced1_ood/test.npy \\
                    data/scenes/scenarios/advanced2_crowded/test.npy \\
        --output-dir experiments/runs/scenarios/

Each scenario produces ``<scenario>/results.csv`` and the driver prints a
combined "RMSE at 0 dB" table to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from reconunet.data.scene_dataset import SceneDataset, get_collate
from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_renderer import SceneRenderer
from reconunet.models.deep_learning.EVDUNet import (
    EVDCovarianceReconstructionUNet,
)
from reconunet.models.deep_learning.subspace_models import esprit, root_music

LOG = logging.getLogger("reconunet.evaluate_scenarios")


# ---------------------------------------------------------------------------
# Periodic RMSPE — same conventions as unified_harness.rmspe_deg.
# ---------------------------------------------------------------------------


def _rmspe_deg(pred_rad: np.ndarray, true_rad: np.ndarray) -> np.ndarray:
    """Per-sample mean squared error in deg², sorted on both sides (= optimal
    permutation for scalars, paper eq. 31); NO angular wrap (broadside sinθ is
    injective on [-90°,90°]).  Returns a [B] array of per-sample MSEs so the
    caller pools with sqrt(mean) — the paper's pooled RMSE — instead of
    averaging per-sample RMSPEs (Jensen gap up to ~3× in outlier regimes)."""
    p = np.sort(pred_rad, axis=-1)
    t = np.sort(true_rad, axis=-1)
    err_deg2 = np.rad2deg(p - t) ** 2
    with np.errstate(all="ignore"):
        return np.nanmean(err_deg2, axis=-1)


# ---------------------------------------------------------------------------
# Classical estimators on a batched complex covariance.  Wrap the differentiable
# implementations from subspace_models so they take a NumPy ground-truth K and
# return predicted angles in radians (broadside-0° convention).
# ---------------------------------------------------------------------------


def _angles_from_root_music(R: torch.Tensor, K: int) -> np.ndarray:
    """Run Root-MUSIC on ``[B, M, M]`` complex covariance.  Returns ``[B, K]``
    in radians (broadside zero) ready for RMSPE."""
    R_cpu = R.detach().cpu()
    B = int(R_cpu.shape[0])
    angles_deg, _, _ = root_music(R_cpu, K, B)
    # subspace_models.root_music returns angles in [0, 180] (cosine convention).
    # Convert to broadside-0° (sine convention).
    return np.deg2rad(angles_deg.numpy() - 90.0)


def _angles_from_esprit(R: torch.Tensor, K: int) -> np.ndarray:
    """Run ESPRIT on ``[B, M, M]`` complex covariance.  Returns ``[B, K]``
    in radians (broadside zero).

    ``subspace_models.esprit`` already produces broadside-0° angles in
    radians (its last line is ``-arcsin(angles / pi)``), so no further
    conversion is needed — unlike ``root_music`` which returns degrees in
    the array-axis convention.
    """
    R_cpu = R.detach().cpu()
    B = int(R_cpu.shape[0])
    angles_rad = esprit(R_cpu, K, B)
    return angles_rad.numpy()


# ---------------------------------------------------------------------------
# Per-scenario evaluation.
# ---------------------------------------------------------------------------


def _evaluate_scenario(
    manifest_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> "pandas.DataFrame":   # noqa: F821
    """Run the four estimator pipelines on one scenario manifest.

    Returns a long-form frame with columns ``[snr_db, method, rmse_deg,
    n_samples]`` where ``method`` ∈ {raw_root_music, raw_esprit,
    reconunet_root_music, reconunet_esprit}.
    """
    import pandas as pd                            # local import — optional dep

    manifest = SceneManifest.load(str(manifest_path))
    meta = manifest.meta
    renderer = SceneRenderer(meta)
    dataset = SceneDataset(manifest, renderer=renderer)
    collate = get_collate("reconunet", meta)

    # Per-scenario K is constant — pull it from the first scene.
    K = int(manifest.raw["n_sources"][0])

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate, drop_last=False,
    )

    rows: List[Dict[str, float]] = []
    # Group RMSPE samples by (snr_db, method) for averaging.
    bucket: Dict[tuple, List[float]] = {}

    with torch.no_grad():
        for batch in loader:
            R_raw = batch["covariance"]                                # [B, M, M] complex
            angles_true = batch["angles_rad"][:, :K].numpy()           # [B, K]
            snr_db = batch["snr_db"].numpy()                           # [B]
            x = batch["input"].to(device)

            # ---- ReconUNet forward → reconstructed covariance --------------
            _eigvals, _eigvecs, R_hat = model(x)
            R_hat = R_hat.detach().cpu()
            # The forward produces a ``[B, M, M]`` complex tensor; ensure dtype.
            if not torch.is_complex(R_hat):
                # Fallback: split real/imag pairs along last dim.  Should not
                # happen for the EVDUNet head but keeps us defensive.
                raise RuntimeError(
                    f"ReconUNet head returned non-complex output of dtype {R_hat.dtype}"
                )

            # ---- Run all four estimator pipelines --------------------------
            results = {
                "raw_root_music":       _angles_from_root_music(R_raw, K),
                "raw_esprit":           _angles_from_esprit(R_raw, K),
                "reconunet_root_music": _angles_from_root_music(R_hat, K),
                "reconunet_esprit":     _angles_from_esprit(R_hat, K),
            }

            # ---- Per-sample MSE, bucketed by SNR ---------------------------
            for method, pred in results.items():
                per_sample = _rmspe_deg(pred, angles_true)             # [B] MSE deg²
                for s, e in zip(snr_db, per_sample):
                    if not np.isfinite(e):
                        continue
                    bucket.setdefault((float(s), method), []).append(float(e))

    for (snr, method), errs in sorted(bucket.items()):
        rows.append({
            "snr_db":    snr,
            "method":    method,
            # Pooled RMSE (paper eq. 31); K is constant within a scenario so
            # the mean of per-sample MSEs equals the per-source pooled mean.
            "rmse_deg":  float(np.sqrt(np.mean(errs))),
            "n_samples": len(errs),
        })
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Checkpoint loading.
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained EVDCovarianceReconstructionUNet from a checkpoint.

    Supports two on-disk formats:

    * The current ``reconunet-train`` format
      (``{"model": state_dict, "cfg": {...}}``) where init args come from
      ``cfg.model.init``.
    * The 2025 legacy format
      (``{"model_state_dict": ..., "model_config": {...}}``) used by the
      original notebook trainer.  Init args come from ``model_config`` and
      the keys still match the current module's attribute names
      (``base_unet.*``, ``eigenvalue_head.*``, ``eigenvector_head.*``) so
      no renaming is needed.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state and "model_config" in state:
        # Legacy notebook format.
        init = dict(state["model_config"])
        sd = state["model_state_dict"]
        epoch = int(state.get("epoch", -1))
        val_loss = float(state.get("val_loss", float("nan")))
        val_rmspe = float("nan")
    else:
        cfg = state.get("cfg", {})
        init = dict(cfg.get("model", {}).get("init", {}))
        sd = state["model"]
        epoch = int(state.get("epoch", -1))
        val_loss = float(state.get("val_loss", float("nan")))
        val_rmspe = float(state.get("val_rmspe_deg", float("nan")))
    init.setdefault("M", 8)
    init.setdefault("tau", 8)
    init.setdefault("activation_type", "anti_rectifier")
    init.setdefault("use_dropout", True)
    model = EVDCovarianceReconstructionUNet(**init).to(device)
    model.load_state_dict(sd)
    model.eval()
    LOG.info(
        "Loaded checkpoint from %s  (epoch %d, val_loss=%.4f, val_rmspe=%.3f°)",
        checkpoint_path, epoch, val_loss, val_rmspe,
    )
    return model


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reconunet-evaluate-scenarios",
        description="Run the trained ReconUNet checkpoint through paper §IV "
                    "scenario manifests and write per-scenario RMSPE tables.",
    )
    parser.add_argument("--checkpoint", "-c", required=True, type=Path,
                        help="Path to a best.pt checkpoint produced by reconunet-train.")
    parser.add_argument("--manifests", "-m", required=True, nargs="+", type=Path,
                        help="One or more scenario test manifests "
                             "(data/scenes/scenarios/<scenario>/test.npy).")
    parser.add_argument("--output-dir", "-o", required=True, type=Path,
                        help="Output root; per-scenario subdirs are created automatically.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Inference batch size (default 128).")
    parser.add_argument("--device", default=None,
                        help="Override torch device (auto-detects cuda otherwise).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    LOG.info("Device: %s", device)

    model = _load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    summary_rows: List[Dict[str, float]] = []
    for manifest_path in args.manifests:
        scenario = manifest_path.parent.name              # …/scenarios/<name>/test.npy
        LOG.info("=== Scenario %s  (%s) ===", scenario, manifest_path)
        scen_dir = args.output_dir / scenario
        scen_dir.mkdir(parents=True, exist_ok=True)

        df = _evaluate_scenario(manifest_path, model, device,
                                batch_size=int(args.batch_size))
        df.to_csv(scen_dir / "results.csv", index=False)
        LOG.info("  wrote %s  (%d rows)", scen_dir / "results.csv", len(df))

        # Print this scenario's RMSE-vs-SNR pivot.
        pivot = df.pivot_table(index="snr_db", columns="method",
                               values="rmse_deg", aggfunc="mean")
        print(f"\n--- {scenario} ---")
        print(pivot.to_string(float_format=lambda x: f"{x:7.3f}"))

        # Capture the 0-dB row for the combined Table II view.
        if 0.0 in pivot.index:
            row = pivot.loc[0.0]
            for method, val in row.items():
                summary_rows.append({
                    "scenario": scenario,
                    "method":   method,
                    "rmse_deg_at_0dB": float(val),
                })

    # Combined Table II at 0 dB across scenarios.
    if summary_rows:
        summary = pd.DataFrame.from_records(summary_rows)
        summary_path = args.output_dir / "summary_0dB.csv"
        summary.to_csv(summary_path, index=False)
        pivot = summary.pivot_table(index="method", columns="scenario",
                                    values="rmse_deg_at_0dB", aggfunc="mean")
        print("\n=== RMSE (deg) at 0 dB across scenarios ===")
        print(pivot.to_string(float_format=lambda x: f"{x:7.3f}"))
        LOG.info("wrote %s", summary_path)

    # Provenance dump.
    (args.output_dir / "config.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "manifests":  [str(p) for p in args.manifests],
        "batch_size": int(args.batch_size),
        "device":     str(device),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
