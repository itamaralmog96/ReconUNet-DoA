#!/usr/bin/env python3
"""Drive the legacy ``controlled_angle_evaluation.py`` four times — once per
paper §IV scenario — against a single ReconUNet checkpoint.

Why this wrapper exists.  The legacy script is configured via module-level
constants (``UNET_MODEL_PATH``, ``NUM_SOURCES``, ``NUM_MULTIPATH``,
``EVALUATION_MODE``, ``OUTPUT_DIR``).  Editing the file four times by hand
is error-prone and breaks reproducibility.  This wrapper imports the legacy
module, mutates its globals for each scenario, then calls its ``main()``.

Outputs (per scenario, under ``--output-dir``):

  * ``<scenario>/results_mode<N>_<S>src_mp<M>_<imperfect|perfect>.json``
    — raw per-sample RMSE arrays, one entry per algorithm and SNR.
  * ``<scenario>/rmse_vs_snr_*.png`` — paper-style log-y plot.

Plus a combined ``summary_0dB.csv`` reproducing paper Table II layout.

Usage::

    python scripts/analysis/run_legacy_paper_table.py \\
        --checkpoint experiments/runs/reconunet_paper/checkpoints/best.pt \\
        --output-dir experiments/runs/scenarios_paper_v2_legacy/
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# The four paper §IV scenarios.  ``mode`` is the legacy
# ``controlled_angle_evaluation.EVALUATION_MODE``:
#   3 = random angles per SNR (independent),
#   4 = same angles across all SNR levels (paper §IV "consistent angles").
# ``num_sources`` is the legacy ``[main, interference]`` pair; total
# direct sources = main + interference.  ``num_multipath`` is the per-main-
# source coherent replica count.
# ---------------------------------------------------------------------------
SCENARIOS: List[Dict] = [
    # Paper §IV: "Basic: one direct source (no multipath)."
    dict(name="basic",             mode=4, num_sources=[1, 0], num_multipath=0,
         imperfections=True),
    # Paper §IV: "Moderate: two direct sources plus one coherent replica."
    # Paper Table II columns label this "Mod (3+1)" — 3 direct (main+inter)
    # plus 1 coherent replica.  The legacy script's "main + interference"
    # split totals 3 here.
    dict(name="moderate",          mode=4, num_sources=[1, 2], num_multipath=1,
         imperfections=True),
    # Paper §IV: "Advanced 1: one direct source with six coherent replicas
    # (OOD relative to training where M ≤ 3)."
    # Legacy used Mode 3 for this column (independent angles per SNR).
    dict(name="advanced1_ood",     mode=3, num_sources=[1, 0], num_multipath=6,
         imperfections=True),
    # Paper §IV: "Advanced 2: four direct sources with three coherent
    # replicas spread across emitters."
    dict(name="advanced2_crowded", mode=4, num_sources=[1, 3], num_multipath=3,
         imperfections=True),
]


# ---------------------------------------------------------------------------
# Loading the legacy module
# ---------------------------------------------------------------------------


def _import_legacy(repo_root: Path):
    """Import ``scripts/analysis/controlled_angle_evaluation`` after patching
    ``sys.path`` so its bare ``from data.dataset_generator import ...``
    imports resolve to the project's ``src/reconunet`` tree."""
    sys.path.insert(0, str(repo_root / "src" / "reconunet"))
    sys.path.insert(0, str(repo_root / "scripts" / "analysis"))
    return importlib.import_module("controlled_angle_evaluation")


def _make_bridged_model_class(real_class):
    """Return a subclass of the real EVDUNet that bridges the legacy-data and
    training-data signal conventions.

    The two pipelines differ in two places:

    1. **Power scale.**  ``signalgen.ArrayModel`` (what the legacy script
       uses) generates received signals with mean per-element power 1/M,
       so its sample covariance is exactly ``R_legacy = R_train / M``
       where ``R_train`` is what the 2026 ``SceneRenderer`` produces for
       the same source/SNR/imperfections.  Empirically (M=8, K=1, no
       multipath, no imperfections, 20 dB):

           ‖R_train − M · R_legacy‖_F / ‖R_train‖_F  ≈ 0.045
           ‖R_train −     R_legacy‖_F / ‖R_train‖_F  ≈ 0.88

    2. **Steering convention.**  The two pipelines map array-axis angles
       to phase progressions through different conventions
       (``exp(+jπn·cos)`` vs ``exp(-jπn·sin)`` plus a ±90° array-centre
       offset).  Subspace methods are sign-invariant on the eigenvector
       phase progression, so this only matters for the predicted angle's
       sign — which the legacy ESPRIT decoder absorbs via ``arccos`` on
       the rotation eigenvalues.

    The wrapper rescales the input lag-stack by ``M`` to put it in
    training scale.  The output ``R_hat`` is left in training scale —
    ESPRIT/Root-MUSIC are scale-invariant, so this is harmless.  No
    conjugation is applied: at the pair (legacy θ=60°, broadside=-30°)
    the two covariances differ purely by the scalar ``M``, with no
    conjugate-transpose factor.
    """
    class BridgedEVDUNet(real_class):
        def forward(self, x):
            # Rescale legacy input to training-scale.  Imag and real halves
            # (rows :M and M:2M of dim=-2) both get the same multiplier.
            M = x.shape[-1]
            return super().forward(x * float(M))
    BridgedEVDUNet.__name__ = real_class.__name__   # legacy script prints this
    return BridgedEVDUNet


# ---------------------------------------------------------------------------
# Per-scenario driver
# ---------------------------------------------------------------------------


def _run_one(legacy, scenario: Dict, checkpoint_path: Path,
             output_root: Path, samples_per_snr: int, snr_levels: List[int],
             dataset_seed: int) -> Path:
    """Mutate legacy module globals, run the legacy main() once, return the
    expected results JSON path."""
    s = scenario
    # ---- Patch legacy globals ---------------------------------------------
    legacy.EVALUATION_MODE      = int(s["mode"])
    legacy.NUM_SOURCES          = list(s["num_sources"])
    legacy.NUM_MULTIPATH        = int(s["num_multipath"])
    legacy.ARRAY_IMPERFECTIONS  = bool(s["imperfections"])
    legacy.SNR_DB               = list(snr_levels)
    legacy.SAMPLES_PER_SNR      = int(samples_per_snr)
    legacy.UNET_MODEL_PATH      = str(checkpoint_path)
    # The legacy script writes JSONs/PNGs under OUTPUT_DIR using a
    # ``mode<N>_<sources>src_mp<M>_<imperfect|perfect>`` naming convention,
    # so we just point OUTPUT_DIR at a per-scenario subdir.
    scen_dir = output_root / s["name"]
    scen_dir.mkdir(parents=True, exist_ok=True)
    legacy.OUTPUT_DIR           = str(scen_dir)
    legacy.DATASET_GENERATION_SEED = int(dataset_seed)
    # Pin the dataset name so each scenario writes to its own subdir under
    # ``Data/datasets`` (the legacy generator's own scratch location) and
    # we don't accidentally reuse a previous scenario's dataset.
    legacy.DATASET_NAME         = f"controlled_eval_{s['name']}"

    # ---- Run --------------------------------------------------------------
    legacy.main()

    # ---- Locate the JSON the legacy script wrote --------------------------
    suffix = "imperfect" if s["imperfections"] else "perfect"
    n_total = sum(s["num_sources"])
    json_name = (f"results_mode{s['mode']}_{n_total}src_mp{s['num_multipath']}"
                 f"_{suffix}.json")
    json_path = scen_dir / json_name
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected legacy output {json_path} but it wasn't written. "
            f"Check the legacy script's stdout for errors above."
        )
    return json_path


def _rmse_at(d: dict, snr: float, method: str) -> float:
    """Aggregate the per-sample arrays in the legacy JSON the same way the
    legacy script does (RMS over samples)."""
    v = np.asarray(d.get(str(int(snr)), {}).get(method, []), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(v ** 2)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_legacy_paper_table",
        description="Run the legacy controlled_angle_evaluation script across "
                    "all four paper §IV scenarios on one checkpoint.")
    parser.add_argument("--checkpoint", "-c", required=True, type=Path,
                        help="Path to ReconUNet best.pt (any format the legacy "
                             "script accepts: it understands {model_state_dict, "
                             "model_config} as well as the new "
                             "{model, cfg.model.init} format).")
    parser.add_argument("--output-dir", "-o", required=True, type=Path,
                        help="Output root.  Per-scenario subdirs are created.")
    parser.add_argument("--samples-per-snr", type=int, default=1000,
                        help="Paper used 1000 (default).")
    parser.add_argument("--snr-levels", default="-20,-15,-10,-5,0,5,10,15,20",
                        help="Comma-separated SNRs in dB.")
    parser.add_argument("--dataset-seed", type=int, default=20260428,
                        help="Seed for the legacy dataset generator.")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario names to run "
                             "(default: all four).")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snr_levels = [int(s) for s in args.snr_levels.split(",")]

    # The legacy script's ``UNET_MODEL_PATH`` is resolved relative to its own
    # ``__file__`` parent if it's not absolute.  Be explicit.
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    # The legacy script's loader handles two formats:
    #   (a) ``{model_state_dict, model_config, ...}`` — the 2025 trainer format.
    #   (b) a bare ``state_dict``.
    # The current ``reconunet-train`` writes ``{model, epoch, cfg, ...}``, so
    # convert it into format (a) once into a tempfile and feed the legacy
    # script the converted path.  Keeps the legacy script byte-identical.
    import torch
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        legacy_ckpt_path = checkpoint_path
    elif isinstance(raw, dict) and "model" in raw:
        cfg = raw.get("cfg") or {}
        init = (cfg.get("model") or {}).get("init") or {}
        # EVDCovarianceReconstructionUNet defaults if cfg is missing fields.
        init.setdefault("M", 8)
        init.setdefault("tau", 8)
        init.setdefault("activation_type", "anti_rectifier")
        init.setdefault("use_dropout", True)
        converted = {
            "model_state_dict": raw["model"],
            "model_config":     init,
            "epoch":            int(raw.get("epoch", -1)),
            "val_loss":         float(raw.get("val_loss", float("nan"))),
        }
        legacy_ckpt_path = (args.output_dir / "checkpoint_legacy_format.pt").resolve()
        torch.save(converted, str(legacy_ckpt_path))
        print(f"[wrapper] converted reconunet-train checkpoint → "
              f"legacy format at {legacy_ckpt_path}")
    else:
        # Bare state_dict.  The legacy loader handles this directly.
        legacy_ckpt_path = checkpoint_path
    checkpoint_path = legacy_ckpt_path

    # The legacy ``ControlledDatasetGenerator`` writes its temp datasets
    # under cwd / "Data/datasets".  Run from the repo root so this lands in
    # a predictable location.
    os.chdir(repo_root)

    legacy = _import_legacy(repo_root)

    # The model the legacy script builds (line 1278:
    # ``unet_model = EVDCovarianceReconstructionUNet(**model_params)``) needs
    # to apply input/output conjugation to bridge the training vs legacy
    # steering-vector conventions.  Replace the class reference in the
    # legacy module with a subclass that does this transparently — the
    # state-dict loader still works because the subclass inherits all
    # parameters and submodules unchanged.
    legacy.EVDCovarianceReconstructionUNet = _make_bridged_model_class(
        legacy.EVDCovarianceReconstructionUNet
    )

    scenarios = SCENARIOS
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in SCENARIOS if s["name"] in wanted]

    # ---- Run all scenarios -----------------------------------------------
    json_paths: Dict[str, Path] = {}
    for s in scenarios:
        print("\n" + "=" * 72)
        print(f"  Scenario: {s['name']}  (mode={s['mode']}, "
              f"sources={s['num_sources']}, mp={s['num_multipath']}, "
              f"imperfections={s['imperfections']})")
        print("=" * 72)
        json_paths[s["name"]] = _run_one(
            legacy, s, checkpoint_path, args.output_dir,
            samples_per_snr=int(args.samples_per_snr),
            snr_levels=snr_levels,
            dataset_seed=int(args.dataset_seed),
        )

    # ---- Combined Table-II at 0 dB ---------------------------------------
    rows = []
    for s in scenarios:
        with json_paths[s["name"]].open() as fh:
            d = json.load(fh)
        # The legacy script names UNet's variant ``ReconUNet_ESPRIT`` (paper
        # checkpoint) but earlier versions used ``UNet_ESPRIT``.  Cover both.
        for method in d.get("0", {}).keys():
            rows.append({
                "scenario": s["name"],
                "method":   method,
                "rmse_deg_at_0dB": _rmse_at(d, 0, method),
            })

    summary_csv = args.output_dir / "summary_0dB.csv"
    with summary_csv.open("w") as fh:
        fh.write("scenario,method,rmse_deg_at_0dB\n")
        for r in rows:
            fh.write(f"{r['scenario']},{r['method']},{r['rmse_deg_at_0dB']}\n")
    print(f"\n[summary] wrote {summary_csv}")

    # ---- Pretty pivot to stdout ------------------------------------------
    methods_seen = sorted({r["method"] for r in rows})
    scen_names = [s["name"] for s in scenarios]
    print("\n=== RMSE (deg) at 0 dB across scenarios — legacy script ===")
    header = f"{'method':30s}  " + "  ".join(f"{n:>14s}" for n in scen_names)
    print(header)
    print("-" * len(header))
    for m in methods_seen:
        cells = []
        for n in scen_names:
            match = next((r for r in rows
                          if r["method"] == m and r["scenario"] == n), None)
            cells.append(f"{match['rmse_deg_at_0dB']:14.3f}" if match
                         else f"{'-':>14s}")
        print(f"{m:30s}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
