"""Single-entry-point evaluator for the paper's three-way comparison.

Given one :class:`SceneManifest` and a list of ``(name, adapter, checkpoint)``
triples, this harness produces one tidy ``pandas.DataFrame`` of per-model,
per-SNR angle-error metrics — everything the paper's Fig. 4-6 need.

Why this file exists
--------------------
The legacy evaluation lived in notebook cells with model-specific paths
hard-coded.  Reproducing the three-way plot required re-executing three
different notebooks on three different datasets that weren't guaranteed to
be identical.  The unified harness solves that:

* **One manifest**: every model is evaluated on the exact same scenes.
* **One loop**: the harness pipes each batch through every registered model
  and records results in a single DataFrame.
* **One output**: a CSV per experiment and a set of publication-ready
  matplotlib figures.

The harness is deliberately thin — model instantiation, checkpoint loading,
and output normalization all happen through
:class:`~reconunet.models.third_party._base.BaselineAdapter`.  Adding a new
baseline is a new adapter, not a new harness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Metric: Root-Mean-Square Periodic Error (RMSPE)
# ---------------------------------------------------------------------------


def rmspe_deg(
    pred_rad: np.ndarray,      # [B, K] predicted angles
    true_rad: np.ndarray,      # [B, K] ground-truth angles (sorted ascending)
    reduce: str = "mean",
) -> np.ndarray:
    """Permutation-invariant root-mean-square error on an SO(2) manifold.

    Predictions and ground truth are both sorted ascending before the
    comparison — equivalent to Hungarian matching when ``K ≤ 3`` because the
    ground truth is already well-separated by ``min_separation_deg``.  For
    larger K, swap in :func:`scipy.optimize.linear_sum_assignment`.
    """
    assert pred_rad.shape == true_rad.shape, (pred_rad.shape, true_rad.shape)
    pred = np.sort(pred_rad, axis=-1)
    true = np.sort(true_rad, axis=-1)
    # Wrap difference into (-π/2, π/2] to make it periodic.
    diff = pred - true
    diff = (diff + np.pi / 2.0) % np.pi - np.pi / 2.0
    err = np.rad2deg(diff) ** 2
    per_sample = np.sqrt(np.mean(err, axis=-1))
    if reduce == "mean":
        return np.mean(per_sample)
    if reduce == "none":
        return per_sample
    raise ValueError(f"Unknown reduction {reduce!r}")


# ---------------------------------------------------------------------------
# Monte-Carlo Cramér-Rao Lower Bound (for the overlay curve)
# ---------------------------------------------------------------------------


def stochastic_crlb_deg(
    M: int,
    T: int,
    angles_rad: np.ndarray,
    snr_db: float,
    element_spacing_lambda: float = 0.5,
) -> float:
    """Stochastic CRLB on DoA variance for K Gaussian sources, one ULA,
    returned in **degrees²**.

    Implements the standard Stoica–Nehorai result (IEEE TASSP 1990, §III).
    The returned scalar is the per-source variance floor in deg²; the paper
    plots :math:`\\sqrt{\\text{CRLB}}` in degrees.

    This is a minimal, readable implementation; for K > 3 the matrix inverse
    becomes the bottleneck and a closed-form ULA approximation should be used
    instead.
    """
    K = angles_rad.size
    snr = 10.0 ** (snr_db / 10.0)
    m = np.arange(M, dtype=np.float64)[:, None]
    a = np.exp(-1j * 2 * np.pi * element_spacing_lambda * m * np.sin(angles_rad[None, :]))
    d = (-1j * 2 * np.pi * element_spacing_lambda
         * np.cos(angles_rad[None, :]) * m) * a
    P_a_perp = np.eye(M) - a @ np.linalg.pinv(a.conj().T @ a) @ a.conj().T
    J = 2.0 * snr * T * np.real((d.conj().T @ P_a_perp @ d))
    # Per-source variance in rad² → deg²
    var_rad2 = np.real(np.linalg.inv(J).diagonal())
    return float(np.mean(np.rad2deg(np.sqrt(var_rad2)) ** 2))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Declarative description of one model slot in the harness."""

    name: str
    adapter: Any                # BaselineAdapter instance OR classic-algorithm callable
    checkpoint: Optional[str] = None
    init_cfg: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    is_classic: bool = False    # True for MUSIC, MVDR, Root-MUSIC, etc.


@dataclass
class HarnessConfig:
    manifest_path: str                      # data/scenes/test.npy
    snr_sweep_db: Sequence[float] = (-10, -5, 0, 5, 10, 15, 20)
    models: Sequence[ModelSpec] = ()
    batch_size: int = 256
    device: str = "cpu"
    output_dir: str = "experiments/paper_figures/three_way_comparison"
    seed: int = 20260420


def evaluate(config: HarnessConfig) -> "pandas.DataFrame":   # noqa: F821
    """Run every model against every SNR bucket and return a long-form frame.

    Returns
    -------
    DataFrame with columns::

        model          str
        snr_db         float
        rmse_deg       float      (RMSPE averaged over the SNR bucket)
        crlb_deg       float      (√(mean per-source CRLB) in degrees)
        n_samples      int
        latency_ms     float      (per-batch inference wall time)
    """
    import pandas as pd                 # local import — optional dep
    import torch
    from torch.utils.data import DataLoader

    from reconunet.data.scene_dataset import SceneDataset, get_collate
    from reconunet.data.scene_manifest import ManifestMeta, SceneManifest

    manifest = SceneManifest.load(config.manifest_path)
    meta = manifest.meta
    dataset = SceneDataset(manifest)

    rows: List[Dict[str, Any]] = []
    torch.manual_seed(config.seed)

    # ----------------- bucket scenes by SNR -------------------------------
    snr_edges = np.asarray(config.snr_sweep_db, dtype=np.float32)
    all_snr = manifest.raw["snr_db"]
    buckets: Dict[float, np.ndarray] = {
        float(edge): np.where(np.abs(all_snr - edge) < 1.0)[0]
        for edge in snr_edges
    }

    # ----------------- per-model loop --------------------------------------
    for spec in config.models:
        collate_fn = get_collate(_collate_for(spec), meta)

        if spec.is_classic:
            model = spec.adapter                           # callable(X) → angles
        else:
            model = spec.adapter.build_model(spec.init_cfg).to(spec.device)
            model.eval()
            if spec.checkpoint:
                spec.adapter.load_checkpoint(model, spec.checkpoint)

        for snr, idx in buckets.items():
            if idx.size == 0:
                continue
            subset = torch.utils.data.Subset(dataset, idx.tolist())
            loader = DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
            errs: List[float] = []
            crlbs: List[float] = []
            t0 = time.perf_counter()
            with torch.no_grad():
                for batch in loader:
                    if spec.is_classic:
                        pred_rad = spec.adapter(batch, meta)    # [B, K] radians
                    else:
                        x = batch["input"].to(spec.device)
                        out = spec.adapter.forward(model, x, meta=_meta_dict(meta, spec))
                        pred_rad = out.angles_pred.detach().cpu().numpy()
                    true_rad = batch["angles_rad"].numpy()
                    K = int(batch["n_sources"][0].item())
                    true_rad = true_rad[:, :K]
                    pred_rad = pred_rad[:, :K]
                    errs.append(rmspe_deg(pred_rad, true_rad, reduce="mean"))
                    for row_true in true_rad:
                        crlbs.append(
                            stochastic_crlb_deg(
                                M=meta.M, T=meta.T,
                                angles_rad=row_true,
                                snr_db=snr,
                                element_spacing_lambda=meta.element_spacing_lambda,
                            )
                        )
            t1 = time.perf_counter()
            rows.append(
                {
                    "model":      spec.name,
                    "snr_db":     snr,
                    "rmse_deg":   float(np.mean(errs)),
                    "crlb_deg":   float(np.sqrt(np.mean(crlbs))),
                    "n_samples":  int(idx.size),
                    "latency_ms": 1e3 * (t1 - t0) / max(1, len(loader)),
                }
            )

    df = pd.DataFrame.from_records(rows)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "manifest_fingerprint": manifest.fingerprint(),
                "snr_sweep_db": list(config.snr_sweep_db),
                "models": [
                    {"name": s.name, "checkpoint": s.checkpoint, "is_classic": s.is_classic}
                    for s in config.models
                ],
            },
            indent=2,
        )
    )
    return df


# ---------------------------------------------------------------------------
# Publication plots
# ---------------------------------------------------------------------------


def plot_rmse_vs_snr(df: "pandas.DataFrame", output_path: str) -> None:   # noqa: F821
    """Stylized RMSE-vs-SNR plot with CRLB floor overlay, as in the paper."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, sub in df.sort_values("snr_db").groupby("model"):
        ax.plot(sub["snr_db"], sub["rmse_deg"], marker="o", label=name)
    crlb = df.groupby("snr_db")["crlb_deg"].mean().sort_index()
    ax.plot(crlb.index, crlb.values, "k--", linewidth=1.0, label="CRLB")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSPE (deg)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collate_for(spec: ModelSpec) -> str:
    if spec.is_classic:
        return "reconunet"        # classics only need labels; any collate works
    return spec.adapter.name      # adapters carry a canonical name


def _meta_dict(meta, spec: ModelSpec) -> Dict[str, Any]:
    out = {
        "M":                 meta.M,
        "T":                 meta.T,
        "tau":               meta.tau,
        "element_spacing":   meta.element_spacing_lambda,
        "K_max":             meta.K_max,
    }
    out.update(spec.init_cfg)
    return out


__all__ = [
    "HarnessConfig",
    "ModelSpec",
    "evaluate",
    "plot_rmse_vs_snr",
    "rmspe_deg",
    "stochastic_crlb_deg",
]
