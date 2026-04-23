"""``reconunet-evaluate`` — CLI wrapper around :mod:`unified_harness`.

Reads an evaluation YAML config that enumerates a list of trained model
checkpoints to benchmark, the test-split manifest to benchmark *on*, and
the SNR sweep to bucket by.  Writes ``results.csv`` + ``rmse_vs_snr.png``
into the output directory.

YAML schema::

    manifest: data/scenes/test.npy
    output_dir: experiments/runs/eval_2026-04-21/
    snr_sweep_db: [-10, -5, 0, 5, 10, 15, 20]
    batch_size: 256
    models:
      - name: reconunet
        class_path: reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet
        init: { M: 8, tau: 8 }
        checkpoint: experiments/runs/reconunet/checkpoints/best.pt
        collate: reconunet
      - name: subspacenet
        adapter_class: reconunet.models.third_party.subspacenet_adapter.SubspaceNetAdapter
        init: { M: 3, tau: 8, diff_method: root_music }
        checkpoint: experiments/runs/subspacenet/checkpoints/best.pt
        collate: subspacenet
      - name: subvit
        adapter_class: reconunet.models.third_party.subvit_adapter.SubViTAdapter
        init: { M: 8, K_max: 3, grid_size: 121 }
        checkpoint: experiments/runs/subvit/checkpoints/best.pt
        collate: subvit
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from reconunet.evaluation.unified_harness import (
    HarnessConfig,
    ModelSpec,
    evaluate,
    plot_rmse_vs_snr,
)
from reconunet.models.third_party._base import BaselineAdapter, BaselineOutput

LOG = logging.getLogger("reconunet.evaluate")


def _import_dotted(dotted: str):
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {dotted!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class _NativeEVDUNetAdapter(BaselineAdapter):
    """Wrap a native ``EVDCovarianceReconstructionUNet`` so the unified
    harness can treat it like any other adapter.  Angle extraction runs a
    classical MUSIC peak-search on the model's reconstructed covariance.
    """

    name = "reconunet"

    def __init__(self, class_path: str, K: int = 3,
                 grid_size: int = 361, angle_range_deg: tuple = (-90.0, 90.0)):
        self._cls = _import_dotted(class_path)
        self.K = int(K)
        self.grid_size = int(grid_size)
        self.angle_range_deg = tuple(angle_range_deg)

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        # Drop CLI-only knobs that aren't model __init__ args.
        init = {k: v for k, v in cfg.items() if k not in {"K", "grid_size", "angle_range_deg"}}
        return self._cls(**init)

    def prepare_input(self, snapshots, meta):
        from reconunet.data.scene_renderer import lag_stack
        return lag_stack(snapshots, tau=int(meta.get("tau", 8)))

    def forward(self, model, prepped, targets=None, meta=None) -> BaselineOutput:
        out = model(prepped)
        if isinstance(out, tuple) and len(out) == 3:
            _eigvals, _eigvecs, K_recon = out
        else:
            K_recon = out
        # Number of sources to extract.  ``K_max`` from the manifest is the
        # *upper bound* across the corpus (= 8), not the actual K per scene
        # (= 3 for the paper config).  Picking K_max here would shrink the
        # noise sub-space to {0}, breaking Root-MUSIC.  Use the adapter's
        # configured K (overridable via init_cfg["K"]).
        K = int((meta or {}).get("K", self.K))
        # Use the exact same Root-MUSIC routine the training loop uses for
        # val_rmspe, so eval numbers match training-time reporting.
        from reconunet.models.deep_learning.subspace_models import root_music
        angles_deg, _, _ = root_music(K_recon, K, K_recon.shape[0])
        # ``root_music`` returns degrees in [0°, 180°] (broadside = 90°);
        # subtract 90° to put broadside at 0°, matching the manifest's
        # angles_rad convention.
        angles_rad = torch.deg2rad(angles_deg - 90.0)
        return BaselineOutput(angles_pred=angles_rad, extras={"K_recon": K_recon})

    def _music_peak_pick(self, K_recon: torch.Tensor, K: int, M: int) -> torch.Tensor:
        """Classical MUSIC on a batch of complex covariance matrices."""
        device = K_recon.device
        # Eigendecomposition; eigh returns ascending order.
        w, V = torch.linalg.eigh(K_recon)
        # Noise subspace = eigenvectors with the M-K smallest eigenvalues.
        noise_dim = M - K
        En = V[..., :noise_dim]                                  # [B, M, M-K]

        g0, g1 = self.angle_range_deg
        grid = torch.linspace(g0, g1, self.grid_size, device=device)
        grid_rad = torch.deg2rad(grid)
        m = torch.arange(M, device=device).view(1, M, 1)
        # Steering matrix A: [grid_size, M]
        A = torch.exp(-1j * np.pi * m * torch.sin(grid_rad).view(1, 1, -1)).squeeze(0)
        A = A.transpose(0, 1)                                    # [grid_size, M]
        # P(theta) = 1 / || En^H a(theta) ||^2
        # Compute En^H @ A.T  → [B, M-K, grid_size]
        proj = En.conj().transpose(-1, -2) @ A.T.unsqueeze(0)    # [B, M-K, grid_size]
        denom = (proj.abs() ** 2).sum(dim=1)                     # [B, grid_size]
        spectrum = 1.0 / (denom + 1e-12)
        _, idx = torch.topk(spectrum, k=K, dim=-1)               # [B, K]
        peaks_deg, _ = torch.sort(grid[idx], dim=-1)
        return torch.deg2rad(peaks_deg)


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def _build_model_specs(models: list[dict], root: Path) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for m in models:
        name = str(m["name"])
        ckpt = _resolve(m["checkpoint"], root) if "checkpoint" in m else None
        is_classic = bool(m.get("is_classic", False))
        init = dict(m.get("init", {}))
        if "class_path" in m:
            adapter = _NativeEVDUNetAdapter(m["class_path"])
            adapter.name = name
        elif "adapter_class" in m:
            adapter_cls = _import_dotted(m["adapter_class"])
            adapter = adapter_cls(**init)
            adapter.name = name
        else:
            raise ValueError(f"Model {name!r}: must specify class_path or adapter_class")
        out.append(ModelSpec(
            name=name,
            adapter=adapter,
            checkpoint=str(ckpt) if ckpt else None,
            init_cfg=init,
            is_classic=is_classic,
        ))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reconunet-evaluate",
        description="Benchmark one or more trained models against the test manifest.",
    )
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="Path to an evaluation YAML (see docstring).")
    parser.add_argument("--plot/--no-plot", dest="plot", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Emit rmse_vs_snr.png alongside results.csv.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    cfg_path: Path = args.config.resolve()
    with cfg_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    project_root = cfg_path.parents[2]
    manifest_path = _resolve(cfg["manifest"], project_root)
    output_dir = _resolve(cfg["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _build_model_specs(cfg["models"], project_root)

    harness = HarnessConfig(
        manifest_path=str(manifest_path),
        snr_sweep_db=tuple(float(x) for x in cfg["snr_sweep_db"]),
        models=models,
        batch_size=int(cfg.get("batch_size", 128)),
        output_dir=str(output_dir),
        seed=int(cfg.get("seed", 20260420)),
    )

    LOG.info("Running harness on %s with %d model(s) across %d SNR buckets",
             manifest_path.name, len(models), len(harness.snr_sweep_db))
    df = evaluate(harness)
    LOG.info("Wrote %s  (%d rows)", output_dir / "results.csv", len(df))

    if args.plot:
        out_png = output_dir / "rmse_vs_snr.png"
        plot_rmse_vs_snr(df, str(out_png))
        LOG.info("Wrote %s", out_png)

    # Pretty-print a small summary table to stdout.
    pivot = df.pivot_table(index="snr_db", columns="model",
                           values="rmse_deg", aggfunc="mean")
    print(pivot.to_string(float_format=lambda x: f"{x:6.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
