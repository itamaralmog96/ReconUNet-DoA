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
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import yaml

from reconunet.evaluation.unified_harness import (
    HarnessConfig,
    ModelSpec,
    evaluate,
    plot_rmse_vs_snr,
)

LOG = logging.getLogger("reconunet.evaluate")


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def _build_model_specs(models: list[dict], root: Path) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for m in models:
        name = str(m["name"])
        ckpt = _resolve(m["checkpoint"], root) if "checkpoint" in m else None
        is_classic = bool(m.get("is_classic", False))
        if "class_path" in m:
            adapter = m["class_path"]  # harness handles dotted paths uniformly
        elif "adapter_class" in m:
            adapter = m["adapter_class"]
        else:
            raise ValueError(f"Model {name!r}: must specify class_path or adapter_class")
        out.append(ModelSpec(
            name=name,
            adapter=adapter,
            checkpoint=str(ckpt) if ckpt else None,
            init_cfg=dict(m.get("init", {})),
            collate=m.get("collate"),
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
