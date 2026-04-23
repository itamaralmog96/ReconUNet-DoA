"""``reconunet-train`` — single entry-point that trains any of the three
paper baselines (ReconUNet, SubspaceNet, SubViT) from one YAML config.

Design decisions:

* **One loop, three models.**  The config declares *either* ``class_path``
  (for native ``reconunet.models.*`` modules) *or* ``adapter_class`` (for
  third-party submodules wrapped by a :class:`BaselineAdapter`).  The loop
  is identical in both branches.
* **Dataset is always** :class:`SceneDataset` + the model-specific collate
  from :func:`reconunet.data.scene_dataset.get_collate` — so the fair-
  comparison guarantee (byte-identical scenes across models) holds by
  construction.
* **Determinism** is enforced via a master seed in the config; NumPy +
  PyTorch + CUDA RNGs are all derived from it.  The manifest itself was
  already seeded at generation time.
* **Checkpoints** are saved on best-val-rmspe, keyed by (model, epoch).
  Last-epoch checkpoint is always overwritten as ``last.pt`` for resume.
* **AMP** is on for CUDA by default and auto-disabled on CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from reconunet.data.scene_dataset import SceneDataset, get_collate
from reconunet.data.scene_manifest import ManifestMeta, SceneManifest
from reconunet.data.scene_renderer import SceneRenderer
from reconunet.evaluation.unified_harness import rmspe_deg
from reconunet.models.third_party import BaselineAdapter, BaselineOutput

LOG = logging.getLogger("reconunet.train")


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _import_dotted(dotted: str):
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {dotted!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Trade a little throughput for reproducibility across runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _build_loaders(train_cfg: dict, project_root: Path) -> tuple[DataLoader, DataLoader, ManifestMeta]:
    data_cfg_path = project_root / train_cfg["data"]["config"]
    data_cfg = _load_yaml(data_cfg_path)

    man_cfg = data_cfg["manifest"]
    train_path = project_root / man_cfg["train_path"]
    val_path = project_root / man_cfg["val_path"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train manifest not found: {train_path}\n"
            f"Run `reconunet-generate --config {data_cfg_path}` first."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Val manifest not found: {val_path}\n"
            f"Run `reconunet-generate --config {data_cfg_path}` first."
        )

    train_manifest = SceneManifest.load(str(train_path))
    val_manifest = SceneManifest.load(str(val_path))
    meta = train_manifest.meta
    renderer = SceneRenderer(meta)

    train_ds = SceneDataset(train_manifest, renderer=renderer)
    val_ds = SceneDataset(val_manifest, renderer=renderer)

    collate = get_collate(train_cfg["collate"], meta)

    batch_size = int(train_cfg["data"].get("batch_size", 128))
    num_workers = int(train_cfg["data"].get("num_workers", 4))
    pin_memory = bool(train_cfg["data"].get("pin_memory", True))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=pin_memory,
        collate_fn=collate, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    LOG.info("Data ready: train=%d val=%d  batch_size=%d collate=%s",
             len(train_ds), len(val_ds), batch_size, train_cfg["collate"])
    return train_loader, val_loader, meta


# ---------------------------------------------------------------------------
# Model + adapter indirection
# ---------------------------------------------------------------------------


class _AdapterModule(nn.Module):
    """Wrap a :class:`BaselineAdapter` so it behaves like a plain nn.Module.

    This lets the training loop call ``model(batch)`` identically whether
    the user configured a native model or a third-party adapter.
    """

    def __init__(self, adapter: BaselineAdapter, meta: ManifestMeta,
                 build_cfg: dict[str, Any] | None = None):
        super().__init__()
        self._inner = adapter.build_model(dict(build_cfg or {}))
        self.adapter = adapter
        self.meta = meta
        # Build a plain dict view of meta for adapter consumers.
        self._meta_dict = dataclasses.asdict(meta) if dataclasses.is_dataclass(meta) else dict(meta)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self._inner.parameters(recurse=recurse)

    def forward(self, batch: dict[str, Any]) -> BaselineOutput:
        # The collate has already produced the model-specific input tensor
        # under batch["input"].  Pass it through to the adapter.
        prepped = batch["input"]
        targets = batch.get("angles_rad")
        out = self.adapter.forward(self._inner, prepped, targets=targets,
                                   meta=self._meta_dict)
        return out


class _NativeModule(nn.Module):
    """Thin wrapper that calls a native model on ``batch["input"]`` and
    normalises its tuple output into a dict keyed for the loss config.

    Currently understands the EVDUNet-family return signature
    ``(eigvals, eigvecs, K_recon)``.  Other native models can be added by
    extending the dict-construction below.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self._inner = model

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self._inner.parameters(recurse=recurse)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = self._inner(batch["input"])
        if isinstance(out, tuple) and len(out) == 3:
            eigvals, eigvecs, K_recon = out
            return {
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "K_recon": K_recon,
            }
        return out


def _build_model(model_cfg: dict, meta: ManifestMeta) -> nn.Module:
    if "class_path" in model_cfg and "adapter_class" in model_cfg:
        raise ValueError("Config must specify exactly one of class_path / adapter_class")
    if "class_path" in model_cfg:
        cls = _import_dotted(model_cfg["class_path"])
        init = dict(model_cfg.get("init", {}))
        model = cls(**init)
        LOG.info("Built native model %s with %d params",
                 model_cfg["class_path"], sum(p.numel() for p in model.parameters()))
        return _NativeModule(model)
    if "adapter_class" in model_cfg:
        cls = _import_dotted(model_cfg["adapter_class"])
        init = dict(model_cfg.get("init", {}))
        adapter = cls(**init)
        module = _AdapterModule(adapter, meta, build_cfg=init)
        LOG.info("Built adapter %s with %d params",
                 model_cfg["adapter_class"],
                 sum(p.numel() for p in module.parameters()))
        return module
    raise ValueError("Model config needs class_path or adapter_class")


# ---------------------------------------------------------------------------
# Optimiser + scheduler + loss
# ---------------------------------------------------------------------------


def _build_optimizer(params, optim_cfg: dict) -> torch.optim.Optimizer:
    name = optim_cfg.get("optimizer", "adam").lower()
    lr = float(optim_cfg["lr"])
    wd = float(optim_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def _build_scheduler(opt, sch_cfg: dict | None):
    if not sch_cfg:
        return None
    name = sch_cfg.get("name", "").lower()
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=sch_cfg.get("mode", "min"),
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 4)),
            min_lr=float(sch_cfg.get("min_lr", 1e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(sch_cfg["step_size"]),
            gamma=float(sch_cfg.get("gamma", 0.5)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(sch_cfg["T_max"]),
            eta_min=float(sch_cfg.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unknown scheduler: {name!r}")


def _compute_loss_adapter(
    out: BaselineOutput,
    batch: dict[str, Any],
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Adapters either return their own loss or we apply RMSPE."""
    if out.loss is not None:
        return out.loss, {"loss": float(out.loss.detach())}
    angles_true = batch["angles_rad"]
    if not isinstance(angles_true, torch.Tensor):
        angles_true = torch.as_tensor(angles_true)
    angles_true = angles_true.to(out.angles_pred.device)

    # If the adapter exposes a differentiable spatial spectrum (e.g. SubViT),
    # use multi-hot BCE against the ground-truth angle bins — angles_pred
    # itself comes from a non-differentiable topk and cannot drive grads.
    spectrum = (out.extras or {}).get("spatial_spectrum")
    if spectrum is not None and spectrum.requires_grad:
        bce_w = float(loss_cfg.get("bce_weight", 1.0))
        grid = spectrum.shape[-1]
        # Reuse the angle range from the source manifest if encoded into the
        # batch metadata; otherwise default to (-60°, 60°).
        g0_deg, g1_deg = -60.0, 60.0
        targets = torch.zeros_like(spectrum)
        a_deg = torch.rad2deg(angles_true)
        valid = ~torch.isnan(a_deg)
        idx = ((a_deg - g0_deg) / (g1_deg - g0_deg) * (grid - 1)).round().long()
        idx = idx.clamp(0, grid - 1)
        rows = torch.arange(spectrum.shape[0], device=spectrum.device)
        for k in range(a_deg.shape[1]):
            mask = valid[:, k]
            if mask.any():
                targets[rows[mask], idx[mask, k]] = 1.0
        loss = bce_w * torch.nn.functional.binary_cross_entropy_with_logits(
            spectrum, targets
        )
        return loss, {"loss": float(loss.detach())}

    # ``angles_true`` is padded to K_MAX with NaN; slice to the prediction
    # width (assumes uniform k per batch, which holds for k_choices=[K]).
    K = out.angles_pred.shape[-1]
    angles_true = angles_true[..., :K]
    # Periodic RMSPE in radians — direct drop-in for training loss.
    diff = out.angles_pred.sort(dim=-1).values - angles_true.sort(dim=-1).values
    wrapped = (diff + math.pi / 2) % math.pi - math.pi / 2
    mse = (wrapped ** 2).mean()
    weight = float(loss_cfg.get("rmspel_weight", 1.0))
    loss = weight * torch.sqrt(mse + 1e-12)
    return loss, {"loss": float(loss.detach()), "rmspe_rad": float(torch.sqrt(mse).detach())}


def _compute_loss_native(
    out: Any,
    batch: dict[str, Any],
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Native models: sum of weighted terms keyed in the output dict.

    Native models are expected to return either:

    * an ``nn.Module.forward`` output of type dict with keys that match
      loss_cfg entries (e.g. 'eigval_loss', 'eigvec_loss', ...), **or**
    * a dict with a single 'loss' key (already reduced).
    """
    if isinstance(out, torch.Tensor):
        return out, {"loss": float(out.detach())}
    if isinstance(out, dict) and "loss" in out:
        return out["loss"], {"loss": float(out["loss"].detach())}
    if not isinstance(out, dict):
        raise TypeError(f"Native model must return Tensor or dict, got {type(out)}")

    metrics: dict[str, float] = {}
    total: torch.Tensor | None = None

    # ---- EVDUNet-style outputs: derive standard loss terms on demand. ----
    K_true = batch.get("covariance")
    if "K_recon" in out and K_true is not None:
        K_recon = out["K_recon"]
        K_true_dev = K_true.to(K_recon.device)
        diff = K_recon - K_true_dev
        # Frobenius reconstruction loss, normalised per-sample.
        recon = (diff.abs() ** 2).sum(dim=(-1, -2)).mean()
        out = {**out, "reconstruction": recon}
        if "eigvals" in out and "eigvecs" in out:
            # Synthesise eigval / eigvec supervision from the EVD of K_true so
            # the configured ``eigval_weight`` / ``eigvec_weight`` are usable
            # without an explicit ground-truth EVD field in the manifest.
            with torch.no_grad():
                w_true, V_true = torch.linalg.eigh(K_true_dev)
                w_true = torch.flip(w_true, dims=[-1])         # descending
                V_true = torch.flip(V_true, dims=[-1])
            eigvals_pred = out["eigvals"]
            eigvecs_pred = out["eigvecs"]
            out = {
                **out,
                "eigval": torch.nn.functional.mse_loss(eigvals_pred, w_true.to(eigvals_pred.dtype)),
                # Subspace distance: 1 - mean |<v_pred, v_true>|^2.
                "eigvec": (1.0 - (eigvecs_pred.conj() * V_true).sum(dim=-2).abs().pow(2).mean()).real,
            }

    for key, weight in loss_cfg.items():
        term_key = key.removesuffix("_weight")
        if term_key in out:
            term = out[term_key]
            w = float(weight)
            metrics[term_key] = float(term.detach())
            total = term * w if total is None else total + term * w
    if total is None:
        raise RuntimeError(
            f"None of the loss keys {list(loss_cfg)} matched the model output "
            f"keys {list(out.keys())}."
        )
    metrics["loss"] = float(total.detach())
    return total, metrics


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _extract_pred_angles(out: Any, batch: dict[str, Any]) -> torch.Tensor | None:
    if isinstance(out, BaselineOutput):
        return out.angles_pred
    if isinstance(out, dict) and "angles_pred" in out:
        return out["angles_pred"]
    # Native EVDUNet-style output: derive angles from reconstructed covariance
    # via Root-MUSIC so val_rmspe reports a real number.  Requires uniform K
    # across the batch (holds for k_choices=[K]); root_music returns degrees
    # in [0°, 180°] (acos-of-sin convention), so we subtract 90° and convert
    # to radians to match the BaselineOutput.angles_pred convention.
    if isinstance(out, dict) and "K_recon" in out:
        from reconunet.models.deep_learning.subspace_models import root_music
        K_recon = out["K_recon"]
        angles_true = batch["angles_rad"]
        K = int((~torch.isnan(angles_true[0])).sum().item())
        if K == 0:
            return None
        angles_deg, _, _ = root_music(K_recon, K, K_recon.shape[0])
        return torch.deg2rad(angles_deg - 90.0)
    return None


def train_loop(cfg: dict, project_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Device: %s", device)

    seed = int(cfg.get("seed", 20260420))
    _seed_everything(seed)

    train_loader, val_loader, meta = _build_loaders(cfg, project_root)
    model = _build_model(cfg["model"], meta).to(device)
    is_adapter = isinstance(model, _AdapterModule)
    # Native EVDUNet also uses complex tensors; AMP must be off for it.
    has_complex = is_adapter or isinstance(model, _NativeModule)

    optim = _build_optimizer(model.parameters(), cfg["optim"])
    sched = _build_scheduler(optim, cfg["optim"].get("scheduler"))
    grad_clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    if has_complex and amp_enabled:
        # Third-party adapters operate on complex tensors (covariance / lag
        # stack); CUDA's ComplexHalf path is not implemented for matmul, so
        # AMP must be disabled for them.
        LOG.info("Disabling AMP for complex-tensor model.")
        amp_enabled = False
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    loss_cfg = dict(cfg.get("loss", {}))
    loss_fn = _compute_loss_adapter if is_adapter else _compute_loss_native

    ckpt_dir = project_root / cfg["train"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = project_root / cfg["train"]["tensorboard_dir"]
    tb_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tb_dir))
    except Exception:                                           # pragma: no cover
        writer = None
        LOG.warning("TensorBoard unavailable — continuing without logging")

    best_val = math.inf
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stopping_patience", 10**9))
    stale = 0
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="epoch", leave=True)
    for epoch in epoch_bar:
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        train_bar = tqdm(train_loader, desc=f"ep {epoch:>3}/{epochs} train",
                         unit="batch", leave=False)
        for i, batch in enumerate(train_bar):
            batch = _to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(batch)
                loss, _metrics = loss_fn(out, batch, loss_cfg)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            running += float(loss.detach())
            n_batches += 1
            train_bar.set_postfix(loss=f"{running / n_batches:.4f}")
        train_bar.close()
        train_loss = running / max(1, n_batches)

        # ---- validation -----------------------------------------------------
        model.eval()
        val_rmspe_sum = 0.0
        val_count = 0
        val_loss_sum = 0.0
        val_samples = 0
        val_bar = tqdm(val_loader, desc=f"ep {epoch:>3}/{epochs} val  ",
                       unit="batch", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                batch = _to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(batch)
                    loss, _m = loss_fn(out, batch, loss_cfg)
                n_in_batch = batch["angles_rad"].shape[0]
                val_loss_sum += float(loss.detach()) * n_in_batch
                val_samples += n_in_batch
                pred = _extract_pred_angles(out, batch)
                if pred is not None:
                    K = pred.shape[-1]
                    angles_true_np = batch["angles_rad"][..., :K].detach().cpu().numpy()
                    angles_pred_np = pred.detach().cpu().numpy()
                    per_sample = rmspe_deg(angles_pred_np, angles_true_np, reduce="none")
                    val_rmspe_sum += float(per_sample.sum())
                    val_count += per_sample.shape[0]
                running_val_loss = val_loss_sum / max(1, val_samples)
                running_rmspe = (val_rmspe_sum / val_count) if val_count else float("nan")
                val_bar.set_postfix(loss=f"{running_val_loss:.4f}",
                                    rmspe=f"{running_rmspe:.2f}°")
        val_bar.close()
        val_loss = val_loss_sum / max(1, len(val_loader.dataset))
        val_rmspe = val_rmspe_sum / max(1, val_count) if val_count else float("nan")

        dt = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]
        tqdm.write(
            f"epoch {epoch:>3}/{epochs}  train={train_loss:.4f}  "
            f"val={val_loss:.4f}  val_rmspe={val_rmspe:.3f}°  "
            f"lr={lr_now:.2e}  [{dt:.1f}s]"
        )
        epoch_bar.set_postfix(train=f"{train_loss:.4f}",
                              val=f"{val_loss:.4f}",
                              rmspe=f"{val_rmspe:.2f}°",
                              lr=f"{lr_now:.1e}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metric/val_rmspe_deg", val_rmspe, epoch)
            writer.add_scalar("optim/lr", lr_now, epoch)

        history.append(dict(epoch=epoch, train=train_loss, val=val_loss,
                            val_rmspe_deg=val_rmspe, lr=lr_now))

        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(val_loss)
        elif sched is not None:
            sched.step()

        # Persist the inner model's state_dict so it can be reloaded by
        # plain `Model(**init).load_state_dict(...)` without the training
        # wrapper.
        inner_state = (model._inner.state_dict()
                       if hasattr(model, "_inner") else model.state_dict())
        torch.save({"model": inner_state, "epoch": epoch,
                    "cfg": cfg}, ckpt_dir / "last.pt")
        score = val_rmspe if val_count else val_loss
        if score < best_val:
            best_val = score
            stale = 0
            torch.save({"model": inner_state, "epoch": epoch,
                        "cfg": cfg, "val_rmspe_deg": val_rmspe},
                       ckpt_dir / "best.pt")
            tqdm.write(f"  ↳ new best (val_rmspe={val_rmspe:.3f}°); saved best.pt")
        else:
            stale += 1
            if stale >= patience:
                tqdm.write(f"Early stopping at epoch {epoch} (patience={patience})")
                break
    epoch_bar.close()

    with (ckpt_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)
    if writer is not None:
        writer.close()

    return {"best_val": best_val, "history_path": str(ckpt_dir / "history.json"),
            "best_ckpt": str(ckpt_dir / "best.pt")}


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reconunet-train",
        description="Train one of the three paper baselines from a YAML config.",
    )
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="Path to configs/train/{reconunet,subspacenet,subvit}.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the config's seed.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override train.epochs (useful for dry runs).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    cfg = _load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    # Project root = parent of the configs/ directory (two levels up from the
    # specific config file).  Robust to any cwd.
    project_root = args.config.resolve().parents[2]
    result = train_loop(cfg, project_root)
    LOG.info("Training finished.  Best-val=%.4f  ckpt=%s",
             result["best_val"], result["best_ckpt"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
