"""Adapter around the DOA-ViT model from Zhou et al.'s DOA_est_Master repo.

Upstream repo (pinned via git submodule):
    third_party/doa_est_master/ ↦ https://github.com/zzb-nice/DOA_est_Master.git

The upstream ViT lives at
``models/dl_model/vision_transformer/vit_model.py::VisionTransformer``.
It subclasses ``Grid_Based_network`` (at
``models/dl_model/grid_based_network.py``) which owns a grid-to-angle lookup
used in "sp_mode" (spatial-pseudo-spectrum mode).  We route through
``sp_mode=True`` so the forward pass emits class-scores over a uniform DoA
grid; the adapter then extracts the top-K grid cells as predicted angles.

The submodule doesn't install cleanly as a Python package, so we put the
submodule root on ``sys.path`` and import with its own relative paths.
The upstream uses an embedding layer at
``vision_transformer/embeding_layer.py::scm_embeding``.

Input/output contract
---------------------
* Input to the upstream ViT: the SCM embedding consumes
  :math:`K_x \\in \\mathbb{C}^{B \\times M \\times M}` presented as
  :math:`[B, 2, M, M]` (real/imag channels).  We build this from the shared
  snapshots inside :meth:`prepare_input`.
* Output (sp_mode=True, logits=False): spatial spectrum class scores over the
  ``out_dims``-sized grid.  We peak-pick top-K and map grid index → angle via
  the model's own ``sp_to_doa`` method.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ._base import BaselineAdapter, BaselineOutput


def _submodule_root() -> Path:
    """``<ReconUNet>/third_party/doa_est_master`` as an absolute path."""
    return Path(__file__).resolve().parents[4] / "third_party" / "doa_est_master"


def _ensure_on_path() -> None:
    root = _submodule_root()
    if not root.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"DOA_est_Master submodule not found at {root}. "
            "Run `git submodule update --init third_party/doa_est_master`."
        )
    path_str = str(root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    # The upstream uses an unqualified top-level ``models`` package, but
    # ``reconunet.models`` (a regular package) is also reachable as top-level
    # ``models`` once any classic module appends ``src/reconunet`` to sys.path
    # — and a regular package always wins over the upstream's namespace
    # package.  Pre-register a synthetic ``models`` module pointing at the
    # upstream directory so subsequent ``from models.x import …`` imports
    # resolve here.
    upstream_models = root / "models"
    existing = sys.modules.get("models")
    if existing is None or not any(
        Path(p).resolve() == upstream_models.resolve()
        for p in getattr(existing, "__path__", [])
    ):
        mod = types.ModuleType("models")
        mod.__path__ = [str(upstream_models)]
        sys.modules["models"] = mod


class SubViTAdapter(BaselineAdapter):
    """Canonical-interface wrapper around DOA-ViT (DOA_est_Master's
    :class:`VisionTransformer`)."""

    name = "subvit"

    def __init__(
        self,
        M: int = 8,
        K_max: int = 3,
        grid_size: int = 121,          # 1-deg resolution over (-60°, 60°)
        angle_range_deg: tuple = (-60.0, 60.0),
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
    ) -> None:
        self.M = M
        self.K_max = K_max
        self.grid_size = grid_size
        self.angle_range_deg = angle_range_deg
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

    # --- construction -------------------------------------------------------

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        _ensure_on_path()
        # Deferred imports: the submodule lays out ``models/dl_model/...``
        from models.dl_model.vision_transformer.vit_model import VisionTransformer  # type: ignore
        from models.dl_model.vision_transformer.embeding_layer import scm_embeding  # type: ignore

        M = int(cfg.get("M", self.M))
        embed_dim = int(cfg.get("embed_dim", self.embed_dim))
        depth = int(cfg.get("depth", self.depth))
        num_heads = int(cfg.get("num_heads", self.num_heads))
        grid_size = int(cfg.get("grid_size", self.grid_size))
        angle_range_deg = tuple(cfg.get("angle_range_deg", self.angle_range_deg))

        embed_layer = scm_embeding(M=M, ebedding_dim=embed_dim)

        # Grid_Based_network kwargs drive sp_mode=True:
        grid_kwargs = {
            "start_angle": float(angle_range_deg[0]),
            "end_angle": float(angle_range_deg[1]),
            "step": (float(angle_range_deg[1]) - float(angle_range_deg[0]))
                    / max(1, grid_size - 1),
        }

        model = VisionTransformer(
            embed_layer=embed_layer,
            out_dims=grid_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            sp_mode=True,
            **grid_kwargs,
        )
        # Remember grid for later peak-picking.
        model._grid_size = grid_size
        model._grid_start = float(angle_range_deg[0])
        model._grid_end = float(angle_range_deg[1])
        return model

    # --- data-plane ---------------------------------------------------------

    def prepare_input(self, snapshots: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """Convert ``snapshots`` of shape ``[B, M, T]`` into the ViT's expected
        real/imag stacked covariance ``[B, 2, M, M]``.

        The embedding layer (``scm_embeding``) transposes and flattens, so we
        give it exactly what it expects — a 4-D tensor whose channel axis
        holds ``(Re{K_x}, Im{K_x})``.
        """
        if snapshots.is_complex():
            B, M, T = snapshots.shape
            K = snapshots @ snapshots.conj().transpose(-1, -2) / T  # [B, M, M] complex
            out = torch.stack([K.real, K.imag], dim=1)              # [B, 2, M, M]
        else:
            # Some upstream datasets already provide [B, 2, M, M] directly.
            assert snapshots.dim() == 4 and snapshots.shape[1] == 2, (
                f"SubViTAdapter: expected complex [B, M, T] or [B, 2, M, M] real; "
                f"got {tuple(snapshots.shape)}"
            )
            out = snapshots
        return out.float()

    def forward(
        self,
        model: nn.Module,
        prepped: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BaselineOutput:
        logits = model(prepped, logits=False)   # [B, grid_size] spatial spectrum
        # Peak-pick the TRUE per-sample source count when the trainer threads it
        # through ``meta['n_sources']`` (variable-K, like SubspaceNetAdapter);
        # otherwise fall back to the configured K_max.  Picking a fixed K_max on
        # a variable-K corpus injects spurious peaks on low-K samples and badly
        # inflates RMSPE (3 phantom angles on every K=1 scene).
        K_max = int(self.K_max)
        n_sources = None if meta is None else meta.get("n_sources")
        angles_pred = self._peak_pick(logits, model, K=K_max, n_sources=n_sources)

        loss = None
        if targets is not None and self.training_loss_fn is not None:
            loss = self.training_loss_fn(logits, targets)
        return BaselineOutput(
            angles_pred=angles_pred,
            loss=loss,
            extras={"spatial_spectrum": logits},
        )

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _peak_pick(spectrum: torch.Tensor, model: nn.Module, K: int,
                   n_sources: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return grid-peak DoAs in radians, shape ``[B, K]`` (NaN-padded).

        ``torch.topk`` returns peaks in descending spectrum-value order, so the
        first columns are the strongest peaks.  When ``n_sources`` is supplied
        (per-sample true K), peaks beyond each sample's K are set to NaN so a
        K<K_max scene contributes exactly its true sources to the downstream
        NaN-aware RMSPE — no phantom angles.  Output is sorted ascending by
        angle (NaN sorts to the end), matching the eval-harness convention.

        For tightly spaced sources a local-max pass would be more principled;
        we leave that to the evaluation harness (post-processing).
        """
        _, idx = torch.topk(spectrum, k=K, dim=-1)            # [B, K] desc by value
        grid_size = int(getattr(model, "_grid_size", spectrum.shape[-1]))
        g0 = float(getattr(model, "_grid_start", -60.0))
        g1 = float(getattr(model, "_grid_end", 60.0))
        grid_deg = torch.linspace(g0, g1, grid_size, device=spectrum.device)
        angles_deg = grid_deg[idx]                             # [B, K]
        if n_sources is not None:
            ks = torch.as_tensor(n_sources, device=spectrum.device).long().clamp(1, K)
            col = torch.arange(K, device=spectrum.device).unsqueeze(0)    # [1, K]
            keep = col < ks.unsqueeze(1)                       # [B, K]; first k_true (strongest)
            angles_deg = torch.where(keep, angles_deg,
                                     torch.full_like(angles_deg, float("nan")))
        # Sort per-row so predictions come out in ascending angle order (NaN to
        # the end), matching the eval harness's RMSPE / Hungarian-matched loss.
        angles_deg, _ = torch.sort(angles_deg, dim=-1)
        return torch.deg2rad(angles_deg)

    training_loss_fn = None
