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
        sys.path.append(path_str)


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
            "grid_size": grid_size,
            "grid_start": float(angle_range_deg[0]),
            "grid_end": float(angle_range_deg[1]),
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
        K_max = int((meta or {}).get("K_max", self.K_max))
        angles_pred = self._peak_pick(logits, model, K=K_max)

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
    def _peak_pick(spectrum: torch.Tensor, model: nn.Module, K: int) -> torch.Tensor:
        """Return top-K grid cells converted to radians, shape ``[B, K]``.

        Simple ``topk`` peak-picking — good enough for well-separated sources.
        For tightly spaced sources a local-max pass would be more principled;
        we leave that to the evaluation harness (post-processing).
        """
        B = spectrum.shape[0]
        _, idx = torch.topk(spectrum, k=K, dim=-1)            # [B, K]
        grid_size = int(getattr(model, "_grid_size", spectrum.shape[-1]))
        g0 = float(getattr(model, "_grid_start", -60.0))
        g1 = float(getattr(model, "_grid_end", 60.0))
        grid_deg = torch.linspace(g0, g1, grid_size, device=spectrum.device)
        angles_deg = grid_deg[idx]                             # [B, K]
        # Sort per-row so predictions come out in ascending angle order (matches
        # the evaluation harness's RMSPE / Hungarian-matched loss convention).
        angles_deg, _ = torch.sort(angles_deg, dim=-1)
        return torch.deg2rad(angles_deg)

    training_loss_fn = None
