"""PyTorch ``Dataset`` + model-specific collate functions over a scene manifest.

The design separates concerns:

1. :class:`SceneDataset` owns the manifest + renderer and always yields the
   *same canonical observation* (complex snapshot matrix + angle labels).
2. :class:`SceneCollate` instances are cheap post-processors that turn the
   canonical observation into whatever shape the wrapped model consumes.
   Three are provided out of the box — one per paper baseline — and all are
   configured from a single :class:`~reconunet.data.scene_manifest.ManifestMeta`
   so they see identical scenes.

This split is what lets ``train.py --model {reconunet,subspacenet,subvit}``
reuse one manifest on disk and one DataLoader pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .scene_manifest import K_MAX, ManifestMeta, Scene, SceneManifest
from .scene_renderer import RenderResult, SceneRenderer, lag_stack


# ---------------------------------------------------------------------------
# Canonical dataset
# ---------------------------------------------------------------------------


@dataclass
class CanonicalSample:
    """What ``SceneDataset.__getitem__`` emits, before any collate step.

    Carrying the whole :class:`RenderResult` is deliberate: it is ~48 KB for
    M=8/T=512/K=3 scenes but lets the collate functions recycle work (e.g.
    the SubViT path can reuse the covariance that ReconUNet's collate also
    needs) without re-rendering.
    """

    snapshots: torch.Tensor          # complex [M, T]
    covariance: torch.Tensor         # complex [M, M]
    angles_rad: torch.Tensor         # float  [K_MAX]  (NaN padded)
    n_sources: int
    snr_db: float
    scene_id: int


class SceneDataset(Dataset):
    """Deterministic, memory-mappable dataset over a :class:`SceneManifest`.

    Every ``__getitem__`` call redraws noise/sources from the scene's seed, so
    enabling PyTorch's standard ``DataLoader(num_workers > 0)`` is free and
    safe — each worker sees identical output bytes for the same index.
    """

    def __init__(
        self,
        manifest: SceneManifest,
        renderer: Optional[SceneRenderer] = None,
    ):
        self.manifest = manifest
        self.renderer = renderer or SceneRenderer(manifest.meta)

    @property
    def meta(self) -> ManifestMeta:
        return self.manifest.meta

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> CanonicalSample:
        scene = self.manifest[index]
        result = self.renderer.render(scene)

        # Pad NaN angles to K_MAX for ragged-batch support.
        angles_padded = np.full((K_MAX,), np.nan, dtype=np.float32)
        angles_padded[: scene.n_sources] = result.angles_rad.astype(np.float32)

        return CanonicalSample(
            snapshots=torch.from_numpy(result.snapshots),
            covariance=torch.from_numpy(result.covariance),
            angles_rad=torch.from_numpy(angles_padded),
            n_sources=scene.n_sources,
            snr_db=scene.snr_db,
            scene_id=scene.scene_id,
        )


# ---------------------------------------------------------------------------
# Collate functions — one per model family
# ---------------------------------------------------------------------------


class _BaseCollate:
    """Common work for every collate: stack labels + metadata."""

    def __init__(self, meta: ManifestMeta):
        self.meta = meta

    def _stack_common(
        self, batch: Sequence[CanonicalSample]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        angles = torch.stack([b.angles_rad for b in batch], dim=0)
        k = torch.tensor([b.n_sources for b in batch], dtype=torch.int64)
        snr = torch.tensor([b.snr_db for b in batch], dtype=torch.float32)
        ids = [b.scene_id for b in batch]
        return angles, k, snr, ids


class ReconUNetCollate(_BaseCollate):
    """Build the lag-stack tensor ReconUNet (EVDUNet) expects: ``[B, τ, 2M, M]``."""

    def __call__(self, batch: Sequence[CanonicalSample]) -> Dict[str, torch.Tensor]:
        snaps = torch.stack([b.snapshots for b in batch], dim=0)   # [B, M, T] complex
        stack = lag_stack(snaps, tau=self.meta.tau)                # [B, τ, 2M, M] float
        cov = torch.stack([b.covariance for b in batch], dim=0)    # [B, M, M] complex
        angles, k, snr, ids = self._stack_common(batch)
        return {
            "input":       stack,
            "covariance":  cov,
            "angles_rad":  angles,
            "n_sources":   k,
            "snr_db":      snr,
            "scene_ids":   ids,
        }


class SubspaceNetCollate(_BaseCollate):
    """Same lag-stack input as ReconUNet — SubspaceNet consumes ``[B, τ, 2M, M]``.

    Kept as a distinct class so future divergence (different τ, normalization,
    etc.) is a one-class change, not a ``match``-style switch.
    """

    def __call__(self, batch: Sequence[CanonicalSample]) -> Dict[str, torch.Tensor]:
        snaps = torch.stack([b.snapshots for b in batch], dim=0)
        stack = lag_stack(snaps, tau=self.meta.tau)
        angles, k, snr, ids = self._stack_common(batch)
        return {
            "input":       stack,
            "angles_rad":  angles,
            "n_sources":   k,
            "snr_db":      snr,
            "scene_ids":   ids,
        }


class SubViTCollate(_BaseCollate):
    """Produce the real/imag covariance image ``[B, 2, M, M]`` the ViT expects."""

    def __call__(self, batch: Sequence[CanonicalSample]) -> Dict[str, torch.Tensor]:
        K_batched = torch.stack([b.covariance for b in batch], dim=0)  # [B, M, M] complex
        img = torch.stack([K_batched.real, K_batched.imag], dim=1).float()  # [B, 2, M, M]
        angles, k, snr, ids = self._stack_common(batch)
        return {
            "input":       img,
            "angles_rad":  angles,
            "n_sources":   k,
            "snr_db":      snr,
            "scene_ids":   ids,
        }


# ---------------------------------------------------------------------------
# Resolver (config-driven lookup)
# ---------------------------------------------------------------------------


_COLLATE_REGISTRY: Dict[str, Callable[[ManifestMeta], _BaseCollate]] = {
    "reconunet":   ReconUNetCollate,
    "subspacenet": SubspaceNetCollate,
    "subvit":      SubViTCollate,
}


def get_collate(name: str, meta: ManifestMeta) -> _BaseCollate:
    """Factory consumed by training configs: ``get_collate("reconunet", meta)``."""
    key = name.strip().lower()
    if key not in _COLLATE_REGISTRY:
        raise KeyError(
            f"Unknown collate {name!r}; valid options are: {sorted(_COLLATE_REGISTRY)}"
        )
    return _COLLATE_REGISTRY[key](meta)


__all__ = [
    "CanonicalSample",
    "ReconUNetCollate",
    "SceneDataset",
    "SubViTCollate",
    "SubspaceNetCollate",
    "get_collate",
]
