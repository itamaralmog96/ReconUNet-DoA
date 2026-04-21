"""Deterministic parameters → complex snapshot matrices.

``SceneRenderer`` is the *only* component that knows how to turn a
:class:`~reconunet.data.scene_manifest.Scene` into observable array data.
By funneling every baseline model through the same renderer, we guarantee
all three estimators in the paper (ReconUNet / SubspaceNet / SubViT) train
and evaluate on byte-identical observations — the whole point of the reorg.

Forward model
-------------
For a uniform linear array (ULA) of ``M`` isotropic elements at half-wavelength
spacing, ``K`` narrowband plane-wave sources at angles
``θ = (θ₁, …, θ_K)`` with unit-power complex Gaussian symbols ``s(t)`` and
additive white Gaussian noise ``n(t)`` at SNR ``γ`` (dB), the snapshot matrix
is

    X[m, t] = Σ_k a_m(θ_k) · s_k(t)  +  n[m, t]

with steering vector

    a_m(θ) = exp(-j · 2π · (m · d / λ) · sin(θ)) · g_m · exp(j · ψ_m)

where ``g_m``, ``ψ_m`` are per-element gain / phase errors, the element
position is jittered by ``δx`` (fraction of ``d``), and a nearest-neighbour
mutual-coupling coefficient ``c`` may be applied as a Toeplitz mixer.

All randomness is seeded from ``Scene.seed`` so ``render(scene) ↦ X`` is a
pure function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from .scene_manifest import (
    ArrayType,
    ManifestMeta,
    ModulationType,
    Scene,
)

if TYPE_CHECKING:  # pragma: no cover — for static type-checkers only
    import torch

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _steering_vector(
    angles_rad: np.ndarray,
    M: int,
    element_spacing_lambda: float,
    gain_err: np.ndarray,
    phase_err: np.ndarray,
    position_err: np.ndarray,
) -> np.ndarray:
    """Return the ``[M, K]`` complex steering matrix ``A(θ)``.

    All error vectors are shape ``[M]`` and applied element-wise.
    """
    m = np.arange(M, dtype=np.float64)[:, None]          # [M, 1]
    theta = angles_rad[None, :]                           # [1, K]
    d_over_lambda = element_spacing_lambda * (1.0 + position_err[:, None])
    phase = -2.0 * np.pi * m * d_over_lambda * np.sin(theta)
    A = np.exp(1j * phase)                                # [M, K]
    A = A * gain_err[:, None] * np.exp(1j * phase_err[:, None])
    return A


def _mutual_coupling_matrix(M: int, coupling: float) -> np.ndarray:
    """Tri-diagonal nearest-neighbour coupling Toeplitz matrix."""
    if coupling == 0.0:
        return np.eye(M, dtype=np.complex128)
    col = np.zeros(M, dtype=np.complex128)
    col[0] = 1.0
    col[1] = coupling
    # symmetric Toeplitz
    from scipy.linalg import toeplitz  # local import — scipy is a required dep
    return toeplitz(col, col)


@dataclass
class RenderResult:
    """Everything the renderer produces for one scene."""

    snapshots: np.ndarray     # [M, T] complex64
    covariance: np.ndarray    # [M, M] complex64 (sample covariance)
    steering: np.ndarray      # [M, K] complex64
    source_signals: np.ndarray  # [K, T] complex64
    noise: np.ndarray         # [M, T] complex64
    angles_rad: np.ndarray    # [K] float64


class SceneRenderer:
    """Deterministic scene → observations.

    One renderer instance binds to a single :class:`ManifestMeta` and is
    safe to share across DataLoader workers — all state is read-only once
    constructed.

    Examples
    --------
    >>> meta = ManifestMeta(M=8, T=512, tau=8)
    >>> rend = SceneRenderer(meta)
    >>> result = rend.render(scene)
    >>> result.snapshots.shape
    (8, 512)
    """

    def __init__(self, meta: ManifestMeta):
        self.meta = meta

    # --- per-scene rendering ------------------------------------------------

    def render(self, scene: Scene) -> RenderResult:
        rng = np.random.default_rng(int(scene.seed))
        M, T = self.meta.M, self.meta.T

        # --- array calibration draws (per-scene so seed reproduces) --------
        gain_err = 1.0 + (scene.gain_err_dB / 8.686) * rng.standard_normal(M)
        phase_err = np.deg2rad(scene.phase_err_deg) * rng.standard_normal(M)
        position_err = (scene.position_err_pct / 100.0) * rng.standard_normal(M)

        # --- steering ------------------------------------------------------
        K = int(scene.n_sources)
        angles_rad = np.deg2rad(scene.angles_deg[:K].astype(np.float64))
        A = _steering_vector(
            angles_rad,
            M=M,
            element_spacing_lambda=self.meta.element_spacing_lambda,
            gain_err=gain_err,
            phase_err=phase_err,
            position_err=position_err,
        )
        if scene.mutual_coupling != 0.0:
            C = _mutual_coupling_matrix(M, float(scene.mutual_coupling))
            A = C @ A

        # --- source signals ------------------------------------------------
        sources = self._draw_sources(rng, K=K, T=T, modulation=scene.modulation)

        # --- noise (set SNR against unit-variance sources) -----------------
        snr_linear = 10.0 ** (scene.snr_db / 10.0)
        noise_var = 1.0 / snr_linear
        noise = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) \
            * np.sqrt(noise_var / 2.0)

        X = A @ sources + noise                     # [M, T] complex128
        Rxx = (X @ X.conj().T) / float(T)           # [M, M]

        return RenderResult(
            snapshots=X.astype(np.complex64),
            covariance=Rxx.astype(np.complex64),
            steering=A.astype(np.complex64),
            source_signals=sources.astype(np.complex64),
            noise=noise.astype(np.complex64),
            angles_rad=angles_rad,
        )

    # --- source modulation --------------------------------------------------

    @staticmethod
    def _draw_sources(
        rng: np.random.Generator,
        K: int,
        T: int,
        modulation: ModulationType,
    ) -> np.ndarray:
        """Unit-variance complex source signals ``[K, T]``."""
        if modulation in (ModulationType.NARROWBAND, ModulationType.WIDEBAND):
            re = rng.standard_normal((K, T))
            im = rng.standard_normal((K, T))
            return (re + 1j * im) / np.sqrt(2.0)
        if modulation == ModulationType.BPSK:
            return rng.choice(np.array([-1.0, 1.0], dtype=np.complex128), size=(K, T))
        if modulation == ModulationType.QPSK:
            symbols = rng.choice(
                np.array(
                    [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128
                )
                / np.sqrt(2.0),
                size=(K, T),
            )
            return symbols
        raise ValueError(f"Unsupported modulation: {modulation!r}")


# ---------------------------------------------------------------------------
# Lag-stack autocorrelation (used by SubspaceNet + ReconUNet)
# ---------------------------------------------------------------------------


def lag_stack(snapshots: "torch.Tensor", tau: int) -> "torch.Tensor":
    """Build the short-range autocorrelation tensor :math:`R^{\\tau}_x`.

    Parameters
    ----------
    snapshots
        Complex tensor of shape ``[B, M, T]`` (batched) or ``[M, T]`` (one
        sample).  Both are accepted; the function inserts a leading batch
        dimension as needed.
    tau
        Number of lags.  For the paper we used ``τ = 8``.

    Returns
    -------
    Tensor of shape ``[B, τ, 2M, M]`` where the channel axis stacks real over
    imaginary so downstream conv-nets can operate on real-valued tensors.

    Notes
    -----
    The lag-``ℓ`` sample autocorrelation is

        R_ℓ[m, m'] = (1 / (T-ℓ)) · Σ_t  X[m, t+ℓ] · conj(X[m', t])

    so the stack provides a running view of short-range temporal structure
    that both SubspaceNet and ReconUNet ingest.
    """
    import torch  # deferred — lets the rest of the renderer run in numpy-only envs

    squeezed = snapshots.dim() == 2
    if squeezed:
        snapshots = snapshots.unsqueeze(0)
    if not snapshots.is_complex():
        raise TypeError(f"snapshots must be complex; got {snapshots.dtype}")

    B, M, T = snapshots.shape
    if tau >= T:
        raise ValueError(f"tau={tau} must be strictly smaller than T={T}")

    out = torch.zeros((B, tau, 2 * M, M), dtype=torch.float32, device=snapshots.device)
    for lag in range(tau):
        past = snapshots[..., : T - lag - 1]               # [B, M, T-lag-1]
        future = snapshots[..., lag + 1:]                 # [B, M, T-lag-1]
        R = (future @ past.conj().transpose(-1, -2)) / float(T - lag - 1)  # [B, M, M]
        out[:, lag, :M, :] = R.real
        out[:, lag, M:, :] = R.imag

    if squeezed:
        out = out.squeeze(0)
    return out


__all__ = [
    "SceneRenderer",
    "RenderResult",
    "lag_stack",
]
