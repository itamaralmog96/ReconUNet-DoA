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


# ---------------------------------------------------------------------------
# Multipath synthesis — legacy-faithful port of
# reconunet.signalgen.signal_generator.SignalGenerator.add_multipath
# ---------------------------------------------------------------------------

# Legacy default: bandwidth = fs/2 * 0.1 = fs * 0.05  (SignalConfig.__post_init__)
_LEGACY_BW_FRACTION_OF_FS = 0.05


def _sample_multipath_params(
    rng: np.random.Generator,
    num_paths: int,
    max_delay_seconds: float,
    distribution: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw (τ_seconds, dB_loss, φ_rad) for each multipath path — legacy formulas.

    Mirrors ``signal_generator.add_multipath`` lines 308–321:
      * uniform:     τ ~ U(0, τ_max);   dB_loss = 10·(τ/τ_max) + U(1, 2)
      * exponential: τ ~ Exp(τ_max/2);  dB_loss = 10·(1 − e^(-τ/τ_max)) + Exp(2.5)
      * φ ~ N(π, π/4) in both cases.
    """
    if distribution == "uniform":
        tau = rng.uniform(0.0, max_delay_seconds, size=num_paths)
        base_db = 10.0 * (tau / max_delay_seconds)
        extra_db = rng.uniform(1.0, 2.0, size=num_paths)
    elif distribution == "exponential":
        tau = rng.exponential(max_delay_seconds / 2.0, size=num_paths)
        base_db = 10.0 * (1.0 - np.exp(-tau / max_delay_seconds))
        extra_db = rng.exponential(2.5, size=num_paths)
    else:
        raise ValueError(f"Unknown multipath distribution: {distribution!r}")
    db_loss = base_db + extra_db
    phi = rng.normal(np.pi, np.pi / 4.0, size=num_paths)
    return tau, db_loss, phi


def _legacy_bandlimited_sources(
    rng: np.random.Generator, K: int, signal_length: int,
) -> np.ndarray:
    """Legacy ``generate_source_signals`` in full-bandwidth mode, ``[K, N]``.

    Reproduces the iFFT-of-rectangular-filtered-white-noise recipe at
    ``signal_generator.generate_source_signals`` lines 153–179 with
    ``use_full_bandwidth=True`` (the cleanest bandlimited default).  Power is
    normalised to unit variance per source so the direct-path SNR bookkeeping
    in :meth:`SceneRenderer.render` is unchanged.
    """
    out = np.empty((K, signal_length), dtype=np.complex128)
    for k in range(K):
        freq_noise = (rng.standard_normal(signal_length)
                      + 1j * rng.standard_normal(signal_length))
        # use_full_bandwidth=True branch: filter_response = ones, so the iFFT
        # just returns the white complex noise transformed back.  We keep the
        # exact legacy call order (ifft of the noise) to stay byte-close to
        # the reference implementation.
        time_signal = np.fft.ifft(freq_noise, n=signal_length)
        power = np.mean(np.abs(time_signal) ** 2)
        out[k] = time_signal / np.sqrt(power) if power > 0 else time_signal
    return out


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

        # --- direct-path steering -----------------------------------------
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

        # --- multipath (legacy-faithful — see _sample_multipath_params) ----
        # When scene.has_multipath is False the whole block is skipped, so
        # non-multipath manifests render byte-identically to before v1.1.
        has_mp = bool(scene.has_multipath) and int(scene.num_multipath) > 0
        if has_mp:
            mp_result = self._render_with_multipath(
                rng=rng, scene=scene, M=M, T=T, K=K,
                A_direct=A, angles_rad_direct=angles_rad,
                gain_err=gain_err, phase_err=phase_err, position_err=position_err,
            )
            A_full, sources_full, angles_rad_full = mp_result
        else:
            sources_full = self._draw_sources(rng, K=K, T=T, modulation=scene.modulation)
            A_full, angles_rad_full = A, angles_rad

        # --- mutual coupling applies to the full steering matrix -----------
        if scene.mutual_coupling != 0.0:
            C = _mutual_coupling_matrix(M, float(scene.mutual_coupling))
            A_full = C @ A_full

        # --- noise (set SNR against the *direct-path* unit-variance source) -
        # Rationale: we want --snr-db in the manifest to have the same meaning
        # whether or not multipath is on, so the noise variance is fixed to
        # match direct-path K unit-power sources exactly as in the legacy
        # add_noise (signal_power computed on direct s's, not on reflections).
        snr_linear = 10.0 ** (scene.snr_db / 10.0)
        noise_var = 1.0 / snr_linear
        noise = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) \
            * np.sqrt(noise_var / 2.0)

        X = A_full @ sources_full + noise           # [M, T] complex128
        Rxx = (X @ X.conj().T) / float(T)           # [M, M]

        return RenderResult(
            snapshots=X.astype(np.complex64),
            covariance=Rxx.astype(np.complex64),
            steering=A_full.astype(np.complex64),
            source_signals=sources_full.astype(np.complex64),
            noise=noise.astype(np.complex64),
            angles_rad=angles_rad_full,
        )

    # --- legacy-faithful multipath branch ----------------------------------

    def _render_with_multipath(
        self,
        *,
        rng: np.random.Generator,
        scene: Scene,
        M: int,
        T: int,
        K: int,
        A_direct: np.ndarray,
        angles_rad_direct: np.ndarray,
        gain_err: np.ndarray,
        phase_err: np.ndarray,
        position_err: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Produce ``(A_full, sources_full, angles_rad_full)`` with multipath.

        Mirrors ``signal_generator.add_multipath`` (legacy):
          * draws N_mp additional paths, each with a random AoA and a complex
            gain ``10^(-dB/20) · √P · exp(j(φ − 2π f_c τ))`` (line 332);
          * the multipath signal is a ``N_k``-sample-delayed copy of source 0's
            bandlimited-Gaussian realisation (legacy: resample_poly + slice;
            here: FFT-bandlimited draw of length ``T + max_delay_samples``,
            integer-sample slicing — statistically equivalent for the
            iid-Gaussian-at-Nyquist case used throughout the new pipeline).
        """
        num_mp = int(scene.num_multipath)
        fs = float(self.meta.fs_Hz)
        bw = fs * _LEGACY_BW_FRACTION_OF_FS                        # legacy default
        max_delay_seconds = 1.0 / bw                               # legacy line 249
        max_delay_samples = int(max_delay_seconds * fs)
        if max_delay_samples >= T:
            # legacy raises here; we clamp to keep the pipeline robust.
            max_delay_samples = max(1, T // 2)
            max_delay_seconds = max_delay_samples / fs

        # 1) Bandlimited (full-band) source signals of length T+max_delay_samples
        signal_len = T + max_delay_samples
        sources_long = _legacy_bandlimited_sources(rng, K=K, signal_length=signal_len)
        direct = sources_long[:, :T]                               # [K, T]

        # 2) Sample multipath parameters (legacy formulas)
        distribution = "uniform" if int(scene.mp_distribution) == 0 else "exponential"
        tau, db_loss, phi = _sample_multipath_params(
            rng, num_paths=num_mp,
            max_delay_seconds=max_delay_seconds,
            distribution=distribution,
        )
        # Multipath AoAs — legacy draws U[0, 2π); we use the value directly
        # with the new renderer's sin(θ) convention (see module docstring).
        aoa_rad = rng.uniform(0.0, 2.0 * np.pi, size=num_mp)

        # 3) Delay the source[0] signal by N_k samples (legacy line 344).  For
        # the iid-Gaussian regime, N_k ≥ 1 gives an uncorrelated copy of the
        # source — i.e. incoherent multipath, which is exactly what the legacy
        # produces on iid sources.  Coherent behaviour would require
        # genuinely bandlimited sources; the FFT draw above is the closest
        # the new pipeline gets without changing the direct-source model.
        source_power = float(np.mean(np.abs(direct[0]) ** 2))       # should be ≈ 1
        # Legacy uses factor/2 as the sample-shift normaliser.
        factor = float(scene.mp_max_delay_factor) or 10.0
        shift_samples = np.clip(
            (tau * fs * factor / 2.0).astype(np.int64), 1, max_delay_samples
        )
        amp = 10.0 ** (-db_loss / 20.0) * np.sqrt(source_power)
        carrier_phase = -2.0 * np.pi * fs * tau                     # fc ≈ fs here
        complex_gain = amp * np.exp(1j * (phi + carrier_phase))

        mp_signals = np.empty((num_mp, T), dtype=np.complex128)
        for k in range(num_mp):
            N_k = int(shift_samples[k])
            mp_signals[k] = complex_gain[k] * sources_long[0, N_k:N_k + T]

        # 4) Multipath steering columns — same imperfections as direct paths.
        A_mp = _steering_vector(
            aoa_rad, M=M,
            element_spacing_lambda=self.meta.element_spacing_lambda,
            gain_err=gain_err, phase_err=phase_err, position_err=position_err,
        )

        A_full = np.concatenate([A_direct, A_mp], axis=1)           # [M, K+N]
        sources_full = np.concatenate([direct, mp_signals], axis=0) # [K+N, T]
        angles_rad_full = np.concatenate([angles_rad_direct, aoa_rad])
        return A_full, sources_full, angles_rad_full

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
