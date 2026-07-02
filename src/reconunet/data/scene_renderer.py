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


# Default coupling phase and per-diagonal jitter follow the legacy
# ``signalgen.array_processing.ArrayConfig`` settings (v0 thesis defaults), which
# in turn track Almog & Weiss 2026 §II-C eqs. (24)-(25).  ρ (the magnitude of γ)
# is provided per scene via ``Scene.mutual_coupling``.
_DEFAULT_COUPLING_PHASE_DEG: float = -100.0     # legacy ArrayConfig.coupling_phase_deg
_DEFAULT_COUPLING_VARIATION: float = 0.9        # legacy ArrayConfig.coupling_variation


def _steering_vector(
    angles_rad: np.ndarray,
    M: int,
    element_spacing_lambda: float,
    gain_err: np.ndarray,
    phase_err: np.ndarray,
    position_err: np.ndarray,
) -> np.ndarray:
    """Return the ``[M, K]`` complex steering matrix ``A(θ)``.

    Parameters
    ----------
    angles_rad
        Source angles in radians, shape ``[K]``.  ``θ=0`` is the array
        broadside, ``θ`` measured around it (sine convention).
    M
        Number of array elements.  The ULA is assumed along the x-axis, with
        the n-th element nominally at ``(n · d, 0)`` where ``d = element_spacing_lambda · λ``.
    element_spacing_lambda
        Inter-element spacing in wavelengths (``d/λ``).
    gain_err, phase_err
        Real arrays of shape ``[M]`` — multiplicative gain (linear) and additive
        phase (radians) applied per element.
    position_err
        Per-element position perturbation.  Two shapes are accepted:

        * ``[M, 2]``  — paper-faithful: rows are ``(εx_n, εy_n)`` in **wavelengths**,
          so ``ε_n ∼ N(0, σ_pos² · I_2)`` matches Almog & Weiss 2026 eq. (21).
          The phase contribution is
          ``-2π · ((n·d/λ + εx_n) · sin θ + εy_n · cos θ)``,
          covering both axial and transverse jitter.
        * ``[M]``    — legacy / notebook compatibility: interpreted as a
          *fractional* perturbation of ``d/λ`` (axial only), giving
          ``-2π · n · d/λ · (1 + position_err[n]) · sin θ``.  Used by the
          older `_build_verify_array.py` notebook script and a handful of
          ad-hoc callers.

    Notes
    -----
    Composite operator order follows paper eq. (26) ``H = M·G``: gain/phase
    are applied here as ``G = diag(gain_err · exp(j·phase_err))`` (so the
    output of this function is ``G · a_pos``), and the caller multiplies by
    the mutual-coupling matrix ``M`` afterwards (``A_full = C @ A_full`` in
    :meth:`SceneRenderer.render`).
    """
    n = np.arange(M, dtype=np.float64)[:, None]          # [M, 1] — element index
    theta = angles_rad[None, :]                           # [1, K]

    if position_err.ndim == 2:
        # Paper-faithful: 2-D Gaussian (εx, εy) in λ units.
        if position_err.shape != (M, 2):
            raise ValueError(
                f"2-D position_err must be [M, 2]; got {position_err.shape}"
            )
        x_lambda = n * element_spacing_lambda + position_err[:, 0:1]   # [M, 1]
        y_lambda = position_err[:, 1:2]                                # [M, 1]
        phase = -2.0 * np.pi * (
            x_lambda * np.sin(theta) + y_lambda * np.cos(theta)
        )
    elif position_err.ndim == 1:
        # Legacy axial-fractional convention (kept for back-compat).
        if position_err.shape != (M,):
            raise ValueError(
                f"1-D position_err must be [M]; got {position_err.shape}"
            )
        d_over_lambda = element_spacing_lambda * (1.0 + position_err[:, None])
        phase = -2.0 * np.pi * n * d_over_lambda * np.sin(theta)
    else:
        raise ValueError(
            f"position_err must have 1 or 2 dims; got ndim={position_err.ndim}"
        )

    A = np.exp(1j * phase)                                # [M, K]
    A = A * gain_err[:, None] * np.exp(1j * phase_err[:, None])
    return A


def _mutual_coupling_matrix(
    M: int,
    rho: float,
    *,
    rng: Optional[np.random.Generator] = None,
    phase_deg: float = _DEFAULT_COUPLING_PHASE_DEG,
    variation: float = _DEFAULT_COUPLING_VARIATION,
) -> np.ndarray:
    """Reciprocal Toeplitz mutual-coupling matrix ``M = I + E`` (paper eq. 24-25).

    Off-diagonal entries follow a geometric decay with per-diagonal uniform
    jitter,

        ``c_k = γ^k · v_k,        k = 1, …, N-1``
        ``γ   = ρ · exp(j·φ_c)``                          (paper eq. 25)
        ``v_k ∼ U(1 − δ/2, 1 + δ/2)``

    and the matrix is built as ``E[i, j] = c_{|i-j|}`` for ``i ≠ j``, zero on
    the diagonal — symmetric Toeplitz, "reciprocal" in the paper's sense
    (passive sensors couple identically in both directions).

    Parameters
    ----------
    M
        Array size.
    rho
        ``|γ|`` — magnitude of the unit-step coupling coefficient (the
        ``mutual_coupling`` field of :class:`Scene`).  ``ρ = 0`` short-circuits
        to the identity.
    rng
        Optional generator for the per-diagonal jitter ``v_k``.  Threading
        ``Scene.seed`` through here keeps the coupling matrix reproducible
        across re-runs.
    phase_deg, variation
        ``φ_c`` and ``δ`` from eq. (25).  Default to the legacy
        ``ArrayConfig`` settings (φ_c=-100°, δ=0.9) so ``mild`` and ``harsh``
        keep their established meaning while now exercising every off-diagonal.

    Notes
    -----
    The previous nearest-neighbour-only Toeplitz ``[1, ρ, 0, …, 0]`` was a
    simplification; this implementation matches the paper and the legacy
    ``signalgen/array_processing.py::_generate_mutual_coupling_matrix``.
    """
    if rho == 0.0 or M <= 1:
        return np.eye(M, dtype=np.complex128)

    rng = rng if rng is not None else np.random.default_rng()
    gamma = float(rho) * np.exp(1j * np.deg2rad(float(phase_deg)))
    v = rng.uniform(1.0 - variation / 2.0, 1.0 + variation / 2.0, size=M - 1)
    # c_k = γ^k · v_k for k = 1..M-1   (paper eq. 25)
    powers = gamma ** np.arange(1, M)                              # [M-1]
    c = powers * v                                                 # [M-1] complex

    # Symmetric Toeplitz: E[i, j] = c_{|i-j|}, E[i, i] = 0.
    # Build via index-difference broadcast — O(M²) but M ≤ 8 in practice.
    idx = np.arange(M)
    diff = np.abs(idx[:, None] - idx[None, :])                     # [M, M] int
    E = np.zeros((M, M), dtype=np.complex128)
    nonzero = diff > 0
    E[nonzero] = c[diff[nonzero] - 1]                              # c is 1-indexed → diff-1
    return np.eye(M, dtype=np.complex128) + E


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
    """Everything the renderer produces for one scene.

    ``covariance`` is the *corrupted* sample covariance (carries noise,
    array errors, mutual coupling, multipath) — that's the deep network's
    input.  ``covariance_clean`` is the *ideal* covariance computed
    analytically from only the K direct-path angles with a perfect array
    and unit-power sources: ``R_clean = A_ideal @ A_ideal^H``.  It is the
    paper's supervision target for L_eig / L_proj / L_dom / L_rec
    (Almog & Weiss 2026 §IV "Composite loss").

    Shape contract (multipath-aware)
    --------------------------------
    Let ``K`` = ``scene.n_sources`` and ``N`` = ``scene.num_multipath`` (0
    if multipath is disabled).  Then:

    * ``angles_rad`` is **always** ``[K]`` — the *direct-path* angles only.
      This is the labelled ground truth for DoA estimation; multipath
      replicas are not predicted by the model and must not appear here.
    * ``angles_rad_multipath`` is ``[N]`` — the multipath replica AoAs,
      drawn uniformly in ``[0, 2π)``.  Empty array if ``N == 0``.
    * ``steering`` is ``[M, K+N]`` — the augmented steering matrix that
      generated the corrupted snapshots (direct + multipath columns).
    * ``source_signals`` is ``[K+N, T]`` — same row order as ``steering``.

    Older callers that read ``result.angles_rad`` and assumed it had length
    K+N when multipath was on were silently incorrect (it caused a
    broadcasting crash in :class:`SceneDataset`).
    """

    snapshots: np.ndarray            # [M, T] complex64  (corrupted)
    covariance: np.ndarray           # [M, M] complex64  (corrupted sample covariance)
    covariance_clean: np.ndarray     # [M, M] complex64  (ideal A_ideal @ A_ideal^H)
    steering: np.ndarray             # [M, K+N] complex64  (direct + multipath)
    source_signals: np.ndarray       # [K+N, T] complex64  (direct + multipath)
    noise: np.ndarray                # [M, T] complex64
    angles_rad: np.ndarray           # [K] float64        (direct-path angles only)
    angles_rad_multipath: np.ndarray  # [N] float64       (multipath AoAs; [] if N=0)


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
        # Distributions follow Almog & Weiss 2026 §II-C eqs. (21)-(23) and
        # the legacy ``signalgen.array_processing`` implementation:
        #   - g_n ∼ U[10^(-Δ/20), 10^(+Δ/20)]   (eq. 23, Δ = scene.gain_err_dB)
        #   - φ_n ∼ U[-Φ, +Φ] degrees             (eq. 23, Φ = scene.phase_err_deg)
        #   - ε_n ∼ N(0, σ_pos² · I_2)            (eq. 21, σ_pos = scene.position_err_pct/100 in λ)
        # Previously these were Gaussians (gain/phase) and a 1-D fractional
        # spacing perturbation (position) — see PAPER_FAITHFUL_DATASET.md
        # bug-fix log entry "Imperfections aligned with paper §II-C".
        if scene.gain_err_dB > 0.0:
            gmin = 10.0 ** (-float(scene.gain_err_dB) / 20.0)
            gmax = 10.0 ** (+float(scene.gain_err_dB) / 20.0)
            gain_err = rng.uniform(gmin, gmax, size=M)
        else:
            gain_err = np.ones(M, dtype=np.float64)
        if scene.phase_err_deg > 0.0:
            phase_err = np.deg2rad(
                rng.uniform(-float(scene.phase_err_deg),
                            +float(scene.phase_err_deg), size=M)
            )
        else:
            phase_err = np.zeros(M, dtype=np.float64)
        if scene.position_err_pct > 0.0:
            sigma_pos = float(scene.position_err_pct) / 100.0     # in wavelengths
            position_err = rng.normal(0.0, sigma_pos, size=(M, 2))
        else:
            position_err = np.zeros((M, 2), dtype=np.float64)

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
        # The paper-faithful coupling matrix has random per-diagonal jitter
        # ``v_k ~ U[1-δ/2, 1+δ/2]``, so it consumes the same scene-seeded
        # ``rng`` that drove the per-element calibration draws.  This keeps
        # the whole imperfection state reproducible from ``scene.seed``.
        if scene.mutual_coupling != 0.0:
            C = _mutual_coupling_matrix(M, float(scene.mutual_coupling), rng=rng)
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

        # --- paper supervision target: ideal covariance --------------------
        # R_clean = A_ideal @ A_ideal^H computed from ONLY the K direct
        # paths, with no calibration errors, no mutual coupling, and no
        # multipath replicas.  Unit-power sources => P = I_K, so the
        # ensemble form is just A_ideal A_ideal^H.  This is the paper's
        # noise-free, error-free, multipath-free EVD target (Almog & Weiss
        # §IV) and is what the trainer's L_rec / L_eig / L_proj / L_dom
        # consume — *not* the corrupted Rxx above.
        A_ideal = _steering_vector(
            angles_rad,
            M=M,
            element_spacing_lambda=self.meta.element_spacing_lambda,
            gain_err=np.ones(M, dtype=np.float64),
            phase_err=np.zeros(M, dtype=np.float64),
            position_err=np.zeros((M, 2), dtype=np.float64),   # 2-D, paper eq. 22
        )
        R_clean = A_ideal @ A_ideal.conj().T        # [M, M] Hermitian PSD, rank K

        # --- split angles back into (direct, multipath) for the public API.
        # ``angles_rad`` (the loop-scope variable) was always length K and
        # corresponds to scene.angles_deg[:K]; ``angles_rad_full`` is K+N
        # only when multipath is on, otherwise identical.  Honour the
        # documented shape contract by exposing them separately so
        # downstream consumers (especially SceneDataset.__getitem__) don't
        # confuse multipath replica AoAs with ground-truth direct-path
        # angles.  See bug report 2026-04-25 in
        # docs/PAPER_FAITHFUL_TRAINING.md.
        angles_rad_multipath = (
            angles_rad_full[K:].astype(np.float64)
            if angles_rad_full.shape[0] > K
            else np.empty((0,), dtype=np.float64)
        )

        return RenderResult(
            snapshots=X.astype(np.complex64),
            covariance=Rxx.astype(np.complex64),
            covariance_clean=R_clean.astype(np.complex64),
            steering=A_full.astype(np.complex64),
            source_signals=sources_full.astype(np.complex64),
            noise=noise.astype(np.complex64),
            angles_rad=angles_rad.astype(np.float64),  # K direct angles only
            angles_rad_multipath=angles_rad_multipath,
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

        R_ℓ[m, m'] = (1 / (T-ℓ)) · Σ_{t=0}^{T-ℓ-1}  X[m, t] · conj(X[m', t+ℓ])

    i.e. the Hermitian TRANSPOSE of the published paper's Eq. (47), which
    writes x(t)·xᴴ(t−ℓ).  The two carry identical information
    (R_code[ℓ] = R̂_eq47[ℓ]ᴴ) and this direction is what every consumer
    implements — including upstream SubspaceNet's ``data_handler.py`` — so
    the MANUSCRIPT equation should gain a ᴴ rather than the code flipping
    (all trained checkpoints depend on this convention).  Matches the
    legacy training/eval pipeline
    (``reconunet.training.subspace_training.create_autocorrelation_tensor``,
    ``reconunet.data.dataset_generator._compute_autocorr_direct_optimized``,
    and ``scripts/analysis/controlled_angle_evaluation.create_autocorrelation_input_for_unet``).
    Lag indices run ``ℓ = 0, …, τ-1`` so ℓ=0 is the standard sample
    covariance ``X X^H / T`` — the most informative spatial statistic and
    the one classical subspace methods consume directly.

    Historical note.  An earlier refactor (pre-2026-04-27) introduced a
    different formula here: ``R[ℓ] = (1/(T-ℓ-1)) Σ X[m, t+ℓ+1] · conj(X[m', t])``
    which is *off-by-one* (no zero-lag SCM in the stack) and *conjugate-
    transposed* relative to the paper.  Any checkpoint trained against
    that buggy version (e.g. ``experiments/runs/reconunet_paper/`` from
    the 2026-04-26 retrain) will produce broken DOA estimates with this
    fixed formula and must be retrained.  The 2025-09-29 paper-headline
    checkpoint (``evd_unet_denoising_model_20250929_015132.pth``) was
    trained on the correct formula and works directly.
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
        x1 = snapshots[..., : T - lag]                     # [B, M, T-lag]   x[:, t]
        x2 = snapshots[..., lag:]                          # [B, M, T-lag]   x[:, t+lag]
        R = (x1 @ x2.conj().transpose(-1, -2)) / float(T - lag)   # [B, M, M]
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
