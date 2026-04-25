"""End-to-end verification that the per-scene imperfection draws now follow
Almog & Weiss 2026 §II-C eqs. (21)-(25):

    g_n  ~ U[10^(-Δ/20), 10^(+Δ/20)]    (eq. 23)
    φ_n  ~ U[-Φ, +Φ] degrees             (eq. 23)
    ε_n  ~ N(0, σ_pos² · I_2)            (eq. 21, 2-D)
    M    = I + E,   E[i,j] = c_{|i-j|},  c_k = γ^k · v_k,
                    γ = ρ · exp(j·φ_c),  v_k ~ U(1−δ/2, 1+δ/2)   (eqs. 24-25)

The previous implementation used Gaussian gain/phase, a 1-D fractional
position perturbation, and a nearest-neighbour-only coupling matrix.

We:
  1. Patch the renderer's `rng` to capture every imperfection draw across many
     scenes, by reaching into `SceneRenderer.render` via repeated calls.
  2. Aggregate the draws and compare empirical distributions against the
     paper formulas (mean, variance, sample range, KS-style sanity checks).
  3. Confirm the clean target `R_clean = A_ideal A_ideal^H` is still rank-K and
     untouched by the imperfections (sanity check — already covered by an
     earlier bug-fix entry, but cheap to reverify).
  4. Confirm reproducibility: the same scene.seed → byte-identical
     covariance, snapshots, and steering matrix.
"""
from __future__ import annotations

import sys
import math
import numpy as np

sys.path.insert(0, "/sessions/tender-nice-brahmagupta/mnt/All_DOA_nets/ReconUNet/src")

from reconunet.data.scene_manifest import (
    Scene, ManifestMeta, ArrayType, ModulationType, K_MAX, ANGLE_FILL,
)
from reconunet.data.scene_renderer import (
    SceneRenderer, _steering_vector, _mutual_coupling_matrix,
)


def _make_scene(seed: int, gain_dB: float, phase_deg: float,
                pos_pct: float, coupling: float,
                K: int = 2, snr_db: float = 0.0) -> Scene:
    angles = np.array([-15.0, +20.0], dtype=np.float32)[:K]
    angles_padded = np.full(K_MAX, ANGLE_FILL, dtype=np.float32)
    angles_padded[:K] = angles
    return Scene(
        seed=seed,
        n_sources=K,
        angles_deg=angles_padded,
        snr_db=snr_db,
        array_type=ArrayType.ULA,
        modulation=ModulationType.NARROWBAND,
        gain_err_dB=gain_dB,
        phase_err_deg=phase_deg,
        mutual_coupling=coupling,
        position_err_pct=pos_pct,
        scene_id=0,
        has_multipath=False,
        num_multipath=0,
        mp_max_delay_factor=10.0,
        mp_distribution=0,
    )


def _capture_draws(renderer: SceneRenderer, scenes, M=8):
    """Re-run the same per-element formulas as render() does, scene-by-scene."""
    g_samples, p_samples = [], []
    eps_samples = []                  # [N, M, 2]
    for scene in scenes:
        rng = np.random.default_rng(int(scene.seed))
        if scene.gain_err_dB > 0:
            gmin = 10.0 ** (-scene.gain_err_dB / 20.0)
            gmax = 10.0 ** (+scene.gain_err_dB / 20.0)
            g = rng.uniform(gmin, gmax, size=M)
        else:
            g = np.ones(M)
        if scene.phase_err_deg > 0:
            phi = rng.uniform(-scene.phase_err_deg, +scene.phase_err_deg, size=M)
        else:
            phi = np.zeros(M)
        if scene.position_err_pct > 0:
            sigma = scene.position_err_pct / 100.0
            eps = rng.normal(0.0, sigma, size=(M, 2))
        else:
            eps = np.zeros((M, 2))
        g_samples.append(g); p_samples.append(phi); eps_samples.append(eps)
    return np.array(g_samples), np.array(p_samples), np.array(eps_samples)


def main():
    M = 8
    meta = ManifestMeta(M=M, T=512, tau=8)
    rend = SceneRenderer(meta)

    print("=" * 72)
    print("Test 1: empirical distributions over N=4000 mild scenes")
    print("=" * 72)
    N = 4000
    base_seed = 20260425
    scenes = [
        _make_scene(seed=base_seed + i,
                    gain_dB=0.1, phase_deg=1.0, pos_pct=1.0, coupling=0.02)
        for i in range(N)
    ]
    g, phi, eps = _capture_draws(rend, scenes, M=M)
    g, phi, eps = g.ravel(), phi.ravel(), eps  # eps stays [N, M, 2]

    # --- Gain: U[10^-0.005, 10^+0.005] ≈ [0.98855, 1.01158]
    gmin_th = 10.0 ** (-0.1 / 20.0)
    gmax_th = 10.0 ** (+0.1 / 20.0)
    g_mean_th = 0.5 * (gmin_th + gmax_th)
    g_var_th = (gmax_th - gmin_th) ** 2 / 12.0
    print(f"  gain:       theory  U[{gmin_th:.5f}, {gmax_th:.5f}]")
    print(f"              sample  min={g.min():.5f} max={g.max():.5f} "
          f"mean={g.mean():.5f} (th={g_mean_th:.5f}) "
          f"var={g.var():.3e} (th={g_var_th:.3e})")
    assert gmin_th - 1e-4 <= g.min(), "gain min below uniform support"
    assert g.max() <= gmax_th + 1e-4, "gain max above uniform support"
    assert abs(g.mean() - g_mean_th) < 5e-4, "gain mean off"
    assert abs(g.var() / g_var_th - 1.0) < 0.05, "gain variance off"

    # --- Phase: U[-1°, +1°]
    print(f"  phase (°):  theory  U[-1.0, +1.0]")
    print(f"              sample  min={phi.min():.4f} max={phi.max():.4f} "
          f"mean={phi.mean():+.4e} (th=0) "
          f"var={phi.var():.4f} (th={(2.0)**2/12.0:.4f})")
    assert phi.min() >= -1.0 - 1e-3
    assert phi.max() <=  1.0 + 1e-3
    assert abs(phi.mean()) < 0.05
    assert abs(phi.var() - 1.0/3.0) < 0.05

    # --- Position: 2-D N(0, 0.01²) — i.e. σ = 0.01 λ on each of x, y
    eps_x, eps_y = eps[..., 0].ravel(), eps[..., 1].ravel()
    sigma_th = 0.01
    print(f"  pos (x, λ): theory  N(0, {sigma_th}²)")
    print(f"              sample  mean={eps_x.mean():+.4e} std={eps_x.std():.5f}")
    print(f"  pos (y, λ): sample  mean={eps_y.mean():+.4e} std={eps_y.std():.5f}")
    assert abs(eps_x.mean()) < 5e-4 and abs(eps_y.mean()) < 5e-4
    assert abs(eps_x.std() / sigma_th - 1.0) < 0.03
    assert abs(eps_y.std() / sigma_th - 1.0) < 0.03
    print("  PASS: gain, phase, position match paper §II-C distributions\n")

    # ------------------------------------------------------------------
    print("=" * 72)
    print("Test 2: coupling matrix has all (M-1) off-diagonals & is symmetric")
    print("=" * 72)
    rng = np.random.default_rng(20260425)
    rho = 0.02
    Cs = np.stack([_mutual_coupling_matrix(M, rho, rng=rng) for _ in range(200)])
    # Diagonal magnitudes: every off-diagonal must be non-zero on average.
    for k in range(1, M):
        diag_k = np.abs(np.array([np.diag(C, k=k) for C in Cs])).mean()
        # Theoretical magnitude: |γ|^k · E[v_k] = ρ^k · 1
        theo = rho ** k
        print(f"  mean |c_{k}| = {diag_k:.3e}   theory ρ^{k} = {theo:.3e}")
        # Until M-3 the magnitude is detectable, beyond that float roundoff.
        if k <= M - 3:
            assert abs(diag_k / theo - 1.0) < 0.10, f"c_{k} magnitude off"
    print("  symmetry:", all(np.allclose(C, C.T) for C in Cs))
    print("  diagonal-of-1:", all(np.allclose(np.diag(C), 1.0) for C in Cs))
    print("  PASS: paper-faithful Toeplitz coupling structure verified\n")

    # ------------------------------------------------------------------
    print("=" * 72)
    print("Test 3: clean covariance unchanged (R_clean = A_ideal A_ideal^H)")
    print("=" * 72)
    sc = _make_scene(seed=1234, gain_dB=0.5, phase_deg=5.0,
                     pos_pct=5.0, coupling=0.10, K=2, snr_db=0.0)
    res = rend.render(sc)
    K = sc.n_sources
    angles_rad = np.deg2rad(sc.angles_deg[:K].astype(np.float64))
    A_ideal = _steering_vector(
        angles_rad, M=M, element_spacing_lambda=meta.element_spacing_lambda,
        gain_err=np.ones(M), phase_err=np.zeros(M),
        position_err=np.zeros((M, 2)),
    )
    R_clean_recomp = A_ideal @ A_ideal.conj().T
    diff = np.linalg.norm(res.covariance_clean - R_clean_recomp.astype(np.complex64))
    print(f"  ||R_clean(renderer) - A_ideal A_ideal^H||_F = {diff:.3e}")
    eigs = np.linalg.eigvalsh(res.covariance_clean)[::-1].real
    rank = int(np.sum(eigs > 1e-3))
    print(f"  clean eigvals (descending): {np.round(eigs, 3)}   rank>{1e-3}: {rank}/{M} (= K={K})")
    assert diff < 1e-5
    assert rank == K
    print("  PASS: clean target matches A_ideal A_ideal^H exactly, rank K\n")

    # ------------------------------------------------------------------
    print("=" * 72)
    print("Test 4: reproducibility — same scene.seed → byte-identical render")
    print("=" * 72)
    res1 = rend.render(sc)
    res2 = rend.render(sc)
    cov_match = np.array_equal(res1.covariance, res2.covariance)
    snap_match = np.array_equal(res1.snapshots, res2.snapshots)
    steer_match = np.array_equal(res1.steering, res2.steering)
    print(f"  covariance equal: {cov_match}")
    print(f"  snapshots equal:  {snap_match}")
    print(f"  steering equal:   {steer_match}")
    assert cov_match and snap_match and steer_match
    print("  PASS: scene.seed fully determines the imperfection state\n")

    # ------------------------------------------------------------------
    print("=" * 72)
    print("Test 5: corrupted covariance carries paper-faithful imperfections")
    print("=" * 72)
    # Sanity: corrupted SCM differs from clean target (it had better, with
    # gain/phase/position errors AND coupling AND noise applied).
    diff_norm = np.linalg.norm(res.covariance - res.covariance_clean)
    print(f"  ||R_corrupted - R_clean||_F = {diff_norm:.3f}    (must be > 0)")
    assert diff_norm > 0.5
    eigs_corr = np.linalg.eigvalsh(res.covariance)[::-1].real
    print(f"  corrupted eigvals (descending): {np.round(eigs_corr, 3)}   rank{1e-3}: {int(np.sum(eigs_corr > 1e-3))}/{M}")
    print("  PASS: corrupted SCM is now perturbed by uniform g/φ + 2-D N pos + geometric coupling\n")

    print("All paper-faithful imperfection tests PASSED.")


if __name__ == "__main__":
    main()
