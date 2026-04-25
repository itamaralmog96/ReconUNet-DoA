#!/usr/bin/env python3
"""``visualize_scene.py`` — render one scene and plot diagnostic figures.

Produces a 2-row figure:

    Row 1: |R|  (covariance magnitude),   |A|  (steering magnitudes),
           source-signal constellation / IQ scatter.
    Row 2: classical MUSIC spectrum with ground-truth markers.

Usage
-----
    python scripts/verify/visualize_scene.py \\
        data/scenes/verify/tiny/train.npy --index 0 \\
        --out experiments/verify/scene_0.png

    # Also show source-signal constellation (useful for BPSK / QPSK checks):
    python scripts/verify/visualize_scene.py \\
        data/scenes/verify/modulation_qpsk/train.npy --index 0 \\
        --out experiments/verify/qpsk.png --show-constellation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from reconunet.data.scene_manifest import ModulationType, SceneManifest  # noqa: E402
from reconunet.data.scene_renderer import SceneRenderer                   # noqa: E402


def _music_spectrum(R: np.ndarray, K: int, M: int, d_over_lambda: float,
                    scan_deg: np.ndarray) -> np.ndarray:
    """Classical MUSIC on an ULA with nominal (calibration-free) steering."""
    # Eigendecomp of sample covariance; eigh returns ascending.
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    En = eigvecs[:, K:]                       # noise subspace, [M, M-K]

    # Nominal ULA steering matrix, broadside=0°.
    # IMPORTANT: sign convention must match SceneRenderer._steering_vector,
    # which uses  a_m(θ) = exp(-j·2π·m·(d/λ)·sin(θ))  (note the −j).  Using
    # +j here scrambles the orthogonality to E_n (conjugate of true A-columns)
    # and gives one shadow peak near broadside instead of K real peaks.
    m = np.arange(M).reshape(M, 1)
    phi = -2 * np.pi * d_over_lambda * np.sin(np.deg2rad(scan_deg)).reshape(1, -1)
    A = np.exp(1j * m * phi)                  # [M, G]

    P = En.conj().T @ A                       # [M-K, G]
    denom = np.sum(np.abs(P) ** 2, axis=0)    # [G]
    return 1.0 / (denom + 1e-12)


def _format_preset(scene) -> str:
    return (f"gain={scene.gain_err_dB:.3g}dB, phase={scene.phase_err_deg:.3g}°, "
            f"coup={scene.mutual_coupling:.3g}, pos={scene.position_err_pct:.3g}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--index", "-i", type=int, default=0, help="Row index in the manifest.")
    ap.add_argument("--out", "-o", type=Path,
                    default=Path("experiments/verify/scene.png"))
    ap.add_argument("--scan-points", type=int, default=721,
                    help="Angular-grid resolution for the MUSIC spectrum.")
    ap.add_argument("--show-constellation", action="store_true",
                    help="Scatter the first source's IQ samples (good for BPSK / QPSK).")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = SceneManifest.load(args.manifest)
    if args.index < 0 or args.index >= len(man):
        raise SystemExit(f"index {args.index} outside [0, {len(man)})")

    scene = man[args.index]
    renderer = SceneRenderer(man.meta)
    res = renderer.render(scene)
    R, A, X = res.covariance, res.steering, res.snapshots
    M = man.meta.M
    K = int(scene.n_sources)
    true_angles_deg = scene.angles_deg[:K]
    d_over_lambda = float(man.meta.element_spacing_lambda)

    g0, g1 = man.meta.angle_range_deg
    pad = 5.0
    scan_deg = np.linspace(g0 - pad, g1 + pad, args.scan_points)
    spectrum = _music_spectrum(R, K=K, M=M, d_over_lambda=d_over_lambda, scan_deg=scan_deg)
    spectrum_db = 10 * np.log10(spectrum / spectrum.max())

    # Pick the K largest *local maxima* — not the top-K grid samples, which
    # would just return K adjacent grid points of a single tall peak.
    from scipy.signal import find_peaks
    peak_idx, _ = find_peaks(spectrum, distance=3)
    if peak_idx.size >= K:
        top = peak_idx[np.argsort(spectrum[peak_idx])[-K:]]
        est_deg = np.sort(scan_deg[top])
    else:
        # Spectrum has fewer than K peaks; fall back to top-K grid samples
        # but warn the caller so they know this scene's spectrum is degenerate.
        print(f"  [!] only {peak_idx.size} local peak(s) found for K={K}; "
              f"falling back to top-K grid samples.", file=sys.stderr)
        est_idx = np.argsort(spectrum)[-K:]
        est_deg = np.sort(scan_deg[est_idx])

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

    ax_R = fig.add_subplot(gs[0, 0])
    im = ax_R.imshow(np.abs(R), cmap="viridis")
    ax_R.set_title(f"|R|  (M×M = {M}×{M})")
    ax_R.set_xlabel("element"); ax_R.set_ylabel("element")
    fig.colorbar(im, ax=ax_R, fraction=0.046)

    ax_A = fig.add_subplot(gs[0, 1])
    for k in range(K):
        ax_A.plot(np.abs(A[:, k]), marker="o", label=f"src {k} @ {true_angles_deg[k]:+.1f}°")
    ax_A.set_title("|A[:, k]| — steering magnitudes")
    ax_A.set_xlabel("element"); ax_A.set_ylabel("|a_m|")
    ax_A.grid(alpha=0.3); ax_A.legend(fontsize=8)

    ax_scatter = fig.add_subplot(gs[0, 2])
    if args.show_constellation:
        s = res.source_signals[0]
        ax_scatter.scatter(s.real, s.imag, s=4, alpha=0.35)
        ax_scatter.set_title(f"source 0 IQ ({ModulationType(int(scene.modulation)).name})")
        ax_scatter.set_xlabel("I"); ax_scatter.set_ylabel("Q")
        ax_scatter.axhline(0, color="gray", lw=0.5); ax_scatter.axvline(0, color="gray", lw=0.5)
        ax_scatter.set_aspect("equal")
    else:
        x = X[0]
        ax_scatter.plot(x.real, lw=0.7, label="Re")
        ax_scatter.plot(x.imag, lw=0.7, label="Im")
        ax_scatter.set_title("X[0, :] — element 0 snapshots")
        ax_scatter.set_xlabel("t (sample)"); ax_scatter.set_ylabel("amplitude")
        ax_scatter.legend(fontsize=8); ax_scatter.grid(alpha=0.3)

    ax_spec = fig.add_subplot(gs[1, :])
    ax_spec.plot(scan_deg, spectrum_db, lw=1.1)
    for k in range(K):
        ax_spec.axvline(true_angles_deg[k], color="tab:green", ls="--", lw=1,
                        label="true" if k == 0 else None)
    for k in range(K):
        ax_spec.axvline(est_deg[k], color="tab:red", ls=":", lw=1,
                        label="MUSIC peak" if k == 0 else None)
    err = np.abs(np.sort(true_angles_deg) - est_deg)
    ax_spec.set_title(
        f"MUSIC spectrum  |  scene_id={scene.scene_id}  K={K}  "
        f"SNR={scene.snr_db:.1f}dB  [{_format_preset(scene)}]\n"
        f"true={np.sort(true_angles_deg).tolist()}   "
        f"est={est_deg.tolist()}   |err| = {err.tolist()} deg"
    )
    ax_spec.set_xlabel("angle (deg, broadside = 0)")
    ax_spec.set_ylabel("MUSIC (dB, normalised)")
    ax_spec.grid(alpha=0.3); ax_spec.legend(loc="upper right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")
    print(f"  true angles (deg): {np.sort(true_angles_deg).tolist()}")
    print(f"  MUSIC est   (deg): {est_deg.tolist()}")
    print(f"  abs error   (deg): {err.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
