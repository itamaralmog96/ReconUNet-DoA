"""Tests for :mod:`reconunet.evaluation.unified_harness`.

Split into two tiers:

* **Tier-1 (numpy-only):** ``rmspe_deg`` and ``stochastic_crlb_deg`` are pure
  numpy and can be exercised in any environment.  These are the tests that
  *must* pass before we trust any headline number in the paper figures.
* **Tier-2 (torch + the scene pipeline):** end-to-end ``evaluate()`` call on a
  synthetic 8-sample manifest, marked ``@pytest.mark.requires_torch`` and
  skipped automatically when the import fails.  This is the smoke test the
  user should run on their own machine after ``pip install -e .``.

None of these tests touch disk outside ``tmp_path``.  Seeded RNGs only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reconunet.evaluation.unified_harness import rmspe_deg, stochastic_crlb_deg


# ---------------------------------------------------------------------------
# Tier-1: pure numpy
# ---------------------------------------------------------------------------


class TestRmspeDeg:
    def test_zero_error_for_identical_inputs(self):
        angles = np.deg2rad(np.array([[-30.0, 0.0, 30.0]]))
        assert rmspe_deg(angles, angles) == pytest.approx(0.0, abs=1e-9)

    def test_permutation_invariance(self):
        # Ground truth is sorted ascending; prediction swaps the order —
        # RMSPE must still be zero because the harness sorts both sides.
        true = np.deg2rad(np.array([[-20.0, 10.0, 45.0]]))
        pred = np.deg2rad(np.array([[45.0, -20.0, 10.0]]))
        assert rmspe_deg(pred, true) == pytest.approx(0.0, abs=1e-9)

    def test_known_rms_value(self):
        # Deterministic 2-source case: errors are {1°, 2°} per sample →
        # per-sample RMS = sqrt((1 + 4) / 2) = sqrt(2.5) ≈ 1.5811°.
        true = np.deg2rad(np.array([[0.0, 30.0]]))
        pred = np.deg2rad(np.array([[1.0, 32.0]]))
        expected = math.sqrt((1.0**2 + 2.0**2) / 2.0)
        assert rmspe_deg(pred, true) == pytest.approx(expected, rel=1e-6)

    def test_periodic_wrap_near_boundary(self):
        # +89° vs -89°.  On the SO(2) manifold with period π, the short-way
        # distance is only 2°, not 178°.  (Verify the modulo arithmetic.)
        true = np.deg2rad(np.array([[89.0]]))
        pred = np.deg2rad(np.array([[-89.0]]))
        assert rmspe_deg(pred, true) == pytest.approx(2.0, rel=1e-6)

    def test_reduce_none_returns_per_sample(self):
        true = np.deg2rad(np.array([[0.0, 30.0], [0.0, 30.0]]))
        pred = np.deg2rad(np.array([[1.0, 32.0], [0.0, 30.0]]))
        out = rmspe_deg(pred, true, reduce="none")
        assert out.shape == (2,)
        assert out[1] == pytest.approx(0.0, abs=1e-9)

    def test_invalid_reduce_raises(self):
        a = np.zeros((1, 1))
        with pytest.raises(ValueError):
            rmspe_deg(a, a, reduce="max")


class TestStochasticCrlbDeg:
    def test_monotonic_in_snr(self):
        # CRLB must strictly *decrease* with increasing SNR (more power
        # means a tighter estimator floor).
        angles = np.deg2rad(np.array([-10.0, 10.0]))
        low  = stochastic_crlb_deg(M=8, T=512, angles_rad=angles, snr_db=-5.0)
        mid  = stochastic_crlb_deg(M=8, T=512, angles_rad=angles, snr_db=5.0)
        high = stochastic_crlb_deg(M=8, T=512, angles_rad=angles, snr_db=20.0)
        assert low > mid > high > 0.0

    def test_monotonic_in_snapshots(self):
        # Doubling T halves the variance (CRLB ∝ 1/T).
        angles = np.deg2rad(np.array([-15.0, 15.0]))
        v1 = stochastic_crlb_deg(M=8, T=256,  angles_rad=angles, snr_db=0.0)
        v2 = stochastic_crlb_deg(M=8, T=512,  angles_rad=angles, snr_db=0.0)
        v4 = stochastic_crlb_deg(M=8, T=1024, angles_rad=angles, snr_db=0.0)
        assert v1 > v2 > v4 > 0.0
        # 1/T scaling: v2 / v1 should be ≈ 0.5; allow 25 % slack because the
        # stochastic CRLB also has a finite-sample correction term.
        assert v2 / v1 == pytest.approx(0.5, rel=0.25)

    def test_returns_finite_scalar(self):
        v = stochastic_crlb_deg(
            M=8, T=512, angles_rad=np.deg2rad(np.array([0.0])), snr_db=10.0
        )
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert v > 0.0

    def test_reasonable_magnitude_for_paper_operating_point(self):
        # At the paper's headline condition (M=8, T=512, SNR=10 dB, K=3 in
        # [-60, 60]°), the per-source √CRLB should be a sub-degree floor —
        # the whole point of the exercise.  Anything above 5° indicates a
        # units bug.
        angles = np.deg2rad(np.array([-30.0, 0.0, 30.0]))
        var_deg2 = stochastic_crlb_deg(M=8, T=512, angles_rad=angles, snr_db=10.0)
        std_deg = math.sqrt(var_deg2)
        assert 0.01 < std_deg < 5.0


# ---------------------------------------------------------------------------
# Tier-2: end-to-end — only runs if torch + the scene pipeline are available
# ---------------------------------------------------------------------------


try:
    import torch  # noqa: F401

    from reconunet.data.scene_manifest import ManifestMeta, SceneManifest

    _HAVE_TORCH_STACK = True
except Exception:                                               # pragma: no cover
    _HAVE_TORCH_STACK = False


pytestmark_torch = pytest.mark.skipif(
    not _HAVE_TORCH_STACK, reason="torch + reconunet.data required for end-to-end"
)


class _IdentityClassic:
    """Trivial 'classic algorithm' used only to drive the harness loop.

    Returns the ground-truth angles unchanged, so the reported RMSPE for
    this model in every SNR bucket should be ≈ 0°.  This is the cheapest
    possible fixture that exercises the full dispatch path in
    :func:`evaluate`.
    """

    def __call__(self, batch, meta):
        return batch["angles_rad"].numpy()


@pytestmark_torch
def test_evaluate_end_to_end_with_identity_classic(tmp_path):
    from reconunet.data.scene_manifest import ArrayType, ModulationType
    from reconunet.evaluation.unified_harness import (
        HarnessConfig,
        ModelSpec,
        evaluate,
    )

    meta = ManifestMeta(
        M=8, T=128, fs_Hz=2.45e9, tau=8,
        K_max=3,
        angle_range_deg=(-60.0, 60.0),
        snr_range_db=(-10.0, 20.0),
        element_spacing_lambda=0.5,
    )

    manifest = SceneManifest.random(
        n=8,
        meta=meta,
        k_choices=(3,),
        min_separation_deg=5.0,
        array_errors="none",
        array_type=ArrayType.ULA,
        modulation=ModulationType.NARROWBAND,
        rng_seed=20260420,
    )
    manifest_path = tmp_path / "tiny.npy"
    manifest.save(str(manifest_path))

    # Force every sample into the 10 dB bucket — simplest possible sweep.
    manifest.raw["snr_db"][:] = 10.0
    manifest.save(str(manifest_path))

    cfg = HarnessConfig(
        manifest_path=str(manifest_path),
        snr_sweep_db=(10.0,),
        models=[
            ModelSpec(
                name="identity",
                adapter=_IdentityClassic(),
                is_classic=True,
            ),
        ],
        batch_size=4,
        output_dir=str(tmp_path / "results"),
    )
    df = evaluate(cfg)
    assert list(df.columns) == [
        "model", "snr_db", "rmse_deg", "crlb_deg", "n_samples", "latency_ms",
    ]
    assert (df["model"] == "identity").all()
    assert df["rmse_deg"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert df["n_samples"].iloc[0] == 8
    # CRLB floor should be positive — the "identity" model beating it is
    # fine (it cheats by reading the labels); we only check the floor is
    # reported as a real number.
    assert df["crlb_deg"].iloc[0] > 0.0

    csv_path = tmp_path / "results" / "results.csv"
    cfg_path = tmp_path / "results" / "config.json"
    assert csv_path.is_file()
    assert cfg_path.is_file()
