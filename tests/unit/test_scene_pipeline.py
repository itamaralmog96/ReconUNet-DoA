"""End-to-end smoke test for SceneManifest → SceneRenderer → collate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reconunet.data.scene_manifest import (
    K_MAX,
    ArrayType,
    ManifestMeta,
    ModulationType,
    Scene,
    SceneManifest,
)
from reconunet.data.scene_renderer import SceneRenderer, lag_stack
from reconunet.data.scene_dataset import (
    ReconUNetCollate,
    SceneDataset,
    SubspaceNetCollate,
    SubViTCollate,
    get_collate,
)


META = ManifestMeta(M=8, T=128, tau=4, K_max=3)  # tiny for CI


# ---------------------------------------------------------------------------
# Manifest schema invariants
# ---------------------------------------------------------------------------


def test_row_size_is_96_bytes():
    man = SceneManifest.create(META, size=1)
    assert man.raw.dtype.itemsize == 96


def test_roundtrip_manifest(tmp_path: Path):
    rng = np.random.default_rng(0)
    man = SceneManifest.random(META, size=16, rng=rng, k_choices=(1, 2, 3))

    npy = tmp_path / "scenes.npy"
    man.save(npy)

    man2 = SceneManifest.load(npy)
    assert len(man2) == 16
    for i in range(len(man)):
        a, b = man[i], man2[i]
        np.testing.assert_array_equal(a.angles_deg, b.angles_deg)
        assert a.snr_db == pytest.approx(b.snr_db, abs=1e-5)
        assert a.n_sources == b.n_sources


# ---------------------------------------------------------------------------
# Renderer determinism
# ---------------------------------------------------------------------------


def _scene(seed: int = 42, k: int = 2, snr: float = 10.0) -> Scene:
    angles = np.full((K_MAX,), np.nan, dtype=np.float32)
    angles[:k] = np.array([-10.0, 15.0][:k], dtype=np.float32)
    return Scene(
        seed=seed,
        n_sources=k,
        angles_deg=angles,
        snr_db=snr,
        array_type=ArrayType.ULA,
        modulation=ModulationType.NARROWBAND,
        gain_err_dB=0.0,
        phase_err_deg=0.0,
        mutual_coupling=0.0,
        position_err_pct=0.0,
        scene_id=0,
    )


def test_renderer_is_deterministic():
    r1 = SceneRenderer(META).render(_scene(seed=123))
    r2 = SceneRenderer(META).render(_scene(seed=123))
    np.testing.assert_array_equal(r1.snapshots, r2.snapshots)


def test_renderer_snr_is_calibrated_ballpark():
    """Empirical SNR ≈ configured SNR (within ~1 dB) when M,T are reasonable."""
    meta = ManifestMeta(M=8, T=2048, tau=4)    # longer T for stable estimate
    scene = _scene(seed=7, k=1, snr=10.0)
    scene = Scene(**{**scene.__dict__})
    result = SceneRenderer(meta).render(scene)
    signal_power = np.mean(np.abs(result.steering @ result.source_signals) ** 2)
    noise_power = np.mean(np.abs(result.noise) ** 2)
    snr_empirical_db = 10.0 * np.log10(signal_power / noise_power)
    assert abs(snr_empirical_db - 10.0) < 1.0, snr_empirical_db


# ---------------------------------------------------------------------------
# Lag-stack shape invariants
# ---------------------------------------------------------------------------


def test_lag_stack_shape():
    snaps = torch.randn(3, 8, 128, dtype=torch.complex64)
    out = lag_stack(snaps, tau=4)
    assert out.shape == (3, 4, 16, 8)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Collate parity — all three see byte-identical scene data
# ---------------------------------------------------------------------------


def test_collates_agree_on_labels():
    man = SceneManifest.random(META, size=4, rng=np.random.default_rng(1), k_choices=(2,))
    ds = SceneDataset(man)
    batch = [ds[i] for i in range(4)]

    r = ReconUNetCollate(META)(batch)
    s = SubspaceNetCollate(META)(batch)
    v = SubViTCollate(META)(batch)

    # Label tensors must be identical across collates
    assert torch.equal(r["angles_rad"], s["angles_rad"])
    assert torch.equal(r["angles_rad"], v["angles_rad"])
    assert torch.equal(r["n_sources"], s["n_sources"])

    # Shape contracts
    assert r["input"].shape == (4, META.tau, 2 * META.M, META.M)
    assert s["input"].shape == (4, META.tau, 2 * META.M, META.M)
    assert v["input"].shape == (4, 2, META.M, META.M)


def test_collate_registry():
    assert isinstance(get_collate("reconunet", META), ReconUNetCollate)
    assert isinstance(get_collate("subspacenet", META), SubspaceNetCollate)
    assert isinstance(get_collate("subvit", META), SubViTCollate)
    with pytest.raises(KeyError):
        get_collate("not_a_model", META)
