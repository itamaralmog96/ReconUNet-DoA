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
# Source band-limiting and coherent multipath (v1.2 renderer, 2026-09-06)
# ---------------------------------------------------------------------------


def _mp_scene(seed: int, k: int = 1, n_mp: int = 2, snr: float = 40.0) -> Scene:
    base = _scene(seed=seed, k=k, snr=snr)
    return Scene(**{**base.__dict__, "has_multipath": True, "num_multipath": n_mp})


def test_sources_are_bandlimited_and_unit_power():
    """Legacy 0.05·fs band-limiting: unit power and strong lag correlation."""
    meta = ManifestMeta(M=8, T=512, tau=8)
    r = SceneRenderer(meta).render(_scene(seed=3, k=2, snr=40.0))
    s = r.source_signals.astype(np.complex128)
    np.testing.assert_allclose(np.mean(np.abs(s) ** 2, axis=1), 1.0, atol=1e-5)
    T = s.shape[1]
    for lag in range(1, meta.tau):
        rho = abs(np.vdot(s[0, :T - lag], s[0, lag:])) / np.vdot(s[0], s[0]).real
        assert rho > 0.75, (lag, rho)                    # sinc(0.05·7) ≈ 0.81


def test_white_sources_when_bandlimiting_disabled():
    meta = ManifestMeta(M=8, T=512, tau=8, source_bw_frac=None)
    r = SceneRenderer(meta).render(_scene(seed=3, k=1, snr=40.0))
    s = r.source_signals[0].astype(np.complex128)
    rho = abs(np.vdot(s[:-1], s[1:])) / np.vdot(s, s).real
    assert rho < 0.2, rho


def test_multipath_replicas_are_highly_correlated():
    """Paper §II-B regime: replica ≈ delayed copy ⇒ |γ| ≈ 1, near rank-1."""
    meta = ManifestMeta(M=8, T=512, tau=8)
    rend = SceneRenderer(meta)
    gammas, ratios = [], []
    for seed in range(40):
        r = rend.render(_mp_scene(seed, k=1, n_mp=2))
        S = r.source_signals.astype(np.complex128)
        assert S.shape[0] == 3 and r.mp_delay_samples.shape == (2,)
        # legacy mapping: delay = τ·fs/2 ∈ (0, 10] samples for uniform τ ≤ 1/bw
        assert np.all((r.mp_delay_samples > 0) & (r.mp_delay_samples <= 10.0))
        d, m = S[0], S[1]
        gammas.append(abs(np.vdot(d, m)) / (np.linalg.norm(d) * np.linalg.norm(m)))
        A = r.steering.astype(np.complex128)
        Rs = A @ (S @ S.conj().T / S.shape[1]) @ A.conj().T      # noise-free
        ev = np.sort(np.linalg.eigvalsh(Rs))[::-1]
        ratios.append(ev[1] / ev[0])
    assert np.mean(gammas) > 0.8, np.mean(gammas)
    assert np.min(gammas) > 0.5, np.min(gammas)           # sinc(0.05·10) ≈ 0.64
    assert np.median(ratios) < 0.1, np.median(ratios)


def test_no_multipath_has_empty_delays():
    r = SceneRenderer(META).render(_scene(seed=1))
    assert r.mp_delay_samples.shape == (0,)


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
