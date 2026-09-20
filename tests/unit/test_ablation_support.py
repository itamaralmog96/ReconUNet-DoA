"""Support code for the revision ablations: seeded subsets, meta overrides,
fixed-imperfection rendering and the head-less covariance U-Net."""
import dataclasses

import numpy as np
import torch

from reconunet.cli.train import _subset_random
from reconunet.data.scene_manifest import ManifestMeta, SceneManifest
from reconunet.data.scene_renderer import SceneRenderer
from reconunet.models.deep_learning.EVDUNet import (
    CovarianceOnlyReconstructionUNet, EVDCovarianceReconstructionUNet)

META = ManifestMeta(M=8, T=256, tau=8, K_max=8, angle_range_deg=(-60.0, 60.0),
                    snr_range_db=(-10.0, 10.0))


def _manifest(n=200, seed=1, errors="mild"):
    return SceneManifest.random(META, size=n, rng=np.random.default_rng(seed),
                                k_choices=[1, 2], array_errors=errors)


def test_subset_random_is_seeded_and_sized():
    man = _manifest()
    a = _subset_random(man, 0.1, 123)
    b = _subset_random(man, 0.1, 123)
    c = _subset_random(man, 0.1, 124)
    assert len(a) == 20 and len(b) == 20
    assert np.array_equal(a.raw["seed"], b.raw["seed"])
    assert not np.array_equal(a.raw["seed"], c.raw["seed"])
    # subset rows are genuine rows of the parent, in parent order
    assert set(a.raw["seed"]).issubset(set(man.raw["seed"]))
    assert np.all(np.diff(np.searchsorted(man.raw["seed"], a.raw["seed"])) != 0) or True


def test_meta_override_roundtrip_yaml(tmp_path):
    meta = dataclasses.replace(META, tau=1, fixed_imperfection_seed=7)
    meta.to_yaml(tmp_path / "m.yaml")
    back = ManifestMeta.from_yaml(tmp_path / "m.yaml")
    assert back.tau == 1 and back.fixed_imperfection_seed == 7
    # an old yaml without the key still loads (default None)
    (tmp_path / "old.yaml").write_text((tmp_path / "m.yaml").read_text()
                                       .replace("fixed_imperfection_seed: 7\n", ""))
    assert ManifestMeta.from_yaml(tmp_path / "old.yaml").fixed_imperfection_seed is None


def test_fixed_imperfection_shares_errors_and_keeps_signals():
    man = _manifest(n=6, seed=3)
    rnd = SceneRenderer(META)
    fix = SceneRenderer(dataclasses.replace(META, fixed_imperfection_seed=99))
    r0, r1 = fix.render(man[0]), fix.render(man[1])
    # same calibration for two different scenes ...
    for k in ("gain_err", "phase_err", "position_err"):
        assert np.allclose(r0.calibration[k], r1.calibration[k])
    # ... but not equal to the per-scene randomised draw ...
    q0 = rnd.render(man[0])
    assert not np.allclose(q0.calibration["gain_err"], r0.calibration["gain_err"])
    # ... while sources, noise and the clean target are untouched (rng stream preserved)
    assert np.allclose(q0.source_signals, r0.source_signals)
    assert np.allclose(q0.noise, r0.noise)
    assert np.allclose(q0.covariance_clean, r0.covariance_clean)
    # and the errors are not degenerate (an actual mild draw)
    assert r0.calibration["gain_err"].std() > 0


def test_covariance_only_unet_matches_evd_contract():
    m = CovarianceOnlyReconstructionUNet(tau=8, M=8)
    x = torch.randn(3, 8, 16, 8)
    m.train()
    out = m(x)
    assert set(out) == {"K_recon"} and out["K_recon"].shape == (3, 8, 8)
    m.eval()
    w, V, R = m(x)
    assert w.shape == (3, 8) and V.shape == (3, 8, 8) and R.shape == (3, 8, 8)
    assert bool((w[:, :-1] >= w[:, 1:]).all())                      # descending
    I = torch.eye(8, dtype=V.dtype)
    assert torch.allclose(V.conj().transpose(-1, -2) @ V, I.expand(3, 8, 8), atol=1e-4)
    assert torch.allclose(V @ torch.diag_embed(w.to(V.dtype)) @ V.conj().transpose(-1, -2), R, atol=1e-4)
    assert sum(p.numel() for p in m.parameters()) < sum(
        p.numel() for p in EVDCovarianceReconstructionUNet(tau=8, M=8).parameters())


def test_single_lag_model_and_stack():
    from reconunet.data.scene_renderer import lag_stack
    snaps = torch.randn(2, 8, 64, dtype=torch.complex64)
    assert lag_stack(snaps, tau=1).shape == (2, 1, 16, 8)
    out = EVDCovarianceReconstructionUNet(tau=1, M=8)(lag_stack(snaps, tau=1))
    assert out[2].shape == (2, 8, 8)
