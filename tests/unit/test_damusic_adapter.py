"""Correctness tests for the DA-MUSIC baseline integration.

Locks down:
1. **Spectrum parity** — the adapter's batched MUSIC spectrum equals the
   vendored per-sample ``spectrum_calculation`` for the same noise subspace.
2. **End-to-end differentiability** — a forward on random snapshots yields
   finite angles and finite gradients on every parameter group (GRU, fc,
   fc1-3), through the Hermitian-PSD eigh path.
3. **Collate / data plumbing** — ``damusic`` is registered, the collate emits
   raw complex snapshots, and ``k_filter`` subsets a manifest by source count.
4. **Buffers follow the device** (grid steering vectors are registered).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from reconunet.cli.train import _subset_by_k
from reconunet.data.scene_dataset import DAMUSICCollate, get_collate
from reconunet.data.scene_manifest import ManifestMeta, SceneManifest
from reconunet.models.third_party.damusic_adapter import DAMUSICAdapter

N, T, M = 8, 64, 2


def _build(M_: int = M):
    adapter = DAMUSICAdapter(N=N, T=T, M=M_)
    model = adapter.build_model({"N": N, "T": T, "M": M_})
    return adapter, model


def _random_snapshots(B: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(B, N, T, generator=g), torch.randn(B, N, T, generator=g))


def test_batched_spectrum_matches_upstream_per_sample():
    adapter, model = _build()
    g = torch.Generator().manual_seed(1)
    A = torch.complex(torch.randn(3, N, N, generator=g), torch.randn(3, N, N, generator=g))
    R = A @ A.conj().transpose(1, 2)                      # Hermitian PSD
    _, U = torch.linalg.eigh(R)                           # ascending
    Un = U[:, :, : N - M]                                 # noise subspace
    F = Un @ Un.conj().transpose(1, 2)
    ours = adapter.music_spectrum(model, F)               # [3, 361]
    for b in range(3):
        ref, _ = model.spectrum_calculation(Un[b])        # vendored loop
        torch.testing.assert_close(ours[b], ref.to(ours.dtype), rtol=1e-4, atol=1e-4)


def test_forward_shapes_and_finite_gradients():
    adapter, model = _build()
    model.train()
    X = _random_snapshots(B=6)
    out = adapter.forward(model, X, targets=None, meta={"n_sources": torch.full((6,), M)})
    assert out.angles_pred.shape == (6, M)
    assert torch.isfinite(out.angles_pred).all()
    assert out.extras["Rz"].shape == (6, N, N)
    assert "spatial_spectrum" not in out.extras            # would trigger the BCE path
    # RMSPE-style loss against random targets, then check every parameter group.
    target = (torch.rand(6, M) - 0.5) * math.pi / 2
    loss = torch.sqrt(((out.angles_pred.sort(-1).values - target.sort(-1).values) ** 2).mean())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert model.rnn.weight_ih_l0.grad.abs().sum() > 0     # gradient reaches the GRU


def test_eig_route_also_runs():
    adapter = DAMUSICAdapter(N=N, T=T, M=M, hermitian_psd=False)
    model = adapter.build_model({"N": N, "T": T, "M": M, "hermitian_psd": False})
    model.eval()
    with torch.no_grad():
        out = adapter.forward(model, _random_snapshots(B=4, seed=2))
    assert out.angles_pred.shape == (4, M) and torch.isfinite(out.angles_pred).all()


def test_wrong_T_is_rejected():
    adapter, model = _build()
    with pytest.raises(ValueError):
        adapter.forward(model, torch.zeros(2, N, T + 1, dtype=torch.complex64))


def test_collate_registered_and_emits_snapshots():
    meta = ManifestMeta(M=N, T=T, tau=4, K_max=3)
    assert isinstance(get_collate("damusic", meta), DAMUSICCollate)
    man = SceneManifest.random(meta, size=4, rng=np.random.default_rng(0), k_choices=(2,))
    from reconunet.data.scene_dataset import SceneDataset
    ds = SceneDataset(man)
    batch = get_collate("damusic", meta)([ds[i] for i in range(4)])
    assert batch["input"].shape == (4, N, T) and batch["input"].is_complex()
    assert batch["angles_rad"].shape[0] == 4 and (batch["n_sources"] == 2).all()


def test_k_filter_subsets_manifest():
    meta = ManifestMeta(M=N, T=T, tau=4, K_max=3)
    man = SceneManifest.random(meta, size=90, rng=np.random.default_rng(0), k_choices=(1, 2, 3))
    sub = _subset_by_k(man, [2])
    assert 0 < len(sub) < len(man)
    assert all(sub[i].n_sources == 2 for i in range(len(sub)))
    assert sub.meta is man.meta
    both = _subset_by_k(man, [1, 3])
    assert len(both) + len(sub) == len(man)


def test_ensemble_loads_per_k_and_dispatches(tmp_path):
    """Trainer-format checkpoints under k<K>/checkpoints/best.pt → per-K dispatch."""
    from reconunet.models.third_party.damusic_adapter import DAMUSICEnsemble

    for k in (1, 2):
        _, model = _build(k)
        d = tmp_path / f"k{k}" / "checkpoints"
        d.mkdir(parents=True)
        torch.save({"model": model.state_dict(),
                    "cfg": {"model": {"init": {"N": N, "T": T, "M": k, "hermitian_psd": True}}}},
                   d / "best.pt")
    ens = DAMUSICEnsemble.from_run_dir(tmp_path, device="cpu", ks=(1, 2, 3, 4), verbose=False)
    assert ens is not None and ens.ks == [1, 2]
    snaps = _random_snapshots(B=5, seed=3)
    ns = torch.tensor([1, 2, 3, 2, 1])            # K=3 has no model → NaN row
    out = ens.predict(snaps, ns)
    assert out.shape == (5, 3)
    assert torch.isfinite(out[0, :1]).all() and torch.isnan(out[0, 1:]).all()
    assert torch.isfinite(out[1, :2]).all() and torch.isnan(out[1, 2:]).all()
    assert torch.isnan(out[2]).all()
    # Wrong directory → None, not an exception (scripts skip the method).
    assert DAMUSICEnsemble.from_run_dir(tmp_path / "nowhere", verbose=False) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_buffers_follow_device():
    adapter, model = _build()
    model = model.cuda().eval()
    assert model.sv.is_cuda and model.angels.is_cuda
    with torch.no_grad():
        out = adapter.forward(model, _random_snapshots(B=3).cuda())
    assert out.angles_pred.is_cuda and torch.isfinite(out.angles_pred).all()
