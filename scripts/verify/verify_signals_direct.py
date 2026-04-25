# scripts/verify/verify_signals_direct.py
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_renderer import SceneRenderer, _steering_vector

def music_self_consistent(R, K, M, d_over_lambda, scan_deg):
    """MUSIC using the renderer's OWN steering convention. Peaks must land on truth."""
    w, V = np.linalg.eigh(R)
    En = V[:, np.argsort(w)[::-1]][:, K:]               # noise subspace
    m = np.arange(M, dtype=np.float64)[:, None]
    phi = -2.0 * np.pi * m * d_over_lambda * np.sin(np.deg2rad(scan_deg))[None, :]
    A_scan = np.exp(1j * phi)                           # SAME SIGN as renderer
    proj = En.conj().T @ A_scan                         # [M-K, G]
    return 1.0 / (np.sum(np.abs(proj) ** 2, axis=0) + 1e-12)

man = SceneManifest.load("data/scenes/verify/tiny/train.npy")
rend = SceneRenderer(man.meta)

# Pick 3 scenes with errors=none, high SNR (if any; otherwise any 3 scenes)
scan = np.linspace(-90.0, 90.0, 1801)                   # 0.1° grid, broadside-0°
for idx in [0, 1, 2]:
    scene = man[int(idx)]
    res = rend.render(scene)
    K = int(scene.n_sources)
    true = np.sort(scene.angles_deg[:K].astype(np.float64))
    # Local-max peak picker (NOT argsort top-K)
    from scipy.signal import find_peaks
    sp = music_self_consistent(res.covariance, K, man.meta.M,
                               man.meta.element_spacing_lambda, scan)
    pk_idx, _ = find_peaks(sp, distance=5)              # min 0.5° apart
    top = pk_idx[np.argsort(sp[pk_idx])[-K:]]
    est = np.sort(scan[top])
    print(f"scene {idx}  SNR={scene.snr_db:+.1f} dB  K={K}")
    print(f"   true = {true.tolist()}")
    print(f"   est  = {est.tolist()}")
    print(f"   |err|= {np.abs(true - est).tolist()}")