Basic scenario (K = 1, no multipath, mild imperfections), 1000 scenes per SNR. MUSIC uses the paper's 1° scan grid on [-60°, 60°]; 'refined' adds the three-point parabolic peak refinement; Root-MUSIC is gridless. A spurious peak is a scene whose selected maximum is more than 1°/3°/5° from the true angle.

| SNR (dB) | estimator | median abs err (°) | pooled RMSE (°) | mean abs err (°) | frac > 1° | frac > 3° | frac > 5° | n |
|---|---|---|---|---|---|---|---|---|
| 20 | MUSIC-grid | 0.269 | 0.371 | 0.305 | 0.0020 | 0.0000 | 0.0000 | 1000 |
| 20 | MUSIC-refined | 0.192 | 5.302 | 0.561 | 0.0060 | 0.0060 | 0.0060 | 1000 |
| 20 | Root-MUSIC | 0.147 | 0.229 | 0.182 | 0.0000 | 0.0000 | 0.0000 | 1000 |
| 0 | MUSIC-grid | 0.270 | 0.380 | 0.310 | 0.0060 | 0.0000 | 0.0000 | 1000 |
| 0 | MUSIC-refined | 0.188 | 3.972 | 0.450 | 0.0060 | 0.0050 | 0.0050 | 1000 |
| 0 | Root-MUSIC | 0.163 | 0.251 | 0.198 | 0.0000 | 0.0000 | 0.0000 | 1000 |

Reading: at 20 dB the refined grid-MUSIC median error is 0.192° against 0.147° for Root-MUSIC, i.e. the estimator itself is fine; the pooled RMSE gap (5.302° vs 0.229°) is driven by the 0.60 % of scenes whose selected peak is spurious (> 3°), which RMSE squares. Without refinement the grid alone contributes a 0.269° median quantisation error.

Data: `experiments/runs/sweeps_20260920/music_verification.csv`.
