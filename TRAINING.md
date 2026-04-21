# TRAINING — DOA three-way comparison

This runbook walks through the complete workflow on a fresh clone:

1. Push this repo to GitHub (one-time, from the original machine).
2. Clone + install on the RTX 2000 Ada box.
3. Generate the tiny dry-run scenes, train all three models, evaluate.
4. Promote to the full 2 M-sample corpus once the dry-run is green.

---

## 0. A note on the RTX 2000 Ada 16 GB

Solid choice for this workload, a few trade-offs worth knowing.

**What you get.** Ada Lovelace architecture (2023, same generation as RTX
40-series), 2,816 CUDA cores, 88 tensor cores, 16 GB GDDR6 at 224 GB/s
bandwidth, ~12 TFLOPS FP32 and ~45-50 TFLOPS TF32 via tensor cores. 70 W
TDP (single-slot, no external PCIe power cable), which is why it's a
workstation card rather than a gaming SKU.

**How it compares.** Performance-wise it sits between an RTX 4060 and
RTX 4060 Ti 16 GB — roughly 60-70 % of an RTX 4070's throughput. Memory
bandwidth is the biggest deficit (224 GB/s vs 504 GB/s on a 4070), which
matters for memory-bound workloads like large batched covariance ops.

**Fit for this project.** The three models have small parameter counts
(ReconUNet ~2-5 M, SubspaceNet <1 M, SubViT ~15-25 M), and our dataloader
bottleneck is CPU-side scene rendering, not GPU flops. 16 GB VRAM gives
ample headroom:

| Model       | Batch 256 peak VRAM | Verdict                     |
|-------------|---------------------|-----------------------------|
| ReconUNet   | ~3-4 GB             | plenty of slack             |
| SubspaceNet | ~1-2 GB             | trivial                     |
| SubViT      | ~8-10 GB            | comfortable at batch 128    |

Expected wall-clock on this GPU with AMP enabled:

| Dataset size | ReconUNet | SubspaceNet | SubViT  |
|-------------:|----------:|------------:|--------:|
| 10 k (tiny)  | ~3-5 min  | ~2-4 min    | ~6-10 min |
| 200 k (med)  | ~2-3 h    | ~1-2 h      | ~5-6 h  |
| 2 M (full)   | ~15-20 h  | ~8-12 h     | ~30-40 h |

**Recommendation.** Fine for the dry-run and the medium corpus. For the
full 2 M corpus you'd probably want to leave it running overnight per
model. If the full run is time-critical, consider renting a cloud GPU
(A10 or L4) for the SubViT leg since it's ~2× the epoch cost of the
other two.

**One hardware gotcha.** The RTX 2000 Ada has no fan in some OEM
configurations (fully blower or passive). Check that your chassis
airflow actually cools it under a sustained 70 W load — a 24 h training
run will expose any thermal weakness.

---

## 1. Push to a fresh private GitHub repo (one-time)

Run these in `ReconUNet/`. Replace `itamaralmog` with your actual GitHub
handle.

```bash
# (optional) First commit lives locally already — just verify:
git status              # should show "No commits yet" with files staged

# Sanity check: total size under 100 MB
git ls-files --cached | xargs -I{} stat -c '%s {}' {} | \
    awk '{sum+=$1} END {printf "%.1f MB\n", sum/1024/1024}'

# Commit — DO NOT skip hooks
git commit -m "Initial commit — three-way DOA comparison pipeline"

# Create the private repo on GitHub (needs gh auth login beforehand):
gh auth login          # once per machine
gh repo create itamaralmog/doa-nets \
    --private \
    --source=. \
    --remote=origin \
    --push
```

If you don't have `gh` installed, the manual equivalent is:

```bash
# 1. Create the repo via the GitHub web UI (Private, no README/license/gitignore).
# 2. Then:
git remote add origin git@github.com:itamaralmog/doa-nets.git
git branch -M main
git commit -m "Initial commit — three-way DOA comparison pipeline"
git push -u origin main
```

The two third-party modules are git submodules (`.gitmodules`), so GitHub
stores only their pinned commit SHAs. They'll be cloned on the GPU box in
step 2.

---

## 2. Clone + install on the RTX 2000 Ada machine

```bash
git clone --recurse-submodules git@github.com:itamaralmog/doa-nets.git
cd doa-nets

# If you forgot --recurse-submodules:
# git submodule update --init --recursive

# Create a fresh venv (Python >= 3.10 recommended; 3.11 is best).
python3 -m venv .venv
source .venv/bin/activate

# Install the CUDA 12.1 PyTorch wheels that match the RTX 2000 Ada driver.
# Replace cu121 with your CUDA toolkit if different (nvidia-smi to check).
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the package itself + dev tools:
pip install -e ".[dev]"

# Confirm CUDA is live:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA RTX 2000 Ada Generation
```

---

## 3. Tiny dry-run (~15 min end-to-end)

Run all four commands **in the repo root**. They're idempotent — safe to
rerun.

```bash
# 3a. Generate 10 k / 1 k / 1 k scene manifests (takes ~2 seconds):
reconunet-generate --config configs/data/tiny.yaml

# 3b. Point each training config at the tiny manifest temporarily.  The
#     --epochs 5 flag overrides the YAML's 80-epoch setting for a sanity
#     pass.  Note: the configs reference configs/data/shared_manifest.yaml
#     by default — for the dry run, override DATA with an env var or by
#     editing the `data.config` line in each training YAML to point at
#     tiny.yaml.  Simpler: run the convenience script below.

# 3c. Train all three (sequentially — each uses the full GPU):
reconunet-train --config configs/train/reconunet.yaml    --epochs 5
reconunet-train --config configs/train/subspacenet.yaml  --epochs 5
reconunet-train --config configs/train/subvit.yaml       --epochs 5

# 3d. Unified evaluation across all three trained checkpoints:
reconunet-evaluate --config configs/eval/tiny.yaml
```

Expected outputs after 3d:

```
experiments/runs/eval_tiny/
├── results.csv        # model,snr_db,rmse_deg,crlb_deg,n_samples,latency_ms
├── rmse_vs_snr.png    # log-y plot with CRLB overlay
└── config.json        # full cfg snapshot for reproducibility
```

The stdout pivot table should look roughly like:

```
model      reconunet  subspacenet  subvit
snr_db
-10.0         9.123        9.540   8.987
 -5.0         4.201        4.687   4.051
  0.0         1.512        1.703   1.470
  5.0         0.598        0.680   0.571
 10.0         0.239        0.294   0.224
 15.0         0.102        0.149   0.099
 20.0         0.052        0.084   0.049
```

(Exact numbers will differ — the seeded RNG guarantees reproducibility
across runs *on the same machine*, not across GPU models.)

If any model shows NaN RMSE or an RMSE >> 30° at high SNR, something is
mis-wired — stop and open `experiments/runs/<model>/tb/` in TensorBoard
to diagnose before burning overnight cycles on the full corpus.

---

## 4. Full paper-scale run (only after dry-run is green)

```bash
# 4a. Edit configs/data/shared_manifest.yaml — confirm train_size=2_000_000.
#     (Default is 2 M / 50 k / 50 k; leave it alone unless ablating.)

# 4b. Generate the full corpus (~2 minutes, 183 MB disk):
reconunet-generate --config configs/data/shared_manifest.yaml

# 4c. Train each model on the full 80-epoch schedule.  Run them one at a
#     time — parallelising across the single RTX 2000 will thrash the
#     16 GB VRAM ceiling.  Use tmux/screen so disconnects don't kill the
#     job:
tmux new -s train
reconunet-train --config configs/train/reconunet.yaml    # ~15-20 h
# detach (Ctrl-b d), reattach with `tmux attach -t train`

# Then the other two:
reconunet-train --config configs/train/subspacenet.yaml  # ~8-12 h
reconunet-train --config configs/train/subvit.yaml       # ~30-40 h

# 4d. Author a full eval config (copy configs/eval/tiny.yaml, change
#     `manifest:` to data/scenes/test.npy, `output_dir:` to
#     experiments/runs/eval_full/), then:
reconunet-evaluate --config configs/eval/full.yaml
```

---

## 5. Monitoring a live run

```bash
# TensorBoard (in another terminal):
tensorboard --logdir experiments/runs --port 6006

# GPU utilisation:
watch -n 2 nvidia-smi

# Loss/val curves stream live into TensorBoard, and best checkpoints are
# saved to experiments/runs/<model>/checkpoints/best.pt whenever val
# RMSPE improves.  Resume from last.pt by re-running the same command —
# the trainer does NOT currently auto-resume; that's a one-line TODO in
# train_loop().
```

---

## 6. Troubleshooting

| Symptom                               | Probable cause / fix                                                  |
|---------------------------------------|------------------------------------------------------------------------|
| `CUDA out of memory` on SubViT        | Drop `batch_size` to 64 in `configs/train/subvit.yaml`                |
| Training loss is NaN on epoch 1       | AMP instability — set `train.amp: false` in the config                |
| Val RMSPE >> train RMSPE              | Overfitting — increase dropout or decrease `lr`                       |
| `FileNotFoundError: data/scenes/...`  | You forgot step 3a / 4b — run `reconunet-generate` first              |
| Submodule import errors (SubspaceNet) | `git submodule update --init --recursive` after clone                 |
| Scenes regenerate differently after change in `sampling.rng_seed` | **This is intentional**. Changing the seed invalidates every checkpoint. Don't change it mid-project. |

---

Generated 2026-04-20 alongside the pipeline reorg described in
`docs/REORG_PLAN.md`.
