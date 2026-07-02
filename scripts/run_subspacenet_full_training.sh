#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full SubspaceNet training (2M paper_corpus, variable-K, paper recipe) followed
# by the per-K head-to-head vs ReconUNet on the shared multipath test set.
#
# Usage:
#   scripts/run_subspacenet_full_training.sh            # full run (300-epoch cap, early-stop 25)
#   scripts/run_subspacenet_full_training.sh 50         # cap at 50 epochs
#   nohup scripts/run_subspacenet_full_training.sh > /dev/null 2>&1 &   # detached
#
# Estimated time: ~5-10 h on GPU (early stopping usually ends it sooner).
# NOTE: writes to experiments/runs/subspacenet_paper/, overwriting any prior
# run there (the old crashed run). Edit RUN_DIR below to keep that.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repo root (this script lives in scripts/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_CONFIG="configs/train/subspacenet_paper.yaml"
RUN_DIR="experiments/runs/subspacenet_paper"
CKPT="$RUN_DIR/checkpoints/best.pt"
RECONUNET_CKPT="experiments/runs/reconunet_paper/checkpoints/best.pt"
TEST_MANIFEST="data/scenes/subnet_genk/test.npy"
EPOCHS="${1:-}"   # optional positional arg: cap epochs

mkdir -p "$RUN_DIR/checkpoints"
LOG="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[run] repo:    $REPO_ROOT"
echo "[run] config:  $TRAIN_CONFIG  (2M paper_corpus, variable-K, Adam 1e-4, batch 2048)"
echo "[run] outputs: $RUN_DIR"
echo "[run] log:     $LOG"

EPOCH_ARG=()
if [[ -n "$EPOCHS" ]]; then
  EPOCH_ARG=(--epochs "$EPOCHS")
  echo "[run] epochs override: $EPOCHS"
fi

echo "[run] ===== training (this can take several hours) ====="
# pipefail (set above) makes a trainer failure propagate through `tee`.
reconunet-train --config "$TRAIN_CONFIG" "${EPOCH_ARG[@]}" 2>&1 | tee "$LOG"
echo "[run] ===== training done; best checkpoint: $CKPT ====="

# Per-K head-to-head: SubspaceNet vs ReconUNet vs classical Root-MUSIC/ESPRIT.
if [[ -f "$CKPT" && -f "$RECONUNET_CKPT" && -f "$TEST_MANIFEST" ]]; then
  echo "[run] ===== per-K comparison vs ReconUNet ====="
  python scripts/analysis/eval_subnet_genk.py \
    --checkpoint "$CKPT" \
    --manifest "$TEST_MANIFEST" \
    --reconunet-checkpoint "$RECONUNET_CKPT" \
    --output-dir "$RUN_DIR"
else
  echo "[run] skipping comparison (missing one of: best.pt / reconunet best.pt / test manifest)"
fi
echo "[run] all done. results in $RUN_DIR"
