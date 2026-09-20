#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sequential retrain of EVERY learned model on the corrected renderer
# (coherent multipath, band-limited sources — 2026-09-06) for the Sensors
# major revision.  Stages run one after another on the single GPU because
# eigh-head models (SubspaceNet, DA-MUSIC) are starved by any concurrent GPU job.
#
#   1. ReconUNet            configs/train/reconunet_paper.yaml
#   2. DA-MUSIC K=1..4      configs/train/damusic_paper_k{1,2,3,4}.yaml
#   3. SubspaceNet          configs/train/subspacenet_paper.yaml
#   4. SubViT (full cap.)   configs/train/subvit_paper.yaml
#
# Launch detached so it survives terminal close:
#   setsid nohup bash scripts/run_revision_retrain_sequence.sh > /dev/null 2>&1 &
# Progress:  tail -f experiments/runs/revision_retrain_20260906/status.log
# Per-stage logs: experiments/runs/revision_retrain_20260906/<stage>.log
# A failing stage is logged (rc != 0) and the sequence CONTINUES with the next.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
PY=DOA_env/bin/python
RUN=experiments/runs/revision_retrain_20260906
mkdir -p "$RUN"
STATUS="$RUN/status.log"
echo $$ > "$RUN/orchestrator.pid"

log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

stage() {
  local name=$1; shift
  log "START $name  ($*)"
  local t0=$SECONDS
  "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  local best
  best=$(grep -oE 'val_rmspe=[0-9.]+°' "$RUN/$name.log" | tail -1)
  log "END $name rc=$rc elapsed=$(( (SECONDS-t0)/60 ))min last_${best:-val_rmspe=n/a}"
}

log "PIPELINE START (pid $$) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
stage reconunet   $PY -m reconunet.cli.train --config configs/train/reconunet_paper.yaml
for K in 1 2 3 4; do
  stage damusic_k$K $PY -m reconunet.cli.train --config configs/train/damusic_paper_k$K.yaml
done
stage subspacenet $PY -m reconunet.cli.train --config configs/train/subspacenet_paper.yaml
stage subvit      $PY -m reconunet.cli.train --config configs/train/subvit_paper.yaml
log "PIPELINE DONE"
