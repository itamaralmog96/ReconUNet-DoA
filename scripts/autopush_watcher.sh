#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Auto-publish watcher for the 2026-09-20 revision pipeline.
#
# Every INTERVAL seconds (default 1800): rewrite RESULTS_STATUS.md, stage the
# result paths, commit ("auto: results update <UTC>") and push origin main.
# Stops itself once the pipeline process (PIPELINE_PID) has exited, after a
# final cycle committed as "auto: all runs complete".  Own output goes to
# experiments/runs/autopush.log (tracked).
#
# Launch (detached, survives logout):
#   GH_TOKEN="$(gh auth token)" PIPELINE_PID=<pid> setsid nohup \
#       bash scripts/autopush_watcher.sh > /dev/null 2>&1 &
# GH_TOKEN is read from the environment only (never written to disk) so pushes
# do not depend on the desktop keyring being unlocked.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=experiments/runs/autopush.log
INTERVAL=${AUTOPUSH_INTERVAL:-1800}
PY=DOA_env/bin/python
: "${PIPELINE_PID:?set PIPELINE_PID}"
: "${GH_TOKEN:?set GH_TOKEN}"
export GH_TOKEN
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -u '+%FT%TZ') $*" >> "$LOG"; }
echo $$ > experiments/runs/autopush.pid

ADD_PATHS=(RESULTS_STATUS.md docs/revision_facts.md docs/figs_revision
           experiments/runs/ablation_20260920 experiments/runs/sweeps_20260920
           experiments/runs/revision2_20260920 experiments/runs/eval_coherent_20260910
           experiments/runs/autopush.log)

push() {
  local n
  for n in 1 2 3; do
    if git -c credential.helper= \
           -c credential.helper='!f() { echo username=x-access-token; echo "password=${GH_TOKEN}"; }; f' \
           push -q origin main >> "$LOG" 2>&1; then
      log "push ok"; return 0
    fi
    log "push failed (attempt $n)"; sleep 60
  done
  return 1
}

cycle() {   # $1 = commit message
  $PY scripts/results_status.py >> "$LOG" 2>&1 || log "results_status.py failed"
  local p
  for p in "${ADD_PATHS[@]}" experiments/runs/damusic_paper/*_v2* experiments/runs/*/status*.log; do
    [ -e "$p" ] && git add "$p" >> "$LOG" 2>&1
  done
  if git diff --cached --quiet; then
    log "nothing new to commit"
  else
    git commit -q -m "$1" >> "$LOG" 2>&1 && log "committed: $1 ($(git rev-parse --short HEAD))"
  fi
  push || true
}

log "watcher start pid=$$ pipeline_pid=$PIPELINE_PID interval=${INTERVAL}s"
while :; do
  cycle "auto: results update $(date -u +%FT%TZ)"
  if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
    log "pipeline pid $PIPELINE_PID has exited — final cycle"
    sleep 5
    cycle "auto: all runs complete"
    log "watcher exit"
    exit 0
  fi
  sleep "$INTERVAL"
done
