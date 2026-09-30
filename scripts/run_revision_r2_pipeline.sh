#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Revision round 2 pipeline, 2026-09-30 (unattended; Sensors sensors-4536109):
#   A. full-scale training of ReconUNet-C (covariance-only ReconUNet, L_rec only,
#      configs/train/reconunet_c_paper.yaml); on failure resume once from last.pt
#   B. re-evaluation with ReconUNet-C as an extra method next to ReconUNet (DA-MUSIC
#      v2 ensemble): paper test split, scenario sweep, Table II mild / III harsh (all
#      back ends, all SNRs), bootstrap CIs, snapshot + separation sweeps, cost, MUSIC
#      verification, MUSIC-spectrum figure
#   C. source-bandwidth sweep (decoupled = replica delays fixed at bw 0.05; coupled)
#   D. export CSVs to docs/revision_r2, append "R2 (2026-09-30)" sections to
#      docs/revision_facts.md, final status (docs/revision_r2/FINAL_STATUS.md)
# Every stage after training continues on failure; ReconUNet-C-dependent stages are
# skipped (logged) if training produced no best.pt.  All outputs go to NEW directories.
# Status: experiments/runs/revision_r2_20260930/status.log (START/END rc lines).
# Launch:  setsid nohup bash scripts/run_revision_r2_pipeline.sh > /dev/null 2>&1 &
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
PY=DOA_env/bin/python
# R2_* overrides exist only for the pre-launch dry run on the smoke checkpoint.
D=${R2_DATE:-20260930}
RUN=experiments/runs/revision_r2_$D
E=experiments/runs/eval_r2_$D
SW=experiments/runs/sweeps_r2_$D
TRAIN_CFG=${R2_TRAIN_CFG:-configs/train/reconunet_c_paper.yaml}
RUNDIR=${R2_RUNDIR:-experiments/runs/reconunet_c_paper}
CKD=$RUNDIR/checkpoints
RC=$CKD/best.pt
ENS=experiments/runs/damusic_paper/ensemble_v2
FIG=${R2_FIG:-docs/figs_revision/r2}
DOCS=${R2_DOCS:-docs/revision_r2}
BW_EXTRA=(${R2_BW_EXTRA:-})
mkdir -p "$RUN" "$E" "$SW" "$FIG" "$DOCS"
STATUS=$RUN/status.log
echo $$ > "$RUN/pipeline.pid"
export MPLBACKEND=Agg
log()  { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
stage() {   # stage NAME CMD...
  local name=$1; shift
  log "START $name  ($*)"
  local t0=$SECONDS
  "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc elapsed=$(( (SECONDS-t0)/60 ))min"
  return $rc
}
skip() { log "SKIP $1 (no ReconUNet-C checkpoint at $RC)"; }
FACTS="$PY scripts/analysis/revision_facts.py"
H="R2 (2026-09-30)"
log "PIPELINE START pid=$$ gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) git=$(git rev-parse --short HEAD)"

# ---------------- A. ReconUNet-C training ------------------------------------
[ -e "$CKD/best.pt" ] && { log "REFUSING to overwrite existing $CKD/best.pt"; exit 2; }
T0=$SECONDS
stage train_reconunet_c $PY -m reconunet.cli.train --config $TRAIN_CFG
if [ $? -ne 0 ]; then
  if [ -f "$CKD/last.pt" ]; then
    log "RETRY train_reconunet_c: resume from last.pt"
    stage train_reconunet_c_resume $PY -m reconunet.cli.train --config $TRAIN_CFG --resume "$CKD/last.pt"
  fi
fi
log "TRAINING total wall-clock $(( (SECONDS-T0)/60 ))min"
stage train_summary $PY scripts/analysis/r2_report.py train-summary --run-dir $RUNDIR --status-log "$STATUS" -o "$RUN/reconunet_c_training.csv"
if [ -f "$RC" ]; then HAVE_RC=1; RCARG=(--reconunet-c "$RC"); else HAVE_RC=0; RCARG=(); log "WARNING: no $RC — Part B skipped; Part C runs without ReconUNet-C"; fi

# ---------------- B. re-evaluation with ReconUNet-C --------------------------
if [ $HAVE_RC = 1 ]; then
  stage eval_paper_testset_r2  $PY scripts/analysis/compare_paper_testset.py -o "$E/paper_testset_r2"  --damusic-dir "$ENS" "${RCARG[@]}" --dump-errors "$E/paper_testset_r2/errors.npz"
  stage eval_scenario_sweep_r2 $PY scripts/analysis/scenario_sweep.py        -o "$E/scenario_sweep_r2" --damusic-dir "$ENS" "${RCARG[@]}" --dump-errors "$E/scenario_sweep_r2/errors.npz"
  stage eval_table2_mild_r2    $PY scripts/analysis/reproduce_table2.py      -o "$E/table2_mild_r2"    --damusic-dir "$ENS" "${RCARG[@]}" --dump-errors "$E/table2_mild_r2/errors.npz"
  stage eval_table2_harsh_r2   $PY scripts/analysis/reproduce_table2.py      -o "$E/table2_harsh_r2"   --damusic-dir "$ENS" "${RCARG[@]}" --dump-errors "$E/table2_harsh_r2/errors.npz" --scenarios-root data/scenes/scenarios_harsh
  stage bootstrap_ci_r2 bash -c "
    rc=0; for d in paper_testset_r2 scenario_sweep_r2 table2_mild_r2 table2_harsh_r2; do
      if [ -f $E/\$d/errors.npz ]; then $PY scripts/analysis/bootstrap_ci.py $E/\$d/errors.npz -o $E/\$d/\${d}_ci.csv --table \$d || rc=1; else echo \"missing \$d\"; rc=1; fi
    done; exit \$rc"
  stage snapshot_sweep_r2     $PY scripts/analysis/snapshot_sweep.py   -o "$SW" "${RCARG[@]}"
  stage separation_sweep_r2   $PY scripts/analysis/separation_sweep.py -o "$SW" "${RCARG[@]}" --damusic-dir "$ENS"
  stage cost_r2               $PY scripts/analysis/cost_r2.py -o "$SW" "${RCARG[@]}" --status-log "$STATUS"
  stage music_verification_r2 bash -c "$PY scripts/analysis/music_verification.py -o $SW --reconunet-c $RC > $SW/music_verification.md"
  stage figures_r2            $PY scripts/analysis/revision_figures.py --reconunet-c "$RC" --out-dir "$FIG"
else
  for s in eval_paper_testset_r2 eval_scenario_sweep_r2 eval_table2_mild_r2 eval_table2_harsh_r2 bootstrap_ci_r2 snapshot_sweep_r2 separation_sweep_r2 cost_r2 music_verification_r2 figures_r2; do skip $s; done
fi

# ---------------- C. source-bandwidth sweep ----------------------------------
stage bw_sweep_decoupled $PY scripts/analysis/bandwidth_sweep.py --mode decoupled -o "$SW" --damusic-dir "$ENS" "${RCARG[@]}" "${BW_EXTRA[@]}"
stage bw_sweep_coupled   $PY scripts/analysis/bandwidth_sweep.py --mode coupled   -o "$SW" --damusic-dir "$ENS" "${RCARG[@]}" "${BW_EXTRA[@]}"
stage bw_plot            $PY scripts/analysis/bandwidth_sweep.py --mode plot      -o "$SW" --fig-dir "$FIG"

# ---------------- D. export + facts ------------------------------------------
stage export_r2 $PY scripts/analysis/r2_report.py export --eval-dir "$E" --sweeps-dir "$SW" --run-dir "$RUN" --dest "$DOCS"
facts_r2() {
  local fail=0
  af() { $FACTS append-csv "$@" || { echo "FAILED append-csv $*"; fail=$((fail+1)); }; }
  am() { $FACTS append-md "$@"  || { echo "FAILED append-md $*";  fail=$((fail+1)); }; }
  [ -f "$RUN/reconunet_c_training.csv" ] && af --title "$H — ReconUNet-C training (full scale)" --csv "$RUN/reconunet_c_training.csv" \
     --note "ReconUNet-C = CovarianceOnlyReconstructionUNet (no eigen heads) trained with L_rec only on the full 2 M / 50 k corpus with the paper schedule (configs/train/reconunet_c_paper.yaml); checkpoint = minimum validation loss. Validation losses of the two rows are NOT comparable (L_rec alone vs the 4-term composite loss); validation RMSPE is (Root-MUSIC on R_hat, true K). Wall-clock from the R2 status log (single RTX 2000 Ada)."
  [ -f "$E/paper_testset_r2/paper_testset_by_K.csv" ] && af --title "$H — Paper test split by K, ReconUNet-C added (median RMSPE _med / pooled RMSE _mean, deg)" --csv "$E/paper_testset_r2/paper_testset_by_K.csv" \
     --note "Same scenes and code path as the R1 table (DA-MUSIC v2 ensemble); ReconUNet-C uses the same Root-MUSIC back end with the true per-scene K."
  [ -f "$E/paper_testset_r2/paper_testset_r2_ci.csv" ] && af --title "$H — Paper test split, bootstrap 95 % CIs" --csv "$E/paper_testset_r2/paper_testset_r2_ci.csv" \
     --cols scenario,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi
  [ -f "$E/scenario_sweep_r2/scenario_sweep.csv" ] && af --title "$H — Scenario sweep (pooled RMSE vs SNR, deg), ReconUNet-C added" --csv "$E/scenario_sweep_r2/scenario_sweep.csv"
  [ -f "$E/scenario_sweep_r2/scenario_sweep_r2_ci.csv" ] && af --title "$H — Scenario sweep bootstrap 95 % CIs at 0 dB (all SNRs in the CSV)" --csv "$E/scenario_sweep_r2/scenario_sweep_r2_ci.csv" --filter snr_db=0 \
     --cols scenario,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi
  for p in mild harsh; do
    case $p in mild) tn="II (mild)";; *) tn="III (harsh)";; esac
    [ -f "$E/table2_${p}_r2/table2_${p}_r2_ci.csv" ] && af --title "$H — Table $tn at 0 dB with ReconUNet-C back ends, bootstrap 95 % CIs" \
       --csv "$E/table2_${p}_r2/table2_${p}_r2_ci.csv" --filter snr_db=0 --cols scenario,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi
    [ -f "$DOCS/table2_${p}_reconunet_backends_by_snr.csv" ] && af --title "$H — Table $tn: ReconUNet vs ReconUNet-C, every back end, every SNR (pooled RMSE, deg)" \
       --csv "$DOCS/table2_${p}_reconunet_backends_by_snr.csv"
  done
  [ -f "$SW/snapshot_sweep.csv" ]   && af --title "$H — Snapshot sweep, Moderate at 0 dB, ReconUNet-C added" --csv "$SW/snapshot_sweep.csv"
  [ -f "$SW/separation_sweep.csv" ] && af --title "$H — Separation sweep, K=2 at 0 and −5 dB (RMSE, median, resolution probability), ReconUNet-C added" --csv "$SW/separation_sweep.csv"
  [ -f "$SW/cost_summary_r2.csv" ]  && af --title "$H — Cost: ReconUNet-C vs ReconUNet (same measurement as item 8)" --csv "$SW/cost_summary_r2.csv"
  [ -f "$SW/cost_latency_r2.csv" ]  && af --title "$H — Cost: latency detail" --csv "$SW/cost_latency_r2.csv"
  [ -s "$SW/music_verification.md" ] && am --title "$H — MUSIC verification incl. ReconUNet-C" --md "$SW/music_verification.md"
  [ -f "$SW/bandwidth_sweep.csv" ] && af --title "$H — Source-bandwidth sweep, replica delays fixed at the training bandwidth 0.05 (primary)" --csv "$SW/bandwidth_sweep.csv" \
     --cols scenario,bw_frac,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi \
     --note "0 dB, mild imperfections, T = 512, 1000 scenes per scenario; identical scenes (seeds, angles, replica AoAs/gains/delays, imperfections, noise) at every bandwidth, only the source spectrum changes. moderate = paper Moderate manifest (2 direct + 1 replica), crowded = 4 + 3, moderate3 = seeded 3 + 1 draw. Models trained at bw 0.05 only."
  [ -f "$SW/bandwidth_sweep_coupled.csv" ] && af --title "$H — Source-bandwidth sweep, coupled (replica delays ∝ 1/bw as in the renderer; secondary)" --csv "$SW/bandwidth_sweep_coupled.csv" \
     --cols scenario,bw_frac,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi \
     --note "White sources use the renderer's legacy 0.05 delay range in both modes, so the white rows are identical."
  [ -f "$SW/bandwidth_autocorr.csv" ] && af --title "$H — Measured source autocorrelation |r(l)| (l = 0..7) and direct/replica |γ| per bandwidth (decoupled delays)" --csv "$SW/bandwidth_autocorr.csv"
  [ -f "$SW/bandwidth_autocorr_coupled.csv" ] && af --title "$H — Measured |r(l)| and |γ| per bandwidth (coupled delays)" --csv "$SW/bandwidth_autocorr_coupled.csv"
  return $fail
}
stage facts_r2 facts_r2
$PY scripts/results_status_r2.py --final >> "$RUN/results_status.log" 2>&1
log "PIPELINE DONE ($(grep -c ' END .* rc=0 ' "$STATUS") stages rc=0, $(grep ' END ' "$STATUS" | grep -vc ' rc=0 ') failed)"
$PY scripts/results_status_r2.py --final >> "$RUN/results_status.log" 2>&1
