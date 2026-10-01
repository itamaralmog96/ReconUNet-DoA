#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Revision R2b pipeline, 2026-10-01 (unattended; Sensors sensors-4536109):
#   B. train ReconUNet-CB from scratch = ReconUNet-C architecture + L_rec, on the
#      randomised-source-bandwidth corpus (configs/train/reconunet_cb_paper.yaml);
#      on failure resume once from last.pt
#   C. evaluation with ReconUNet-CB next to ReconUNet and ReconUNet-C (DA-MUSIC v2):
#      bandwidth sweep (0 dB decoupled + coupled, all methods; −5 / +10 dB decoupled
#      for Root-MUSIC / ReconUNet-C / ReconUNet-CB), paper test split, scenario sweep,
#      Table II mild / III harsh, matched-wideband test split, bootstrap CIs,
#      snapshot + separation sweeps, cost
#   D. export to docs/revision_r2b, "R2b (2026-10-01)" facts sections, final status
# Every stage after training continues on failure.  All outputs in NEW directories.
# Status: experiments/runs/revision_r2b_20261001/status.log (START/END rc lines).
# Launch:  setsid nohup bash scripts/run_revision_r2b_pipeline.sh > /dev/null 2>&1 &
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
PY=DOA_env/bin/python
# R2B_* overrides exist only for the pre-launch dry run on the smoke checkpoint.
D=${R2B_DATE:-20261001}
RUN=experiments/runs/revision_r2b_$D
E=experiments/runs/eval_r2b_$D
SW=experiments/runs/sweeps_r2b_$D
TRAIN_CFG=${R2B_TRAIN_CFG:-configs/train/reconunet_cb_paper.yaml}
CKD=${R2B_CKD:-experiments/runs/reconunet_cb_paper/checkpoints}
CBK=$CKD/best.pt
RCK=experiments/runs/reconunet_c_paper/checkpoints/best.pt
R2STATUS=experiments/runs/revision_r2_20260930/status.log
ENS=experiments/runs/damusic_paper/ensemble_v2
BWCFG=configs/data/paper_corpus_bwrand.yaml
FIG=${R2B_FIG:-docs/figs_revision/r2b}
DOCS=${R2B_DOCS:-docs/revision_r2b}
BW_EXTRA=(${R2B_BW_EXTRA:-})
mkdir -p "$RUN" "$E" "$SW" "$FIG" "$DOCS"
STATUS=$RUN/status.log
echo $$ > "$RUN/pipeline.pid"
export MPLBACKEND=Agg R2B_DATE=$D
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
FACTS="$PY scripts/analysis/revision_facts.py"
H="R2b (2026-10-01)"
log "PIPELINE START pid=$$ gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) git=$(git rev-parse --short HEAD)"

# ---------------- B. ReconUNet-CB training (from scratch) --------------------
[ -e "$CBK" ] && { log "REFUSING to overwrite existing $CBK"; exit 2; }
[ -f "$RCK" ] || { log "missing ReconUNet-C checkpoint $RCK"; exit 2; }
T0=$SECONDS
stage train_reconunet_cb $PY -m reconunet.cli.train --config "$TRAIN_CFG"
if [ $? -ne 0 ] && [ -f "$CKD/last.pt" ]; then
  log "RETRY train_reconunet_cb: resume from last.pt"
  stage train_reconunet_cb_resume $PY -m reconunet.cli.train --config "$TRAIN_CFG" --resume "$CKD/last.pt"
fi
log "TRAINING total wall-clock $(( (SECONDS-T0)/60 ))min"
stage train_summary $PY scripts/analysis/r2b_report.py train-summary --status-log "$STATUS" -o "$RUN/reconunet_cb_training.csv"
if [ -f "$CBK" ]; then CB=(--reconunet-cb "$CBK"); else CB=(); log "WARNING: no $CBK — evaluating without ReconUNet-CB"; fi
RC=(--reconunet-c "$RCK")

# ---------------- C1. bandwidth sweep (primary result first) ------------------
stage bw_sweep_decoupled $PY scripts/analysis/bandwidth_sweep.py --mode decoupled -o "$SW" --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" "${BW_EXTRA[@]}"
stage bw_sweep_coupled   $PY scripts/analysis/bandwidth_sweep.py --mode coupled   -o "$SW" --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" "${BW_EXTRA[@]}"
stage bw_sweep_m5dB  $PY scripts/analysis/bandwidth_sweep.py --mode decoupled --snr -5 --tag _m5dB  -o "$SW" "${RC[@]}" "${CB[@]}" "${BW_EXTRA[@]}" --methods Root-MUSIC ReconUNet-C ReconUNet-CB
stage bw_sweep_p10dB $PY scripts/analysis/bandwidth_sweep.py --mode decoupled --snr 10 --tag _p10dB -o "$SW" "${RC[@]}" "${CB[@]}" "${BW_EXTRA[@]}" --methods Root-MUSIC ReconUNet-C ReconUNet-CB
stage bw_plot $PY scripts/analysis/bandwidth_sweep.py --mode plot -o "$SW" --fig-dir "$FIG" --fig-name bandwidth_sweep_r2b

# ---------------- C2-C7. remaining evaluations -------------------------------
stage eval_paper_testset_r2b  $PY scripts/analysis/compare_paper_testset.py -o "$E/paper_testset_r2b"  --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" --dump-errors "$E/paper_testset_r2b/errors.npz"
stage eval_scenario_sweep_r2b $PY scripts/analysis/scenario_sweep.py        -o "$E/scenario_sweep_r2b" --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" --dump-errors "$E/scenario_sweep_r2b/errors.npz"
stage eval_table2_mild_r2b    $PY scripts/analysis/reproduce_table2.py      -o "$E/table2_mild_r2b"    --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" --dump-errors "$E/table2_mild_r2b/errors.npz"
stage eval_table2_harsh_r2b   $PY scripts/analysis/reproduce_table2.py      -o "$E/table2_harsh_r2b"   --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" --dump-errors "$E/table2_harsh_r2b/errors.npz" --scenarios-root data/scenes/scenarios_harsh
stage eval_paper_testset_bwrand_r2b $PY scripts/analysis/compare_paper_testset.py -o "$E/paper_testset_bwrand_r2b" --damusic-dir "$ENS" "${RC[@]}" "${CB[@]}" \
      --render-meta-from "$BWCFG" --dump-errors "$E/paper_testset_bwrand_r2b/errors.npz"
stage bootstrap_ci_r2b bash -c "
  rc=0; for d in paper_testset_r2b scenario_sweep_r2b table2_mild_r2b table2_harsh_r2b paper_testset_bwrand_r2b; do
    if [ -f $E/\$d/errors.npz ]; then $PY scripts/analysis/bootstrap_ci.py $E/\$d/errors.npz -o $E/\$d/\${d}_ci.csv --table \$d || rc=1; else echo \"missing \$d\"; rc=1; fi
  done; exit \$rc"
stage snapshot_sweep_r2b   $PY scripts/analysis/snapshot_sweep.py   -o "$SW" "${RC[@]}" "${CB[@]}"
stage separation_sweep_r2b $PY scripts/analysis/separation_sweep.py -o "$SW" "${RC[@]}" "${CB[@]}" --damusic-dir "$ENS"
if [ -f "$CBK" ]; then
  stage cost_r2b $PY scripts/analysis/cost_r2.py -o "$SW" "${RC[@]}" --status-log "$R2STATUS" --reconunet-cb "$CBK" --status-log-cb "$STATUS"
else
  log "SKIP cost_r2b (no $CBK)"
fi

# ---------------- D. export + facts ------------------------------------------
stage export_r2b $PY scripts/analysis/r2b_report.py export --eval-dir "$E" --sweeps-dir "$SW" --run-dir "$RUN" --dest "$DOCS"
CICOLS=scenario,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi
BWCOLS=scenario,bw_frac,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi
facts_r2b() {
  local fail=0
  af() { $FACTS append-csv "$@" || { echo "FAILED append-csv $*"; fail=$((fail+1)); }; }
  am() { $FACTS append-md "$@"  || { echo "FAILED append-md $*";  fail=$((fail+1)); }; }
  [ -s "$DOCS/HEADLINE.md" ] && am --title "$H — Headline (ReconUNet-CB, randomised source bandwidth)" --md "$DOCS/HEADLINE.md"
  [ -f "$RUN/reconunet_cb_training.csv" ] && af --title "$H — ReconUNet-CB training (randomised-bandwidth corpus, from scratch)" --csv "$RUN/reconunet_cb_training.csv" \
     --note "ReconUNet-CB = CovarianceOnlyReconstructionUNet + L_rec, identical to ReconUNet-C except the training corpus: the paper manifests rendered with a per-scene source bandwidth (white w.p. 0.15, else log-uniform on [0.01, 0.4], drawn from rng [scene.seed, 20261001]) and replica delays fixed to the bw-0.05 distribution (configs/data/paper_corpus_bwrand.yaml). Checkpoint selection on the randomised-bandwidth validation split, so its validation numbers are not comparable with the other rows. The L_rec target A_ideal A_ideal^H is bandwidth-independent (direct paths only, analytic)."
  [ -f "$SW/bandwidth_sweep.csv" ] && af --title "$H — Source-bandwidth sweep, 0 dB, decoupled delays (primary), ReconUNet-CB added" --csv "$SW/bandwidth_sweep.csv" --cols "$BWCOLS"
  [ -f "$SW/bandwidth_sweep_coupled.csv" ] && af --title "$H — Source-bandwidth sweep, 0 dB, coupled delays" --csv "$SW/bandwidth_sweep_coupled.csv" --cols "$BWCOLS"
  [ -f "$SW/bandwidth_sweep_m5dB.csv" ] && af --title "$H — Source-bandwidth sweep, −5 dB, decoupled" --csv "$SW/bandwidth_sweep_m5dB.csv" --cols "$BWCOLS"
  [ -f "$SW/bandwidth_sweep_p10dB.csv" ] && af --title "$H — Source-bandwidth sweep, +10 dB, decoupled" --csv "$SW/bandwidth_sweep_p10dB.csv" --cols "$BWCOLS"
  [ -f "$SW/bandwidth_autocorr.csv" ] && af --title "$H — Measured |r(l)| and |γ| per bandwidth (decoupled, 0 dB)" --csv "$SW/bandwidth_autocorr.csv"
  [ -f "$E/paper_testset_r2b/paper_testset_by_K.csv" ] && af --title "$H — Paper test split by K (median _med / pooled RMSE _mean, deg), ReconUNet-CB added" --csv "$E/paper_testset_r2b/paper_testset_by_K.csv"
  [ -f "$E/paper_testset_r2b/paper_testset_r2b_ci.csv" ] && af --title "$H — Paper test split, bootstrap 95 % CIs" --csv "$E/paper_testset_r2b/paper_testset_r2b_ci.csv" --cols "$CICOLS"
  [ -f "$E/paper_testset_bwrand_r2b/paper_testset_by_K.csv" ] && af --title "$H — Matched wideband: paper test split at per-scene random bandwidth, by K" --csv "$E/paper_testset_bwrand_r2b/paper_testset_by_K.csv"
  [ -f "$E/paper_testset_bwrand_r2b/paper_testset_bwrand_r2b_ci.csv" ] && af --title "$H — Matched wideband test split, bootstrap 95 % CIs" --csv "$E/paper_testset_bwrand_r2b/paper_testset_bwrand_r2b_ci.csv" --cols "$CICOLS"
  [ -f "$E/scenario_sweep_r2b/scenario_sweep.csv" ] && af --title "$H — Scenario sweep (pooled RMSE vs SNR, deg), ReconUNet-CB added" --csv "$E/scenario_sweep_r2b/scenario_sweep.csv"
  [ -f "$E/scenario_sweep_r2b/scenario_sweep_r2b_ci.csv" ] && af --title "$H — Scenario sweep bootstrap 95 % CIs at 0 dB" --csv "$E/scenario_sweep_r2b/scenario_sweep_r2b_ci.csv" --filter snr_db=0 --cols "$CICOLS"
  for p in mild harsh; do
    case $p in mild) tn="II (mild)";; *) tn="III (harsh)";; esac
    [ -f "$E/table2_${p}_r2b/table2_${p}_r2b_ci.csv" ] && af --title "$H — Table $tn at 0 dB, bootstrap 95 % CIs" \
       --csv "$E/table2_${p}_r2b/table2_${p}_r2b_ci.csv" --filter snr_db=0 --cols "$CICOLS"
    [ -f "$DOCS/table2_${p}_reconunet_backends_by_snr.csv" ] && af --title "$H — Table $tn: ReconUNet / ReconUNet-C / ReconUNet-CB, every back end, every SNR (pooled RMSE, deg)" \
       --csv "$DOCS/table2_${p}_reconunet_backends_by_snr.csv"
  done
  [ -f "$SW/snapshot_sweep.csv" ]   && af --title "$H — Snapshot sweep, Moderate at 0 dB, ReconUNet-CB added" --csv "$SW/snapshot_sweep.csv"
  [ -f "$SW/separation_sweep.csv" ] && af --title "$H — Separation sweep (RMSE, median, resolution probability), ReconUNet-CB added" --csv "$SW/separation_sweep.csv"
  [ -f "$SW/cost_summary_r2.csv" ]  && af --title "$H — Cost: ReconUNet / ReconUNet-C / ReconUNet-CB" --csv "$SW/cost_summary_r2.csv"
  return $fail
}
stage facts_r2b facts_r2b
$PY scripts/results_status_r2b.py --final >> "$RUN/results_status.log" 2>&1
log "PIPELINE DONE ($(grep -c ' END .* rc=0 ' "$STATUS") stages rc=0, $(grep ' END ' "$STATUS" | grep -vc ' rc=0 ') failed)"
$PY scripts/results_status_r2b.py --final >> "$RUN/results_status.log" 2>&1
