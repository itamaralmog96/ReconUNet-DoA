#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Revision pipeline 2026-09-20 (unattended):
#   A. ReconUNet ablation, reduced protocol (9 variants, two at a time on the GPU,
#      failed variants retried once sequentially) -> ablation_eval -> facts
#   B. DA-MUSIC K=2,3,4 exact continuation to <=600 epochs -> 4 eval jobs with the
#      v2 ensemble (outputs *_v2 next to the 2026-09-10 originals) -> facts
#   C. Cheap evaluation-only extras (each continues on failure): bootstrap CIs,
#      snapshot sweep, separation sweep, FBSS baseline, figures, MUSIC check
# Status: experiments/runs/revision2_20260920/status.log (START/END rc lines);
# per-stage logs next to it; ablation and DA-MUSIC also keep their own status logs.
# Launch:  setsid nohup bash scripts/run_revision2_pipeline.sh > /dev/null 2>&1 &
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
PY=DOA_env/bin/python
RUN=experiments/runs/revision2_20260920
ABL=experiments/runs/ablation_20260920
DMP=experiments/runs/damusic_paper
E=experiments/runs/eval_coherent_20260910
SW=experiments/runs/sweeps_20260920
mkdir -p "$RUN" "$ABL" "$SW"
STATUS=$RUN/status.log
echo $$ > "$RUN/pipeline.pid"
export MPLBACKEND=Agg
log()  { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
stage() {   # stage NAME [EXTRA_STATUS_LOG|-] CMD...
  local name=$1 extra=$2; shift 2
  log "START $name  ($*)"; [ "$extra" != "-" ] && echo "$(date -u '+%F %T') START $name" >> "$extra"
  local t0=$SECONDS
  "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc elapsed=$(( (SECONDS-t0)/60 ))min"
  [ "$extra" != "-" ] && echo "$(date -u '+%F %T') END $name rc=$rc elapsed=$(( (SECONDS-t0)/60 ))min" >> "$extra"
  return $rc
}
FACTS="$PY scripts/analysis/revision_facts.py"
log "PIPELINE START pid=$$ gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# ---------------- A. ablation ------------------------------------------------
VARIANTS=(01_full 02_rec_only 03_rec_proj 04_no_dom 05_no_eig 06_single_lag 07_relu 08_no_evd_heads 09_fixed_imperf)
train_variant() { stage "abl_train_$1" "$ABL/status.log" $PY -m reconunet.cli.train --config "configs/train/ablation_$1.yaml"; }
for pair in "01_full 02_rec_only" "03_rec_proj 04_no_dom" "05_no_eig 06_single_lag" "07_relu 08_no_evd_heads" "09_fixed_imperf"; do
  for v in $pair; do train_variant "$v" & done
  wait
done
for v in "${VARIANTS[@]}"; do            # retry once, alone, anything that did not produce a checkpoint
  [ -f "$ABL/$v/checkpoints/best.pt" ] || { log "RETRY $v (no best.pt)"; train_variant "$v"; }
done
stage abl_eval "$ABL/status.log" $PY scripts/analysis/ablation_eval.py --ablation-dir "$ABL"
stage facts_ablation - bash -c "
  $FACTS append-csv --title 'Ablation — reduced protocol (10 % seeded subset, 40 epochs), Root-MUSIC back end' --csv $ABL/ablation_results.csv \
     --note 'Columns: pooled RMSE (paper eq. 31), median per-scene RMSPE, relative covariance error ||R_hat-R*||_F/||R*||_F, leading-K projector distance ||P_hat-P*||_F, relative eigengap error |gap_hat-gap*|/gap* (all from eigh(R_hat)); eval sets: paper test split (3000 scenes per K), Moderate and Crowded at 0 dB (1000 scenes), and for 01/09 the paper test split rendered with the single fixed imperfection realisation used to train 09.' &&
  $FACTS append-csv --title 'Route comparison — covariance route (eigh of R_hat) vs subspace route (EVD-head eigenvectors)' --csv $ABL/route_comparison.csv"

# ---------------- B. DA-MUSIC continuation -----------------------------------
for K in 2 3 4; do
  mkdir -p "$DMP/k${K}_v2/checkpoints"
  cp -n "$DMP/k$K/checkpoints/history.json" "$DMP/k${K}_v2/checkpoints/" 2>/dev/null
  cp -n "$DMP/k$K/checkpoints/best.pt"      "$DMP/k${K}_v2/checkpoints/" 2>/dev/null   # v2 best starts as v1 best
  stage "damusic_k${K}_v2" "$DMP/status_v2.log" $PY -m reconunet.cli.train --config "configs/train/damusic_paper_k${K}_v2.yaml"
done
ENS=$DMP/ensemble_v2; mkdir -p "$ENS"
ln -sfn ../k1 "$ENS/k1"
for K in 2 3 4; do
  if [ -f "$DMP/k${K}_v2/checkpoints/best.pt" ]; then ln -sfn "../k${K}_v2" "$ENS/k$K"; else ln -sfn "../k$K" "$ENS/k$K"; fi
done
cat > "$ENS/README.md" <<TXT
Symlink ensemble used for the *_v2 evaluations: k1 -> ../k1 (converged in v1), k2..k4 -> ../k<K>_v2 (600-epoch continuations).
TXT
stage eval_paper_testset_v2  - $PY scripts/analysis/compare_paper_testset.py -o "$E/paper_testset_v2"  --damusic-dir "$ENS" --dump-errors "$E/paper_testset_v2/errors.npz"
stage eval_scenario_sweep_v2 - $PY scripts/analysis/scenario_sweep.py        -o "$E/scenario_sweep_v2" --damusic-dir "$ENS" --dump-errors "$E/scenario_sweep_v2/errors.npz"
stage eval_table2_mild_v2    - $PY scripts/analysis/reproduce_table2.py      -o "$E/table2_mild_v2"    --damusic-dir "$ENS" --dump-errors "$E/table2_mild_v2/errors.npz"
stage eval_table2_harsh_v2   - $PY scripts/analysis/reproduce_table2.py      -o "$E/table2_harsh_v2"   --damusic-dir "$ENS" --dump-errors "$E/table2_harsh_v2/errors.npz" --scenarios-root data/scenes/scenarios_harsh
stage facts_damusic - bash -c "
  $FACTS append-history --title 'DA-MUSIC K=2,3,4 continuation to 600 epochs' --runs k2_v2 k3_v2 k4_v2 &&
  $FACTS append-csv --title 'Paper test split with the DA-MUSIC v2 ensemble (pooled RMSE / median per K, all methods)' --csv $E/paper_testset_v2/paper_testset_by_K.csv &&
  $FACTS append-csv --title 'Table II (mild) at 0 dB with the DA-MUSIC v2 ensemble' --csv $E/table2_mild_v2/table2_full.csv --filter snr_db=0.0 &&
  $FACTS append-csv --title 'Table II (harsh) at 0 dB with the DA-MUSIC v2 ensemble' --csv $E/table2_harsh_v2/table2_full.csv --filter snr_db=0.0"

# ---------------- C. extras (evaluation only; each continues on failure) -----
stage bootstrap_ci - bash -c "
  for d in paper_testset_v2 scenario_sweep_v2 table2_mild_v2 table2_harsh_v2; do
    [ -f $E/\$d/errors.npz ] && $PY scripts/analysis/bootstrap_ci.py $E/\$d/errors.npz -o $E/\$d/\${d}_ci.csv --table \$d || echo \"skip \$d\"
  done; true"
stage snapshot_sweep     - $PY scripts/analysis/snapshot_sweep.py   -o "$SW"
stage separation_sweep   - $PY scripts/analysis/separation_sweep.py -o "$SW"
stage fbss_baseline      - $PY scripts/analysis/fbss_baseline.py    -o "$SW"
stage revision_figures   - $PY scripts/analysis/revision_figures.py
stage music_verification - bash -c "$PY scripts/analysis/music_verification.py -o $SW > $SW/music_verification.md && $FACTS append-md --title 'MUSIC verification' --md $SW/music_verification.md"
stage facts_extras - bash -c "
  [ -f $SW/fbss_baseline.csv ]     && $FACTS append-csv --title 'FBSS Root-MUSIC baseline at 0 dB (Moderate, Crowded; mild)' --csv $SW/fbss_baseline.csv --filter snr_db=0.0;
  [ -f $SW/snapshot_sweep.csv ]    && $FACTS append-csv --title 'Snapshot sweep, Moderate at 0 dB' --csv $SW/snapshot_sweep.csv;
  [ -f $SW/separation_sweep.csv ]  && $FACTS append-csv --title 'Separation sweep, K=2 at 0 and −5 dB (RMSE, median, resolution probability)' --csv $SW/separation_sweep.csv;
  [ -f $E/table2_mild_v2/table2_mild_v2_ci.csv ] && $FACTS append-csv --title 'Bootstrap 95 % CIs, Table II mild at 0 dB' --csv $E/table2_mild_v2/table2_mild_v2_ci.csv --filter snr_db=0.0 --cols scenario,method,n_scenes,rmse_deg,rmse_ci_lo,rmse_ci_hi,median_rmspe_deg,median_ci_lo,median_ci_hi;
  true"
$PY scripts/results_status.py >> "$RUN/results_status.log" 2>&1
log "PIPELINE DONE"
