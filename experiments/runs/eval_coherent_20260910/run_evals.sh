#!/usr/bin/env bash
# Post-retrain evaluation suite on the corrected (coherent-multipath) renderer,
# all four learned models + classical + CRLB.  Sequential on the GPU.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=DOA_env/bin/python
E=experiments/runs/eval_coherent_20260910
S=$E/status.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$S"; }
run(){ local name=$1; shift; log "START $name"; "$@" > "$E/$name.log" 2>&1; log "END $name rc=$?"; }
run paper_testset  $PY scripts/analysis/compare_paper_testset.py -o $E/paper_testset
run scenario_sweep $PY scripts/analysis/scenario_sweep.py -o $E/scenario_sweep
run table2_mild    $PY scripts/analysis/reproduce_table2.py -o $E/table2_mild
run table2_harsh   $PY scripts/analysis/reproduce_table2.py -o $E/table2_harsh --scenarios-root data/scenes/scenarios_harsh
log "EVALS DONE"
