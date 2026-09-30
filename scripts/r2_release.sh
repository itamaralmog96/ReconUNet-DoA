#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Revision round 2: publish the ReconUNet-C checkpoint as GitHub release
# `revision-r2-checkpoints`, on a new tag of the same name at the current HEAD.
# Run AFTER the auto-push watcher has made its final "auto: all runs complete"
# commit (the R2 launcher chains it after scripts/autopush_watcher.sh).
# Never touches the existing revision-r1-checkpoints release; never force-pushes.
# Needs GH_TOKEN in the environment.  Log: experiments/runs/revision_r2_20260930/release.log
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
RUN=experiments/runs/revision_r2_20260930
LOG=$RUN/release.log
TAG=revision-r2-checkpoints
CK=experiments/runs/reconunet_c_paper/checkpoints
: "${GH_TOKEN:?set GH_TOKEN}"
export GH_TOKEN
log() { echo "$(date -u '+%FT%TZ') $*" | tee -a "$LOG"; }

if [ ! -f "$CK/best.pt" ]; then log "no $CK/best.pt — release skipped"; exit 1; fi
if gh release view "$TAG" > /dev/null 2>&1; then log "release $TAG already exists — nothing to do"; exit 0; fi
STAGE=$RUN/release; mkdir -p "$STAGE"
cp "$CK/best.pt"        "$STAGE/reconunet_c_paper_best.pt"
cp configs/train/reconunet_c_paper.yaml "$STAGE/reconunet_c_paper.yaml"
cp "$CK/history.json"   "$STAGE/reconunet_c_paper_history.json"
SHA=$(git rev-parse HEAD)
EPOCH=$(DOA_env/bin/python -c "import torch;c=torch.load('$CK/best.pt',map_location='cpu',weights_only=False);print(c.get('epoch','?'), round(float(c.get('val_loss', c.get('best_val', float('nan')))),5), round(float(c.get('val_rmspe_deg', float('nan'))),3))")
cat > "$STAGE/notes.md" <<TXT
ReconUNet-C — the covariance-only ReconUNet (\`CovarianceOnlyReconstructionUNet\`, no eigen heads, trained with L_rec only), full-scale paper protocol, Sensors revision round 2 (2026-09-30).

* \`reconunet_c_paper_best.pt\` — best-validation checkpoint (epoch, val loss, val RMSPE°: ${EPOCH}); load with \`scripts/analysis/_revision_common.load_reconunet\` (class read from the checkpoint's config).
* \`reconunet_c_paper.yaml\` — training config (\`configs/train/reconunet_c_paper.yaml\`).
* \`reconunet_c_paper_history.json\` — per-epoch training history.

Results: \`docs/revision_r2/\` and \`docs/revision_facts.md\` (sections "R2 (2026-09-30)") at commit ${SHA}.
The R1 checkpoints remain in release \`revision-r1-checkpoints\`.
TXT
log "creating tag $TAG at $SHA"
git tag -a "$TAG" -m "Revision R2: ReconUNet-C checkpoint and results (2026-09-30)" "$SHA" >> "$LOG" 2>&1 || log "tag exists locally"
for n in 1 2 3; do
  git -c credential.helper= -c credential.helper='!f() { echo username=x-access-token; echo "password=${GH_TOKEN}"; }; f' \
      push -q origin "refs/tags/$TAG" >> "$LOG" 2>&1 && { log "tag pushed"; break; }
  log "tag push failed (attempt $n)"; sleep 60
done
for n in 1 2 3; do
  gh release create "$TAG" --verify-tag --title "Revision R2 checkpoint: ReconUNet-C (2026-09-30)" --notes-file "$STAGE/notes.md" \
     "$STAGE/reconunet_c_paper_best.pt" "$STAGE/reconunet_c_paper.yaml" "$STAGE/reconunet_c_paper_history.json" >> "$LOG" 2>&1 \
     && { log "release created: $(gh release view "$TAG" --json url -q .url 2>/dev/null)"; exit 0; }
  log "release create failed (attempt $n)"; sleep 60
done
exit 1
