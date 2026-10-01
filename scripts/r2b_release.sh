#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Revision R2b: add the ReconUNet-CB checkpoint (+ training config, data config,
# history) as EXTRA assets to the existing GitHub release `revision-r2-checkpoints`
# with `gh release upload` (no --clobber: existing assets are never altered).
# Fallback, only if that upload fails: new release `revision-r2b-checkpoints` on a
# new tag at HEAD.  Run after the auto-push watcher's final commit (the R2b
# launcher chains it).  Needs GH_TOKEN.  Log: experiments/runs/revision_r2b_20261001/release.log
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
RUN=experiments/runs/revision_r2b_20261001
LOG=$RUN/release.log
CK=experiments/runs/reconunet_cb_paper/checkpoints
: "${GH_TOKEN:?set GH_TOKEN}"
export GH_TOKEN
log() { echo "$(date -u '+%FT%TZ') $*" | tee -a "$LOG"; }

if [ ! -f "$CK/best.pt" ]; then log "no $CK/best.pt — release skipped"; exit 1; fi
STAGE=$RUN/release; mkdir -p "$STAGE"
cp "$CK/best.pt"      "$STAGE/reconunet_cb_paper_best.pt"
cp configs/train/reconunet_cb_paper.yaml "$STAGE/reconunet_cb_paper.yaml"
cp configs/data/paper_corpus_bwrand.yaml "$STAGE/paper_corpus_bwrand.yaml"
cp "$CK/history.json" "$STAGE/reconunet_cb_paper_history.json"
FILES=("$STAGE/reconunet_cb_paper_best.pt" "$STAGE/reconunet_cb_paper.yaml" "$STAGE/paper_corpus_bwrand.yaml" "$STAGE/reconunet_cb_paper_history.json")
log "before: $(gh release view revision-r2-checkpoints --json assets -q '[.assets[].name]|join(",")' 2>&1)"
for n in 1 2 3; do
  if gh release upload revision-r2-checkpoints "${FILES[@]}" >> "$LOG" 2>&1; then
    log "uploaded to revision-r2-checkpoints; after: $(gh release view revision-r2-checkpoints --json assets -q '[.assets[].name]|join(",")')"
    exit 0
  fi
  log "upload failed (attempt $n)"; sleep 60
done
log "FALLBACK: creating revision-r2b-checkpoints"
TAG=revision-r2b-checkpoints; SHA=$(git rev-parse HEAD)
git tag -a "$TAG" -m "Revision R2b: ReconUNet-CB checkpoint (2026-10-01)" "$SHA" >> "$LOG" 2>&1 || log "tag exists locally"
git -c credential.helper= -c credential.helper='!f() { echo username=x-access-token; echo "password=${GH_TOKEN}"; }; f' \
    push -q origin "refs/tags/$TAG" >> "$LOG" 2>&1 && log "tag pushed"
printf 'ReconUNet-CB (covariance-only ReconUNet, L_rec only, trained on the randomised-source-bandwidth corpus), Sensors revision R2b. Results: docs/revision_r2b/ at %s.\n' "$SHA" > "$STAGE/notes.md"
gh release create "$TAG" --verify-tag --title "Revision R2b checkpoint: ReconUNet-CB (2026-10-01)" --notes-file "$STAGE/notes.md" "${FILES[@]}" >> "$LOG" 2>&1 \
  && { log "release created: $TAG"; exit 0; }
log "release create failed"; exit 1
