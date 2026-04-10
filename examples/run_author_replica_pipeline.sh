#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ylhc/miniconda3/envs/babybench/bin/python}"
DEVICE="${DEVICE:-cuda}"
POLL_SECONDS="${POLL_SECONDS:-60}"

BASE_DIR="${BASE_DIR:-$ROOT_DIR/results_author_replica/intrinsic_motivation_stelios_x_giannis_smooth}"
DIFFICULT_DIR="${DIFFICULT_DIR:-$ROOT_DIR/results_author_replica/resume_random_init}"
AFTER_DIR="${AFTER_DIR:-$ROOT_DIR/results_author_replica/intrinsic_motivation_stelios_x_giannis_after_difficult_task}"

BASE_MODEL="$BASE_DIR/stelios_x_giannis_ppo_final.zip"
DIFFICULT_MODEL="$DIFFICULT_DIR/stelios_x_giannis_ppo_final.zip"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf "[%s] %s\n" "$(timestamp)" "$*"
}

mkdir -p "$DIFFICULT_DIR" "$AFTER_DIR"

log "Author-replica pipeline controller started"
log "Waiting for base final checkpoint: $BASE_MODEL"

while [[ ! -f "$BASE_MODEL" ]]; do
  sleep "$POLL_SECONDS"
done

log "Detected base final checkpoint"

if [[ -f "$DIFFICULT_MODEL" ]]; then
  log "Difficult final checkpoint already exists, skipping difficult stage"
else
  log "Starting difficult stage"
  log "Difficult stage log: $DIFFICULT_DIR/train.log"
  "$PYTHON_BIN" "$ROOT_DIR/examples/intrinsic_motivation_expand_on_difficult_task.py" \
    --device "$DEVICE" \
    --save_dir "$DIFFICULT_DIR" \
    --resume_model "$BASE_MODEL" \
    > "$DIFFICULT_DIR/train.log" 2>&1
  log "Difficult stage completed"
fi

if [[ -f "$DIFFICULT_MODEL" ]]; then
  log "Starting after stage"
  log "After stage log: $AFTER_DIR/train.log"
  "$PYTHON_BIN" "$ROOT_DIR/examples/intrinsic_motivation_after_difficult_task.py" \
    --device "$DEVICE" \
    --save_dir "$AFTER_DIR" \
    --resume_model "$DIFFICULT_MODEL" \
    > "$AFTER_DIR/train.log" 2>&1
  log "After stage completed"
else
  log "Difficult final checkpoint missing, after stage will not start"
  exit 1
fi

log "Author-replica pipeline finished"
