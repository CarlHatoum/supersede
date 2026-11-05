#!/usr/bin/env bash
# Run librarian-agent jobs one after another (one Ollama model loaded at a time).
# Each line of the queue file holds the arguments of one agent_lineage run.
# Runs are resumable: decisions are cached per pair, so re-running the queue
# only does what is missing.
#
#   scripts/agent_queue.sh scripts/queues/night1.txt
set -u
queue="$1"
py="${PY:-.venv/bin/python}"
log_dir="results/agent/logs"
mkdir -p "$log_dir"
prev_model=""
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  model=$(sed -n 's/.*--model \([^ ]*\).*/\1/p' <<<"$line")
  model=${model:-ollama:edtorre/qwen3.5-hermes}
  if [ -n "$prev_model" ] && [ "$model" != "$prev_model" ]; then
    ollama stop "${prev_model#ollama:}" >/dev/null 2>&1 || true
  fi
  prev_model="$model"
  name=$(tr -c 'a-zA-Z0-9.' '_' <<<"$line" | cut -c1-120)
  echo "[$(date '+%F %T')] start: $line"
  # shellcheck disable=SC2086
  $py -m supersede.experiments.agent_lineage $line >"$log_dir/$name.log" 2>&1
  echo "[$(date '+%F %T')] exit $?: $line"
done <"$queue"
echo "[$(date '+%F %T')] queue done"
