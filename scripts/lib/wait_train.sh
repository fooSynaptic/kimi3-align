#!/usr/bin/env bash
# Shared wait helper for k3-align train chains.
# Usage:
#   source "$K3/scripts/lib/wait_train.sh"
#   wait_train <label> <log> <expected_steps> [poll_secs]
wait_train() {
  local label=$1
  local log=$2
  local expected=${3:?expected_steps required}
  local poll=${4:-60}
  local pidfile=${WAIT_TRAIN_PIDFILE:-${K3:-.}/logs/e1_rl.pid}
  local last=""
  local prev=-1

  echo "[wait_train] wait $label log=$log expected_steps=$expected poll=${poll}s"
  while true; do
    if [[ -f "$pidfile" ]]; then
      local pid
      pid=$(cat "$pidfile")
      if ! ps -p "$pid" >/dev/null 2>&1; then
        if grep -aEq "NameEntryExistsError|low on memory|EngineCallError|OutOfMemoryError" "$log" 2>/dev/null; then
          echo "[wait_train] $label FAIL (fatal error in log)" >&2
          tail -60 "$log" >&2 || true
          return 1
        fi
        last=$(grep -oE "Train step [0-9]+/" "$log" 2>/dev/null | tail -1 || true)
        local n=0
        if [[ "$last" =~ Train\ step\ ([0-9]+)/ ]]; then
          n="${BASH_REMATCH[1]}"
        fi
        if (( n >= expected )); then
          echo "[wait_train] $label SUCCESS (reached step $n >= $expected)"
          return 0
        fi
        if grep -q "Training completes" "$log" 2>/dev/null && (( n >= expected )); then
          echo "[wait_train] $label SUCCESS (Training completes at step $n)"
          return 0
        fi
        echo "[wait_train] $label FAIL (driver exited; last=$last expected>=$expected)" >&2
        tail -60 "$log" >&2 || true
        return 1
      fi
    fi
    last=$(grep -oE "Train step [0-9]+/" "$log" 2>/dev/null | tail -1 || true)
    if [[ "$last" != "$prev" && -n "$last" ]]; then
      echo "[wait_train] $label progress $last"
      prev=$last
    fi
    local n=0
    if [[ "$last" =~ Train\ step\ ([0-9]+)/ ]]; then
      n="${BASH_REMATCH[1]}"
    fi
    if grep -aq "Training completes" "$log" 2>/dev/null && (( n < expected )); then
      echo "[wait_train] $label FAIL (Training completes at step $n < $expected)" >&2
      tail -40 "$log" >&2 || true
      return 1
    fi
    if grep -aEq "Timeout waiting for key .*update_weights|EngineCallError|update_weights_disk failed" "$log" 2>/dev/null; then
      echo "[wait_train] $label FAIL (weight-update error in log)" >&2
      tail -40 "$log" >&2 || true
      return 1
    fi
    sleep "$poll"
  done
}
