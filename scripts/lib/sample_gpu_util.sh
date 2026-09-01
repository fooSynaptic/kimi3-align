#!/usr/bin/env bash
# Background-sample nvidia-smi util/mem across smoke nodes; gate peak mean util.
# Usage:
#   source sample_gpu_util.sh
#   start_gpu_util_sampler <out_tsv>
#   ... train ...
#   stop_gpu_util_sampler <out_tsv>
#   check_gpu_util_gate <out_tsv> [min_pct=50]

start_gpu_util_sampler() {
  local out=${1:?out tsv}
  local interval=${GPU_UTIL_SAMPLE_SECS:-5}
  local hosts=${GPU_UTIL_HOSTS:-"${HEAD_SSH} ${WORKER_HOSTS}"}
  mkdir -p "$(dirname "$out")"
  printf 'ts\thost\tgpu\tutil\tmem_used_mib\tmem_total_mib\tmem_pct\n' > "$out"

  (
    while true; do
      ts=$(date +%s)
      for h in $hosts; do
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
          "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" \
          2>/dev/null | while IFS=',' read -r idx util used total; do
            idx=${idx// /}; util=${util// /}; used=${used// /}; total=${total// /}
            mpct=0
            if [[ -n "${total:-}" && "$total" -gt 0 ]]; then
              mpct=$(( used * 100 / total ))
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$h" "$idx" "$util" "$used" "$total" "$mpct"
          done
      done
      sleep "$interval"
    done
  ) >>"$out" 2>/dev/null &
  GPU_UTIL_SAMPLER_PID=$!
  echo "$GPU_UTIL_SAMPLER_PID" > "${out}.pid"
  echo "[gpu_util] sampler pid=$GPU_UTIL_SAMPLER_PID out=$out interval=${interval}s"
}

stop_gpu_util_sampler() {
  local out=${1:-}
  local pid=${GPU_UTIL_SAMPLER_PID:-}
  if [[ -z "${pid:-}" && -n "$out" && -f "${out}.pid" ]]; then
    pid=$(cat "${out}.pid")
  fi
  if [[ -n "${pid:-}" ]]; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  unset GPU_UTIL_SAMPLER_PID
}

# Peak of per-timestamp mean(utilization.gpu) over GPUs with mem_used>=min_mem (active).
# Pass if peak_mean_util>=min OR peak_mean_mem_pct>=min.
check_gpu_util_gate() {
  local out=${1:?}
  local min_pct=${2:-50}
  local min_mem=${GPU_UTIL_ACTIVE_MEM_MIB:-1024}
  local nums="${out}.nums"

  if [[ ! -s "$out" ]]; then
    echo "[gpu_util] FAIL empty samples $out" >&2
    return 1
  fi

  awk -F'\t' -v min_mem="$min_mem" -v nums="$nums" '
    NR==1 { next }
    $5+0 >= min_mem {
      key=$1
      util_sum[key]+=$4; util_n[key]++
      mem_sum[key]+=$7; mem_n[key]++
      if ($4+0 > peak_gpu) peak_gpu=$4+0
      if ($7+0 > peak_mem_one) peak_mem_one=$7+0
      # Optional: isolate head-node vLLM cards via GPU_UTIL_HEAD_HOST
      if (ENVIRON["GPU_UTIL_HEAD_HOST"] != "" && $2==ENVIRON["GPU_UTIL_HEAD_HOST"]) {
        h_util_sum[key]+=$4; h_util_n[key]++
        h_mem_sum[key]+=$7; h_mem_n[key]++
      }
      active++
    }
    END {
      peak_mean_util=0; peak_mean_mem=0
      peak_head_util=0; peak_head_mem=0
      for (k in util_n) {
        mu=util_sum[k]/util_n[k]
        mm=mem_sum[k]/mem_n[k]
        if (mu>peak_mean_util) peak_mean_util=mu
        if (mm>peak_mean_mem) peak_mean_mem=mm
      }
      for (k in h_util_n) {
        mu=h_util_sum[k]/h_util_n[k]
        mm=h_mem_sum[k]/h_mem_n[k]
        if (mu>peak_head_util) peak_head_util=mu
        if (mm>peak_head_mem) peak_head_mem=mm
      }
      printf "peak_mean_util=%.1f peak_mean_mem=%.1f peak_head_util=%.1f peak_head_mem=%.1f peak_single_util=%d peak_single_mem=%d active_rows=%d\n",
        peak_mean_util, peak_mean_mem, peak_head_util, peak_head_mem, peak_gpu+0, peak_mem_one+0, active+0
      # util_sm mem_pct head_sm head_mem single_sm single_mem
      printf "%.0f %.0f %.0f %.0f %.0f %.0f\n",
        peak_mean_util, peak_mean_mem, peak_head_util, peak_head_mem, peak_gpu+0, peak_mem_one+0 > nums
    }
  ' "$out"

  local peak_u=0 peak_m=0 head_u=0 head_m=0 single_u=0 single_m=0
  if [[ -f "$nums" ]]; then
    read -r peak_u peak_m head_u head_m single_u single_m < "$nums" || true
  fi
  peak_u=${peak_u:-0}; peak_m=${peak_m:-0}
  head_u=${head_u:-0}; head_m=${head_m:-0}
  single_u=${single_u:-0}; single_m=${single_m:-0}

  # Any of: cluster mean SM/mem, head-node mean SM/mem, or single-GPU peak.
  if (( peak_u >= min_pct || peak_m >= min_pct || head_u >= min_pct || head_m >= min_pct || single_u >= min_pct || single_m >= min_pct )); then
    echo "[gpu_util] PASS min=${min_pct}% (head_util=${head_u}% head_mem=${head_m}% single_util=${single_u}% single_mem=${single_m}%)"
    echo "head_sm=${head_u} head_mem=${head_m} single_sm=${single_u} single_mem=${single_m}" > "${out}.pass"
    return 0
  fi
  echo "[gpu_util] FAIL min=${min_pct}% (head_util=${head_u}% head_mem=${head_m}% single_util=${single_u}% single_mem=${single_m}%)" >&2
  return 1
}
