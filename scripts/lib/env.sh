#!/usr/bin/env bash
# Shared environment for k3-align scripts.
# Source from any script under scripts/:
#   SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
#   # shellcheck source=lib/env.sh
#   source "$SCRIPT_DIR/lib/env.sh"
#
# Required for train/eval (no cluster-path fallbacks in this file):
#   export SAO_ROOT=/path/to/Single-rollout-async-Optimization
#   export MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
#   export DATA_PATH=/path/to/gsm8k_hard
#   export HEAD_SSH=... HEAD_HOST=... WORKER_SPECS=...
#
# Shell options (`set -euo pipefail`) live in callers — sourced lib stays composable across smoke/full wrappers.

_k3_lib_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_k3_scripts_dir=$(cd "${_k3_lib_dir}/.." && pwd)
_k3_repo_root=$(cd "${_k3_scripts_dir}/.." && pwd)

if [[ -z "${K3_ROOT:-}" ]]; then
  if [[ -n "${K3:-}" ]]; then
    K3_ROOT="$K3"
  else
    K3_ROOT="$_k3_repo_root"
  fi
fi

if [[ -z "${SAO_ROOT:-}" && -n "${SAO:-}" ]]; then
  SAO_ROOT="$SAO"
fi

export K3_ROOT
export K3="$K3_ROOT"
export SAO_ROOT="${SAO_ROOT:-}"
export SAO="${SAO_ROOT:-}"

if [[ -n "${SAO_ROOT:-}" ]]; then
  if [[ -z "${AREAL_REPO:-}" ]]; then
    if [[ -d "$SAO_ROOT/vendor/AReaL" ]]; then
      AREAL_REPO="$SAO_ROOT/vendor/AReaL"
    elif [[ -d "$SAO_ROOT/repo/AReaL" ]]; then
      AREAL_REPO="$SAO_ROOT/repo/AReaL"
    fi
  fi
  : "${VENV:=$SAO_ROOT/tmp/phase2-runtime}"
  : "${HF_HOME:=$SAO_ROOT/tmp/hf-home}"
  : "${HF_DATASETS_CACHE:=$SAO_ROOT/tmp/hf-datasets}"
fi

if [[ -n "${VENV:-}" ]]; then
  if [[ -d "$VENV/lib/python3.12/site-packages" ]]; then
    : "${SITE:=$VENV/lib/python3.12/site-packages}"
  else
    SITE=$(ls -d "$VENV"/lib/python3.*/site-packages 2>/dev/null | head -1 || true)
  fi
fi

: "${K3_PY:=$K3/pythonpath}"
: "${CK_ROOT:=$K3/experiments/checkpoints/root/k3-align-math-rl}"
: "${LOG_ROOT:=$K3/experiments/logs/root/k3-align-math-rl}"
: "${CURVES_DIR:=$K3/docs/curves}"
: "${K25_TAU_LOG_RATIO_SQ:=0.01}"
: "${P1_CKPT:=$CK_ROOT/p1-k25-sync-h0-full/actor/epoch4epochstep15globalstep199}"
: "${E0_CKPT:=$K3/experiments/checkpoints/root/k3-e0-math-sft/qwen3-4b-instruct-20gpu-r2/default/epoch2epochstep45globalstep137}"
: "${SFT_DATA_PATH:=$K3/data/math_sft_chat}"
: "${P5_STAGE_A_CKPT:=PLACEHOLDER_STAGE_A}"

export AREAL_REPO="${AREAL_REPO:-}"
export VENV="${VENV:-}"
export SITE="${SITE:-}"
export K3_PY CK_ROOT LOG_ROOT CURVES_DIR
export HF_HOME="${HF_HOME:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"
export K25_TAU_LOG_RATIO_SQ
export MODEL_PATH="${MODEL_PATH:-}"
export DATA_PATH="${DATA_PATH:-}"
export P1_CKPT E0_CKPT SFT_DATA_PATH P5_STAGE_A_CKPT

k3_require_sao() {
  if [[ -z "${SAO_ROOT:-}" || ! -d "${SAO_ROOT}" ]]; then
    echo "[fatal] SAO_ROOT unset or missing. Clone https://github.com/fooSynaptic/Single-rollout-async-Optimization and export SAO_ROOT." >&2
    exit 1
  fi
  if [[ -z "${VENV:-}" || ! -x "${VENV}/bin/python" ]]; then
    echo "[fatal] VENV missing executable python: ${VENV:-<unset>}" >&2
    echo "        After SAO bootstrap, export VENV to that environment." >&2
    exit 1
  fi
}

k3_require_areal() {
  k3_require_sao
  if [[ -z "${AREAL_REPO:-}" || ! -d "${AREAL_REPO}" ]]; then
    echo "[fatal] AREAL_REPO missing: ${AREAL_REPO:-<unset>}" >&2
    exit 1
  fi
}

k3_require_train_paths() {
  k3_require_areal
  if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[fatal] MODEL_PATH unset (Instruct ckpt or HF id)." >&2
    exit 1
  fi
  if [[ -z "${DATA_PATH:-}" ]]; then
    echo "[fatal] DATA_PATH unset (gsm8k_hard directory)." >&2
    exit 1
  fi
}

k3_require_ray_topo() {
  if [[ -z "${HEAD_SSH:-}" || -z "${HEAD_HOST:-}" ]]; then
    echo "[fatal] Set HEAD_SSH (ssh alias) and HEAD_HOST (Ray node IP) for your cluster." >&2
    exit 1
  fi
  if [[ -z "${WORKER_SPECS:-}" ]]; then
    echo "[fatal] Set WORKER_SPECS, e.g. 'worker-a:8:0,1,2,3,4,5,6,7 worker-b:8:0,1,2,3,4,5,6,7'" >&2
    exit 1
  fi
}

k3_mkdirs() {
  mkdir -p "$K3/logs" "$K3/experiments" "$K3/tmp/home" "$K3/tmp/name_resolve" "$CURVES_DIR"
}
