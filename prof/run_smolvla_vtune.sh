#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prof/run_smolvla_vtune.sh [options] -- [smolvla_inference args]

Options:
  --checkpoint_path PATH   Path to pretrained_model directory (default: outputs/train/my_smolvla/checkpoints/last/pretrained_model)
  --dataset_repo_id ID     Dataset repo id (default: lerobot/svla_so101_pickplace)
  --result_dir DIR         VTune result dir (default: prof/vtune_hpc)
  --analysis NAME          VTune analysis type (default: hpc-performance)
  --warmup_iters N         Warmup iterations (default: 0)
  --profile_iters N        Profile iterations (default: 1)
  --vtune_bin PATH         Path to VTune binary (default: /opt/intel/oneapi/vtune/latest/bin64/vtune)
  --python_bin PATH        Python binary (default: python)
  --timeout SECONDS        Kill the run if it exceeds this time (default: 0 = no timeout)
  -h, --help               Show this help
EOF
}

checkpoint_path="outputs/train/my_smolvla/checkpoints/last/pretrained_model"
dataset_repo_id="lerobot/svla_so101_pickplace"
result_dir="prof/vtune_hpc"
analysis="hpc-performance"
warmup_iters=0
profile_iters=1
vtune_bin="/opt/intel/oneapi/vtune/latest/bin64/vtune"
python_bin="python"
timeout_s=0
pass_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path) checkpoint_path="$2"; shift 2 ;;
    --dataset_repo_id) dataset_repo_id="$2"; shift 2 ;;
    --result_dir) result_dir="$2"; shift 2 ;;
    --analysis) analysis="$2"; shift 2 ;;
    --warmup_iters) warmup_iters="$2"; shift 2 ;;
    --profile_iters) profile_iters="$2"; shift 2 ;;
    --vtune_bin) vtune_bin="$2"; shift 2 ;;
    --python_bin) python_bin="$2"; shift 2 ;;
    --timeout) timeout_s="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      pass_args=("$@")
      break
      ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "$vtune_bin" ]]; then
  if command -v vtune >/dev/null 2>&1; then
    vtune_bin="$(command -v vtune)"
  else
    echo "VTune not found at '$vtune_bin' and not in PATH." >&2
    echo "Hint: source /opt/intel/oneapi/setvars.sh then rerun, or pass --vtune_bin." >&2
    exit 1
  fi
fi

mkdir -p "$result_dir"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"

echo "[VTune] Collecting $analysis for SmolVLA inference"
echo "[VTune] Result dir: $result_dir"

run_cmd=(
  "$vtune_bin" -collect "$analysis" -result-dir "$result_dir" --
  "$python_bin" prof/smolvla_inference.py
  --device cpu
  --checkpoint_path "$checkpoint_path"
  --dataset_repo_id "$dataset_repo_id"
  --warmup_iters "$warmup_iters"
  --profile_iters "$profile_iters"
  "${pass_args[@]}"
)

if [[ "$timeout_s" -gt 0 ]]; then
  if command -v timeout >/dev/null 2>&1; then
    run_cmd=(timeout --signal=INT "$timeout_s" "${run_cmd[@]}")
  else
    echo "[VTune] warning: timeout requested but 'timeout' is not available." >&2
  fi
fi

trap 'echo "[VTune] aborting..."; kill -INT "$vtune_pid" 2>/dev/null || true' INT TERM
"${run_cmd[@]}" &
vtune_pid=$!
wait "$vtune_pid"
trap - INT TERM

echo "[VTune] Done. View summary with:"
echo "  $vtune_bin -report summary -result-dir $result_dir"
echo "[VTune] For roofline charts, open in VTune GUI:"
echo "  $vtune_bin -gui -result-dir $result_dir"
