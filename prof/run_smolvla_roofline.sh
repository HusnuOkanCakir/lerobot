#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prof/run_smolvla_roofline.sh [options] -- [smolvla_inference args]

Options:
  --checkpoint_path PATH   Path to pretrained_model directory (default: outputs/train/my_smolvla/checkpoints/last/pretrained_model)
  --dataset_repo_id ID     Dataset repo id (default: lerobot/svla_so101_pickplace)
  --project_dir DIR        Intel Advisor project dir (default: prof/intel_roofline)
  --warmup_iters N         Warmup iterations (default: 0)
  --profile_iters N        Profile iterations (default: 1)
  --advisor_bin PATH       Path to Intel Advisor binary (default: advisor)
  --python_bin PATH        Python binary (default: python)
  --timeout SECONDS        Kill the run if it exceeds this time (default: 0 = no timeout)
  --profile_python MODE    Advisor Python profiling: off|stacks|full (default: off)
  -h, --help               Show this help
EOF
}

checkpoint_path="outputs/train/my_smolvla/checkpoints/last/pretrained_model"
dataset_repo_id="lerobot/svla_so101_pickplace"
project_dir="prof/intel_roofline"
warmup_iters=0
profile_iters=1
advisor_bin="advisor"
python_bin="python"
timeout_s=0
profile_python="off"
extra_args=()
pass_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path) checkpoint_path="$2"; shift 2 ;;
    --dataset_repo_id) dataset_repo_id="$2"; shift 2 ;;
    --project_dir) project_dir="$2"; shift 2 ;;
    --warmup_iters) warmup_iters="$2"; shift 2 ;;
    --profile_iters) profile_iters="$2"; shift 2 ;;
    --advisor_bin) advisor_bin="$2"; shift 2 ;;
    --python_bin) python_bin="$2"; shift 2 ;;
    --timeout) timeout_s="$2"; shift 2 ;;
    --profile_python) profile_python="$2"; shift 2 ;;
    --extra_args)
      read -r -a extra_args <<<"$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      pass_args=("$@")
      break
      ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v "$advisor_bin" >/dev/null 2>&1; then
  echo "Intel Advisor not found at '$advisor_bin'." >&2
  echo "Hint: source /opt/intel/oneapi/setvars.sh then rerun, or pass --advisor_bin." >&2
  exit 1
fi

mkdir -p "$project_dir"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"

echo "[IntelAdvisor] Collecting CPU roofline for SmolVLA inference"
echo "[IntelAdvisor] Project dir: $project_dir"

run_cmd=(
  "$advisor_bin" --collect=roofline --profile-python="$profile_python" --project-dir "$project_dir" --
  "$python_bin" prof/smolvla_inference.py
  --device cpu
  --checkpoint_path "$checkpoint_path"
  --dataset_repo_id "$dataset_repo_id"
  "${extra_args[@]}"
  "${pass_args[@]}"
)

if [[ "$timeout_s" -gt 0 ]]; then
  if command -v timeout >/dev/null 2>&1; then
    run_cmd=(timeout --signal=INT "$timeout_s" "${run_cmd[@]}")
  else
    echo "[IntelAdvisor] warning: timeout requested but 'timeout' is not available." >&2
  fi
fi

trap 'echo "[IntelAdvisor] aborting..."; kill -INT "$advisor_pid" 2>/dev/null || true' INT TERM
"${run_cmd[@]}" &
advisor_pid=$!
wait "$advisor_pid"
trap - INT TERM

echo "[IntelAdvisor] Done. Open the project in Advisor GUI or run:"
echo "  $advisor_bin --report=roofline --project-dir $project_dir"
echo "  $advisor_bin --report=roofline --project-dir $project_dir --report-output $project_dir/roofline.html"
