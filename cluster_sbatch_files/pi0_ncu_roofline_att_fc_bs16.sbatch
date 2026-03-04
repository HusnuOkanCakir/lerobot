#!/bin/bash
#SBATCH -J pi0_ncu_stages_bs16
#SBATCH -p gpu_part
#SBATCH -w safari-gpu2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 20:00:00
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

mkdir -p /mnt/galactica/ocakir/logs
mkdir -p /mnt/galactica/ocakir/tmp/ncu
chmod 700 /mnt/galactica/ocakir/tmp /mnt/galactica/ocakir/tmp/ncu
export TMPDIR=/mnt/galactica/ocakir/tmp/ncu

cd /mnt/galactica/ocakir/smolvla/lerobot

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
conda info --envs || true

export HF_HOME=/mnt/galactica/ocakir/hf_cache
export HF_LEROBOT_HOME=/mnt/galactica/ocakir/hf_cache/lerobot
export PATH=/usr/local/cuda/bin:$PATH

hostname
nvidia-smi || true
python -V

RUN_TS="$(date +%Y%m%d_%H%M%S)"
NCU_OUT_BASE="/mnt/galactica/ocakir/smolvla/lerobot/prof/${RUN_TS}_ncu_nvtx_pi0_bs16_stages_gpu2"
echo "NCU output base: ${NCU_OUT_BASE}"

/usr/local/cuda/bin/ncu --nvtx -f \
  --set roofline \
  --target-processes application-only \
  --replay-mode kernel \
  --nvtx-include "regex:BLOCK\\..*/" \
  -o "${NCU_OUT_BASE}" \
  python /mnt/galactica/ocakir/smolvla/lerobot/prof/pi0_inference.py \
    --checkpoint_path lerobot/pi0_base \
    --device cuda \
    --warmup_iters 2 \
    --profile_iters 1 \
    --fresh_batch_each_iter \
    --time_preprocessor \
    --batch_size 16


# "regex:policy\\.model\\..*/" \


  # --nvtx-include "regex:BLOCK\\.attn\\.core_layer\\..*/" \
  # --nvtx-include "regex:BLOCK\\.fc\\.(qkv|proj|mlp)\\..*/" \

    # --nvtx-include "regex:BLOCK\..*/" \



# /usr/local/cuda/bin/ncu --nvtx \
#   --set roofline \
#   --target-processes all \
#   --replay-mode kernel \
#   -o /mnt/galactica/ocakir/smolvla/lerobot/prof/ncu_roofline_nvtx_fury0_pi0_bs1_attn_core\
#    python /mnt/galactica/ocakir/smolvla/lerobot/prof/pi0_inference.py \
#     --checkpoint_path lerobot/pi0_base \
#     --device cuda \
#     --warmup_iters 5 \
#     --profile_iters 200 \
#     --fresh_batch_each_iter \
#     --time_preprocessor \
#     --batch_size 1



# /usr/local/cuda/bin/ncu --nvtx -f \
#   --set roofline \
#   --target-processes all \
#   --replay-mode kernel \
#   --nvtx-include "regex:BLOCK\\..*/" \
#   -o /mnt/galactica/ocakir/smolvla/lerobot/prof/ncu_nvtx_pi0_bs16_stages_gpu2 \
#   python /mnt/galactica/ocakir/smolvla/lerobot/prof/pi0_inference.py \
#     --checkpoint_path lerobot/pi0_base \
#     --device cuda \
#     --warmup_iters 5 \
#     --profile_iters 200 \
#     --fresh_batch_each_iter \
#     --time_preprocessor \
#     --batch_size 16


  # --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
