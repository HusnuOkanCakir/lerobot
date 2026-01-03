#!/bin/bash
#SBATCH -J pi0_infer
#SBATCH -p gpu_part
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH -t 00:30:00
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

cd /mnt/galactica/ocakir/smolvla/lerobot

. /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# sanity prints
hostname
nvidia-smi || true
python -V

# logs dir exists
mkdir -p /mnt/galactica/ocakir/logs

# export HF_HUB_OFFLINE=1

python /mnt/galactica/ocakir/smolvla/lerobot/prof/simple_pi0_inference.py \
  --checkpoint_path lerobot/pi0_base \
  --device cuda \
  --use_dummy_input \
  --profile_trace \
  --trace_dir prof/trace_inference_pi0/gpu \
  --warmup_iters 1 \
  --profile_iters 1
