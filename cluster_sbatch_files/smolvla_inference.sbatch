#!/bin/bash
#SBATCH -J smolvla_infer
#SBATCH -p gpu_part
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 00:10:00
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

python /mnt/galactica/ocakir/smolvla/lerobot/prof/smolvla_inference.py \
  --checkpoint_path outputs/train/my_smolvla3/checkpoints/last/pretrained_model \
  --dataset_repo_id lerobot/svla_so101_pickplace \
  --device cuda \
  --profile_trace \
  --trace_dir prof/trace_inference/gpu \
  --warmup_iters 5 \
  --profile_iters 20 \
  --profile_layers
