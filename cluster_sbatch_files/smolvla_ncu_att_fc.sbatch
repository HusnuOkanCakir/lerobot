#!/bin/bash
#SBATCH -J smolvla_ncu_att_fc
#SBATCH -p gpu_part
#SBATCH -w safari_gpu1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 02:00:00
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

/usr/local/cuda/bin/ncu -f --nvtx \
  --nvtx-include "regex:BLOCK\\.attn\\..*/" \
  --nvtx-include "regex:BLOCK\\.fc\\..*/" \
  --set roofline \
  --target-processes all \
  --replay-mode kernel \
  -o /mnt/galactica/ocakir/smolvla/lerobot/prof/ncu_roofline_nvtx_safari_gpu1_smolvla_full_launch_bs16_blocks \
  python /mnt/galactica/ocakir/smolvla/lerobot/prof/smolvla_inference.py \
    --checkpoint_path outputs/train/my_smolvla3/checkpoints/last/pretrained_model \
    --dataset_repo_id lerobot/svla_so101_pickplace \
    --device cuda \
    --warmup_iters 5 \
    --profile_iters 200 \
    --nvtx_layers \
    --nvtx_leaf_only \
    --fresh_batch_each_iter \
    --time_preprocessor \
    --batch_size 16

    
