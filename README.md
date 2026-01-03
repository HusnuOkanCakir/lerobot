Original repo: https://github.com/huggingface/lerobot

Installation: https://huggingface.co/docs/lerobot/installation

## Profiling SmolVLA Inference

We have a profiling script at `prof/smolvla_inference.py` to run an inference and collect timing and trace data.

Basic profiling:
```bash
python prof/smolvla_inference.py \
  --checkpoint_path outputs/train/my_smolvla/checkpoints/last/pretrained_model   \
  --dataset_repo_id lerobot/svla_so101_pickplace   \
  --device cuda  \
  --warmup_iters 5  \
  --profile_iters 20 \
  --profile_trace \
  --trace_dir prof/trace_inference/gpu
```

Layer-by-layer:
```bash
python prof/smolvla_inference.py \
  --checkpoint_path outputs/train/my_smolvla/checkpoints/last/pretrained_model   \
  --dataset_repo_id lerobot/svla_so101_pickplace   \
  --device cuda  \
  --warmup_iters 5  \
  --profile_iters 20 \
  --profile_trace \
  --trace_dir prof/trace_inference/gpu_layers\
  --profile_layers \
  --profile_leaf_only
```

When running on a different machine, update:
- `--checkpoint_path` to the local checkpoint or HF model ID.
- `--dataset_repo_id` if the dataset differs or is cached elsewhere.
- `--device` (`cuda`, `cpu`, `mps`) depending on hardware.
- `--trace_dir` to a writable location (use absolute paths on clusters).
