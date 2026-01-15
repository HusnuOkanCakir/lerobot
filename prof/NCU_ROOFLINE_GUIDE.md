# Nsight Compute Roofline: SmolVLA Inference Repro

This guide documents the exact Linux workflow used to profile `prof/smolvla_inference.py` with
Nsight Compute (NCU), export CSV metrics, and plot a combined roofline chart with bounds.

## What This Produces

- An `.ncu-rep` report (NCU output).
- A raw CSV export of all kernel metrics.
- A combined roofline plot (all kernels in one chart).
- A ranked CSV of kernels by distance to the roofline.

Outputs are written under `prof/<rep_basename>/`.

## Prerequisites

- NVIDIA driver compatible with your Nsight Compute version.
- Nsight Compute CLI `ncu` on PATH.
- Python environment with `torch`, `pandas`, and `matplotlib`.
- Dataset used by `prof/smolvla_inference.py` available locally.

### Driver compatibility

If you see:
```
Cuda driver is not compatible with Nsight Compute.
```
the driver is too old for your `ncu`. Fix by either:
- Updating the NVIDIA driver, or
- Installing an older Nsight Compute.

Verify:
```
nvidia-smi
ncu --version
```

### One-time performance counter permission (no sudo later)

If you see:
```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```
apply a one-time driver setting and reboot:
```
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-prof.conf
sudo reboot
```
After reboot, `ncu` runs as a normal user.

[source](https://forums.developer.nvidia.com/t/nvprof-warning-the-user-does-not-have-permission-to-profile-on-the-target-device/72374/5)

## Inference Script Summary

File: `prof/smolvla_inference.py`

What it does:
- Loads a pretrained SmolVLA policy and dataset sample.
- Runs warmup iterations, then timed inference iterations.
- Optional per-layer CPU timing via forward hooks.
- Optional NVTX ranges for layer mapping.
- Optional Torch profiler traces.

Useful flags:
- `--device cuda|cpu|mps|auto`
- `--warmup_iters N`
- `--profile_iters N`
- `--nvtx_layers` and `--nvtx_leaf_only`
- `--profile_trace`

## 1) Run Nsight Compute (roofline)

We used:
```
ncu --nvtx --set roofline  --target-processes all  --replay-mode kernel --launch-count 1000  -o prof/ncu_roofline_nvtx_1000l  -- python ./prof/smolvla_inference.py  --checkpoint_path outputs/train/my_smolvla/checkpoints/last/pretrained_model --dataset_repo_id lerobot/svla_so101_pickplace  --device cuda  --warmup_iters 5  --profile_iters 20 --nvtx_layers --nvtx_leaf_only
```

Flags:
- `--set roofline`: collect roofline metrics.
- `--launch-count <n>`: profile first n kernel launches.
- `--target-processes all`: include child processes.
- `--replay-mode kernel`: replay kernels for full metrics.
- `--export <path>`: write `<path>.ncu-rep`.

Optional: add `--nvtx` to the `ncu` command to capture NVTX ranges in the report.

Output:
- `prof/smolvla_inference_roofline.ncu-rep`

## 2) Export CSV + Plot Roofline + Bounds

Use the helper script:
```
python prof/export_ncu_and_plot.py prof/smolvla_inference_roofline.ncu-rep
```

What it does:
1) `ncu --import <rep> --page raw --csv` -> `<rep_basename>.csv`
2) `plot_ncu_roofline.py` -> `<rep_basename>.png`
3) `plot_ncu_roofline.py --out-list` -> `<rep_basename>_bounds.csv`

Output layout:
```
prof/smolvla_inference_roofline/
  smolvla_inference_roofline.csv
  smolvla_inference_roofline.png
  smolvla_inference_roofline_bounds.csv
```

## Roofline Plot Meaning

- X-axis: arithmetic intensity (FLOP/byte), log scale.
- Y-axis: achieved performance (FLOP/s), log scale.
- Each point is a kernel; size/color reflect duration or memory bw usage.
- Memory roof: performance limit from bandwidth.
- Compute roof: performance limit from peak compute.

Interpretation:
- Left of ridge and below memory roof = memory bound.
- Right of ridge and below compute roof = compute bound.
- Farther below the roof = more headroom.

## Plot Script Details

File: `prof/plot_ncu_roofline.py`

Achieved performance uses per-cycle FP32 metrics:
```
ops_per_cycle = fadd_per_cycle + fmul_per_cycle + ffma_x2_per_cycle
ops_per_sec = ops_per_cycle * cycles_per_sec
```

Traffic uses DRAM bytes per second:
```
bytes_per_sec = dram__bytes.sum.per_second (or first available DRAM bytes column)
```

If `--use-peak-formula` is enabled and the columns exist:
```
peak_flops = derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2
             * sm__cycles_elapsed.avg.per_second
peak_bw = dram__bytes.sum.peak_sustained
          * dram__cycles_elapsed.avg.per_second
```
These  formulas can be found in the ncu gui. [source](https://forums.developer.nvidia.com/t/how-to-export-raw-data-of-roofline-in-nsight-compute-ncu/347635/5)

## NVTX Layer Mapping (optional)

`prof/smolvla_inference.py` can emit NVTX ranges:
- `--nvtx_layers`: enable ranges.
- `--nvtx_leaf_only`: leaf modules only.

NCU exports NVTX names in these columns when present:
- `thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg`
- `Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg`

The plot script uses NVTX names first, then falls back to Kernel Name.

This is useful to understand which kernel is associated with which layer of the model.

## Troubleshooting Tips

- Driver mismatch: update NVIDIA driver or downgrade Nsight Compute.
- Permission error: apply `NVreg_RestrictProfilingToAdminUsers=0` and reboot.
- Too many kernels: reduce with `--launch-count`.
- Missing metrics: try `--set full` (slower) or add explicit metrics.

## Quick Repro Checklist

1) Profile:
```
ncu --nvtx --set roofline  --target-processes all  --replay-mode kernel --launch-count 1000  -o prof/ncu_roofline_nvtx_1000l  -- python ./prof/smolvla_inference.py  --checkpoint_path outputs/train/my_smolvla/checkpoints/last/pretrained_model --dataset_repo_id lerobot/svla_so101_pickplace  --device cuda  --warmup_iters 5  --profile_iters 20 --nvtx_layers --nvtx_leaf_only
```

2) Export + plot:
```
python prof/export_ncu_and_plot.py --rep prof/ncu_roofline_nvtx_1000l.ncu-rep
```
