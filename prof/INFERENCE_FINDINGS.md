## Findings so far

### What the current inference script does
- `prof/smolvla_inference.py` runs `warmup_iters` forward passes, then runs `profile_iters` forward passes that are timed.
- When `--print_iter_lat` is enabled, it prints per-iteration latency for the profiled loop only.

### Warmup vs profile behavior
- Nsight Compute (NCU) profiles all kernel launches in the process unless you filter/skip them; warmup and profiled iterations both contribute.
- Changing `warmup_iters` or `profile_iters` in the current setup does not necessarily change roofline results, because the same kernels are executed repeatedly.

### Reusing a batch vs fresh input
- By default, the script preprocesses one sample once and reuses the same batch for all iterations, which emphasizes steady-state model execution.
- This means input data is already on GPU after the first call, so host->device transfer is not part of the per-iteration timing unless explicitly included.

### Fresh batch + preprocessing
- Added `--fresh_batch_each_iter` to fetch a new dataset sample and run the preprocessor each iteration.
- Added `--time_preprocessor` to include preprocessing (and H2D transfer) inside the timed section.
- With `--time_preprocessor`, the first profiled iteration shows a cold-start spike (initial CUDA setup + preprocessing). Subsequent iterations are much faster, representing steady-state.

### Example observation (local run)
- Cold start: ~1.3 s on the first timed iteration when preprocessing is included.
- Steady-state: ~20–25 ms per iteration with fresh samples + preprocessing.
- With warmup enabled, the profiled loop reflects steady-state only.

### Batch size support
- Added `--batch_size` to control DataLoader batch size.
- Per-iteration logs now print both `batch_latency_s` and `per_sample_latency_s`.
- Summary line reports `avg_batch` and `avg_per_sample`.

### Batch-size scaling observations (local runs)

Fresh batch + preprocessing (`--fresh_batch_each_iter --time_preprocessor`, warmup 5, profile 20):
- Batch 2: avg_per_sample ~0.0177 s (17.7 ms)
- Batch 16: avg_per_sample ~0.0144 s (14.4 ms)
  - Batch latency rises with batch size, but per-sample latency drops modestly (some batching efficiency).

No fresh batch / no preprocessing timing (steady-state model-only, warmup 5, profile 20):
- Batch 2: avg_per_sample ~0.00124 s (1.24 ms)
- Batch 16: avg_per_sample ~0.000167 s (0.167 ms)
  - Per-sample latency drops sharply with larger batch size, consistent with better GPU utilization.

Cold-start effect:
- First profiled iteration can be much larger if warmup is 0 (e.g., several seconds for batch 16 when preprocessing is included).
  Use warmup or exclude the first iteration for steady-state comparisons.

### Iteration-latency time series (batch=8, fresh+preproc)
- `prof/iter_latencies.csv` (100 iterations) shows stable batch latency around ~0.10–0.12 s (per-sample ~0.012–0.015 s).
- Two large spikes appear at iter ~45 and ~95 (~5.9 s), likely due to transient I/O/OS hiccups or GPU stalls.
  These rare spikes strongly affect max and mean.

### Batch sweep (1000 iterations, warmup 5, fresh+preproc)
Source: `prof/batch_sweep1000iter.csv`
- avg_per_sample_s decreases modestly with batch size:
  - bs=1: 0.0336 s
  - bs=2: 0.0303 s
  - bs=4: 0.0284 s
  - bs=8: 0.0275 s
  - bs=16: 0.0274 s
- max_batch_s grows with batch size (up to ~11.93 s at bs=16), indicating occasional large stalls even at steady-state.
