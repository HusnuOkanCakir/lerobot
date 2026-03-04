# PI0 Stage-Split Findings

This note summarizes the current observations from the stage-split PI0 roofline outputs:

- `prof/pi0_stage_split_points.csv` (`bs1`)
- `prof/pi0_stage_split_points_bs16_subsplit.csv` (`bs16`)

Relevant plots:

- `prof/pi0_stage_split_roofline.png` (`bs1`)
- `prof/pi0_stage_split_roofline_bs16_subsplit.png` (`bs16`)

## Stage Mapping

In the current PI0 instrumentation:

- `sum` = the prefix/prefill-like pass in `sample_actions()`
- `gen` = the iterative denoising passes in `denoise_step()`

These are not identical to autoregressive LLM decode semantics, so AttAcc-style conclusions should be transferred carefully.

## Input Files

### `bs1`

`prof/pi0_stage_split_points.csv` includes an attention sub-split:

- `sum_attn_gemm`
- `sum_attn_other`
- `sum_fc`
- `gen_attn_gemm`
- `gen_attn_other`
- `gen_fc`

This file is useful for understanding whether attention is dominated by GEMM-heavy kernels or by lower-AI "other" kernels.

Plot:

<img src="pi0_stage_split_roofline.png" alt="PI0 stage-split roofline, bs1" width="50%">

### `bs16`

`prof/pi0_stage_split_points_bs16_subsplit.csv` includes an attention sub-split:

- `sum_attn_gemm`
- `sum_attn_other`
- `sum_fc`
- `gen_attn_gemm`
- `gen_attn_other`
- `gen_fc`

So `bs1` and `bs16` are now comparable at the same sub-kernel granularity.

Plot:

<img src="pi0_stage_split_roofline_bs16_subsplit.png" alt="PI0 stage-split roofline, bs16" width="50%">

## Main Observations

### 1. `bs1`: PI0 does not look like AttAcc-style autoregressive decode

At `bs1`, the dominant part of generation-stage attention is GEMM-heavy:

- `gen_attn_gemm`: `AI = 256.99`, `time = 0.0645 s`
- `gen_attn_other`: `AI = 2.19`, `time = 0.0081 s`

Within generation attention at `bs1`:

- `gen_attn_gemm` is about `88.8%` of generation-attention time
- `gen_attn_other` is about `11.2%` of generation-attention time

Aggregated generation-stage values at `bs1`:

- `gen_attn`: `AI = 116.73`, `mem_bw_pct = 16.34`, `time = 0.0726 s`
- `gen_fc`: `AI = 23.81`, `mem_bw_pct = 46.59`, `time = 0.1174 s`

Interpretation:

- At `bs1`, PI0 generation attention is not the most memory-bounded block.
- The memory-like part of generation attention exists, but it is a small fraction of generation-attention time.
- This differs from AttAcc's LLM decode case, where generation attention is much more uniformly memory-bound.

### 2. `bs1`: sum stage is overwhelmingly FC-dominated in time

At `bs1`:

- `sum_fc` time share in sum stage: `94.1%`
- `sum_attn` time share in sum stage: `5.9%`

Aggregated sum-stage values at `bs1`:

- `sum_attn`: `AI = 32.84`, `mem_bw_pct = 56.71`, `time = 0.0333 s`
- `sum_fc`: `AI = 86.23`, `mem_bw_pct = 58.96`, `time = 0.5269 s`

Interpretation:

- Even if sum-stage attention is somewhat more memory-leaning than sum-stage FC, its total time share is small.
- Any stage-level partition that moves all sum attention off GPU is unlikely to move end-to-end latency much at `bs1`.

### 3. `bs16`: PI0 moves closer to the AttAcc intuition

At `bs16`, aggregated attention-stage values are:

- `sum_attn`: `AI = 28.98`, `mem_bw_pct = 67.40`, `time = 0.1597 s`
- `sum_fc`: `AI = 71.12`, `mem_bw_pct = 54.71`, `time = 2.8090 s`
- `gen_attn`: `AI = 35.00`, `mem_bw_pct = 53.43`, `time = 0.0388 s`
- `gen_fc`: `AI = 74.86`, `mem_bw_pct = 59.05`, `time = 0.0644 s`

Interpretation:

- At `bs16`, both `sum_attn` and `gen_attn` have lower AI than their FC counterparts.
- This means PI0 becomes more attention-memory-leaning as batch size increases.
- In particular, `gen_attn` at `bs16` is more memory-bounded than `gen_fc`, which is more aligned with the AttAcc direction.

### 4. `bs16`: even then, stage-level attention is not the whole story

At `bs16`:

- `sum_fc` is still about `94.6%` of sum-stage time
- `gen_fc` is still about `62.4%` of gen-stage time
- `gen_attn` is about `37.6%` of gen-stage time

Within generation attention at `bs16`:

- `gen_attn_gemm` is about `81%` of generation-attention time
- `gen_attn_other` is about `19%` of generation-attention time

Sub-split generation attention values at `bs16`:

- `gen_attn_gemm`: `AI = 50.41`, `time = 0.0316 s`
- `gen_attn_other`: `AI = 1.18`, `time = 0.0072 s`

Interpretation:

- Even when attention becomes more memory-leaning, FC remains a large time contributor.
- A pure stage-level policy of "all gen attention on PIM, everything else on GPU" may be directionally reasonable at higher batch sizes, but it is still coarse.

## Current Takeaway For Partitioning

Based on the current PI0 runs:

- `bs1`: AttAcc's partition does not transfer cleanly.
  - `gen_attn` is not the clearest PIM target at stage level.
  - The best PIM candidate is the low-AI attention remainder, not the full generation-attention block.

- `bs16`: the case for generation-stage attention on PIM becomes stronger.
  - `gen_attn` is more memory-bounded than `gen_fc`.
  - But FC still occupies more generation-stage time than attention.

Current practical conclusion:

- For PI0, a finer-grained partition is more defensible than a pure stage-level partition.
- Best conceptual split so far:
  - keep FC on GPU
  - keep attention GEMM-heavy kernels on GPU
  - consider only the low-AI attention remainder as the strongest PIM candidate

## Why PI0 Differs From AttAcc

AttAcc is centered on batched autoregressive decoder inference, where generation attention becomes KV-read/GEMV dominated.

PI0 is different:

- generation is an iterative denoising process, not single-token decode
- the suffix contains a chunk of action tokens rather than one next-token query
- this makes PI0 generation attention more GEMM-like than LLM decode attention, especially at small batch size

That is the main reason `bs1` PI0 does not show the same behavior as AttAcc.

## Caveats

- `bs1` and `bs16` are now both available with attention sub-split summaries.
- The `bs32` range-replay export was not usable for roofline arithmetic intensity because the report contained range-level rows with missing FP work counters (`no data`), so it is excluded from the conclusions here.

## Recommended Next Steps

1. Generate a `bs16` attention sub-split CSV as well, so `bs1` and `bs16` can be compared at the same granularity.
2. Re-run `bs32` with a replay mode that preserves the required FP work counters for roofline plotting.
3. If the goal is a hardware partition study, evaluate PI0 using both:
   - stage-level split (`sum/gen x attn/fc`)
   - sub-kernel split (`attn_gemm` vs `attn_other`)
