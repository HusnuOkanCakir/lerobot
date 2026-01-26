# Roofline Findings: Pi0 vs SmolVLA (Filtered NVTX Runs)

This note summarizes the memory/compute‑bound trends observed in the filtered NCU roofline exports for:

- Pi0: `prof/ncu_roofline_nvtx_safari_gpu1_pi0_full_launch_filtered/ncu_roofline_nvtx_safari_gpu1_pi0_full_launch_filtered_bounds.csv`
- SmolVLA: `prof/ncu_roofline_nvtx_safari_gpu1_smolvla_full_launch_filtered/ncu_roofline_nvtx_safari_gpu1_smolvla_full_launch_filtered_bounds.csv`

## Method

- Data source: `*_bounds.csv` produced by `plot_ncu_roofline.py --out-list`.
- Only NVTX‑tagged kernels (`<default domain>:policy.model...`) were considered.
- A kernel’s bound type is taken from the `bound` column (`memory` or `compute`).
- “Most memory‑bound” is interpreted as **high memory bandwidth usage** and **small vertical distance** to the memory roofline.

## Overall Summary

Both models are heavily memory‑bound during inference. The share of memory‑bound kernels is high in both cases, with SmolVLA even more skewed toward memory‑bound kernels.

### Pi0 (filtered run)

- Total NVTX‑tagged kernels: **3995**
- Memory‑bound kernels: **3371** (~84%)
- Compute‑bound kernels: **624** (~16%)
- Average `mem_bw_pct`: **~33.4%**
- Max `mem_bw_pct`: **~92.6%**

Top memory‑bound‑score NVTX paths (examples):
- `policy.model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm`

**Interpretation:** Pi0’s strongest memory‑bound pressure concentrates in the **PaliGemma language model stack**, particularly layernorms. This suggests normalization and per‑token ops in the VLM are a dominant memory‑bandwidth consumer during inference.

### SmolVLA (filtered run)

- Total NVTX‑tagged kernels: **3783**
- Memory‑bound kernels: **3610** (~95%)
- Compute‑bound kernels: **173** (~5%)
- Average `mem_bw_pct`: **~18.5%**
- Max `mem_bw_pct`: **~88.5%**

Top memory‑bound‑score NVTX paths (examples):
- `policy.model.vlm_with_expert.vlm.model.connector.modality_projection.proj`
- `policy.model.vlm_with_expert.lm_expert.layers.*.mlp.gate_proj`

**Interpretation:** SmolVLA’s highest memory‑bound score appears in the **VLM connector projection** and **expert MLP projections**, indicating that the connector + expert projections are key memory‑bandwidth hotspots.

## Comparison

- **Memory‑bound dominance:** Both are memory‑bound, SmolVLA more so by kernel count.
- **Location of hotspots:**
  - **Pi0:** VLM language model layernorms dominate.
  - **SmolVLA:** connector projection + expert MLP projections dominate.
- **Bandwidth utilization:** Pi0 shows higher average bandwidth utilization than SmolVLA in these runs, though both reach high peak bandwidth.

## Notes

- These results are from filtered NVTX runs; they exclude kernels outside NVTX ranges.
- Different launch counts, replay mode, or device types can shift which kernels are captured.
- The “memory‑bound score” should be interpreted alongside raw `mem_bw_pct` and `vertical_distance`.

