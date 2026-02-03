#!/usr/bin/env python

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import re

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

try:
    from torch.cuda import nvtx
except Exception:
    nvtx = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, POLICY_POSTPROCESSOR_DEFAULT_NAME


def _infer_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_dummy_batch(policy_cfg: PreTrainedConfig, device: str, batch_size: int) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key, feature in policy_cfg.input_features.items():
        shape = tuple(feature.shape)
        batch[key] = torch.zeros((batch_size, *shape), dtype=torch.float32, device=device)
    max_len = getattr(policy_cfg, "tokenizer_max_length", 48)
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
    return batch


def _expected_image_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    image_features = getattr(policy_cfg, "image_features", None)
    if isinstance(image_features, dict):
        return list(image_features.keys())
    return []


def _parse_image_key_map(mappings: list[str]) -> dict[str, str]:
    key_map: dict[str, str] = {}
    for entry in mappings:
        if ":" not in entry:
            raise ValueError(f"--image_key_map must be in SRC:DST format, got: {entry}")
        src, dst = entry.split(":", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"--image_key_map must be in SRC:DST format, got: {entry}")
        key_map[src] = dst
    return key_map


def _apply_image_key_mapping(
    batch: dict[str, torch.Tensor],
    expected_keys: list[str],
    key_map: dict[str, str],
    *,
    fill_missing: bool,
) -> dict[str, torch.Tensor]:
    if not expected_keys:
        return batch
    for src, dst in key_map.items():
        if src in batch and dst not in batch:
            batch[dst] = batch[src]
    present = [key for key in expected_keys if key in batch]
    if not present:
        return batch
    if fill_missing:
        fallback = batch[present[0]]
        for key in expected_keys:
            if key not in batch:
                batch[key] = fallback
    return batch


def _analyze_trace_dir(trace_dir: str) -> None:
    path = Path(trace_dir)
    if not path.exists():
        print(f"[TraceAnalysis] trace_dir does not exist: {trace_dir}")
        return
    traces = sorted(path.rglob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
    if not traces:
        print(f"[TraceAnalysis] no trace files found in: {trace_dir}")
        return
    latest = traces[-1]
    _analyze_trace_file(latest)


def _analyze_trace_file(trace_path: Path) -> None:
    with trace_path.open("r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    kernel_dur_us = 0.0
    memcpy_dur_us = 0.0
    cpu_memmove_dur_us = 0.0
    cpu_memmove_counts = 0
    total_dur_us = 0.0
    cpu_op_dur_us = 0.0
    cpu_op_counts = 0
    cpu_memmove_ops = defaultdict(float)
    for e in events:
        dur = e.get("dur")
        if not isinstance(dur, (int, float)):
            continue
        total_dur_us += dur
        cat = (e.get("cat") or "").lower()
        name = (e.get("name") or "").lower()
        if "kernel" in cat:
            kernel_dur_us += dur
        if "gpu_memcpy" in cat or "memcpy" in name:
            memcpy_dur_us += dur
        if "cpu_op" in cat:
            cpu_op_dur_us += dur
            cpu_op_counts += 1
            if any(
                key in name
                for key in (
                    "copy_",
                    "contiguous",
                    "cat",
                    "transpose",
                    "permute",
                    "reshape",
                    "view",
                    "slice",
                    "index_select",
                    "gather",
                    "scatter",
                )
            ):
                cpu_memmove_dur_us += dur
                cpu_memmove_counts += 1
                cpu_memmove_ops[name] += dur

    total_ms = total_dur_us / 1000.0
    kernel_ms = kernel_dur_us / 1000.0
    memcpy_ms = memcpy_dur_us / 1000.0
    cpu_memmove_ms = cpu_memmove_dur_us / 1000.0
    kernel_pct = (kernel_dur_us / total_dur_us * 100.0) if total_dur_us > 0 else 0.0
    memcpy_pct = (memcpy_dur_us / total_dur_us * 100.0) if total_dur_us > 0 else 0.0
    cpu_memmove_pct = (cpu_memmove_dur_us / total_dur_us * 100.0) if total_dur_us > 0 else 0.0
    print(f"[TraceAnalysis] Latest trace: {trace_path}")
    print(
        "[TraceAnalysis] Durations (ms): "
        f"total={total_ms:.2f} kernel={kernel_ms:.2f} memcpy={memcpy_ms:.2f} "
        f"cpu_memmove={cpu_memmove_ms:.2f}"
    )
    print(
        "[TraceAnalysis] Percent of total duration: "
        f"kernel={kernel_pct:.2f}% memcpy={memcpy_pct:.2f}% cpu_memmove={cpu_memmove_pct:.2f}%"
    )
    if cpu_op_counts > 0:
        cpu_op_pct = (cpu_op_dur_us / total_dur_us * 100.0) if total_dur_us > 0 else 0.0
        print(
            "[TraceAnalysis] CPU ops: "
            f"{cpu_op_counts} events, {cpu_op_dur_us/1000.0:.2f}ms ({cpu_op_pct:.2f}%)"
        )
    if cpu_memmove_counts > 0:
        top_ops = sorted(cpu_memmove_ops.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("[TraceAnalysis] Top CPU memmove-like ops (ms):")
        for name, dur in top_ops:
            print(f"  {name}: {dur/1000.0:.2f}")


class LayerProfiler:
    def __init__(self, device: str, leaf_only: bool = True) -> None:
        self.device = device
        self.leaf_only = leaf_only
        self._start_times: dict[int, float] = {}
        self._records: dict[str, list[float]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._id_to_name: dict[int, str] = {}

    def _sync(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _is_leaf(self, module: torch.nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _pre_hook(self, module: torch.nn.Module, _inputs) -> None:
        self._sync()
        self._start_times[id(module)] = time.perf_counter()

    def _post_hook(self, module: torch.nn.Module, _inputs, _output) -> None:
        self._sync()
        start = self._start_times.get(id(module))
        if start is None:
            return
        elapsed = time.perf_counter() - start
        name = self._id_to_name.get(id(module), module.__class__.__name__)
        self._records[name].append(elapsed)

    def attach(self, module: torch.nn.Module, prefix: str = "model") -> None:
        for name, child in module.named_modules():
            if name == "":
                continue
            if self.leaf_only and not self._is_leaf(child):
                continue
            full_name = f"{prefix}.{name}"
            self._id_to_name[id(child)] = full_name
            self._handles.append(child.register_forward_pre_hook(self._pre_hook))
            self._handles.append(child.register_forward_hook(self._post_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self, top_n: int = 50) -> None:
        rows = []
        for name, times in self._records.items():
            total = sum(times)
            count = len(times)
            rows.append((total, name, count, total / count, min(times), max(times)))
        rows.sort(reverse=True, key=lambda r: r[0])
        print("\n[LayerProfiler] Per-layer timings (seconds):")
        print(f"{'total':>12} {'count':>7} {'avg':>12} {'min':>12} {'max':>12}  name")
        for total, name, count, avg, tmin, tmax in rows[:top_n]:
            print(f"{total:12.6f} {count:7d} {avg:12.6f} {tmin:12.6f} {tmax:12.6f}  {name}")


class NvtxLayerRanges:
    def __init__(self, leaf_only: bool = False) -> None:
        self.leaf_only = leaf_only
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _is_leaf(self, module: torch.nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _pre_hook(self, module: torch.nn.Module, _inputs) -> None:
        if nvtx is None:
            return
        name = getattr(module, "_nvtx_name", module.__class__.__name__)
        nvtx.range_push(name)

    def _post_hook(self, module: torch.nn.Module, _inputs, _output) -> None:
        if nvtx is None:
            return
        nvtx.range_pop()

    def attach(self, module: torch.nn.Module, prefix: str = "model") -> None:
        for name, child in module.named_modules():
            if name == "":
                continue
            if self.leaf_only and not self._is_leaf(child):
                continue
            child._nvtx_name = f"{prefix}.{name}"
            self._handles.append(child.register_forward_pre_hook(self._pre_hook))
            self._handles.append(child.register_forward_hook(self._post_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class BlockLatencyProfiler:
    def __init__(self, device: str, attn_patterns: list[str], fc_patterns: list[str]) -> None:
        self.device = device
        self._attn_re = [re.compile(p) for p in attn_patterns]
        self._fc_re = [re.compile(p) for p in fc_patterns]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._module_category: dict[int, str] = {}
        self._start: dict[int, object] = {}
        self._events: list[tuple[str, object, object]] = []

    def _match(self, name: str) -> str | None:
        if any(r.search(name) for r in self._attn_re):
            return "attn"
        if any(r.search(name) for r in self._fc_re):
            return "fc"
        return None

    def _pre_hook(self, module: torch.nn.Module, _inputs) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._start[id(module)] = evt
        else:
            self._start[id(module)] = time.perf_counter()

    def _post_hook(self, module: torch.nn.Module, _inputs, _output) -> None:
        start = self._start.get(id(module))
        if start is None:
            return
        if self.device == "cuda" and torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._events.append((self._module_category[id(module)], start, end))
        else:
            end = time.perf_counter()
            self._events.append((self._module_category[id(module)], start, end))

    def attach(self, module: torch.nn.Module, prefix: str = "model") -> None:
        for name, child in module.named_modules():
            if name == "":
                continue
            full_name = f"{prefix}.{name}"
            category = self._match(full_name)
            if category is None:
                continue
            self._module_category[id(child)] = category
            self._handles.append(child.register_forward_pre_hook(self._pre_hook))
            self._handles.append(child.register_forward_hook(self._post_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._module_category.clear()
        self._start.clear()
        self._events.clear()

    def begin_iter(self) -> None:
        self._events.clear()

    def end_iter(self) -> dict[str, float]:
        totals_ms = {"attn": 0.0, "fc": 0.0}
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            for category, start_evt, end_evt in self._events:
                totals_ms[category] += start_evt.elapsed_time(end_evt)
        else:
            for category, start_t, end_t in self._events:
                totals_ms[category] += (end_t - start_t) * 1000.0
        return totals_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single PI0 inference on a dataset sample.")
    parser.add_argument(
        "--checkpoint_path",
        default="outputs/train/my_pi0/checkpoints/last/pretrained_model",
        help="Path to the pretrained_model directory from a checkpoint.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        default="HuggingFaceVLA/libero",
        help="Dataset repo id used during training (must be available in local cache).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (dummy batch or DataLoader batch size).",
    )
    parser.add_argument(
        "--profile_layers",
        action="store_true",
        help="Enable per-layer timing using forward hooks.",
    )
    parser.add_argument(
        "--profile_leaf_only",
        action="store_true",
        help="Only profile leaf modules (default: True).",
    )
    parser.add_argument(
        "--profile_top_n",
        type=int,
        default=50,
        help="Number of layers to print in the timing report.",
    )
    parser.add_argument(
        "--nvtx_layers",
        action="store_true",
        help="Emit NVTX ranges per module for Nsight profiling.",
    )
    parser.add_argument(
        "--nvtx_leaf_only",
        action="store_true",
        help="Only emit NVTX ranges for leaf modules.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=3,
        help="Number of warmup iterations before profiling.",
    )
    parser.add_argument(
        "--profile_iters",
        type=int,
        default=10,
        help="Number of iterations to time/profile.",
    )
    parser.add_argument(
        "--print_iter_lat",
        action="store_true",
        help="Print per-iteration latency for the profiled iterations.",
    )
    parser.add_argument(
        "--profile_trace",
        action="store_true",
        help="Enable torch.profiler trace export for TensorBoard.",
    )
    parser.add_argument(
        "--trace_dir",
        default="prof/trace_inference_pi0",
        help="Output directory for TensorBoard trace files.",
    )
    parser.add_argument(
        "--use_dummy_input",
        action="store_true",
        help="Run inference on a synthetic sample to avoid dataset downloads.",
    )
    parser.add_argument(
        "--fresh_batch_each_iter",
        action="store_true",
        help="Fetch a new batch each iteration (dummy or dataset) before select_action.",
    )
    parser.add_argument(
        "--time_preprocessor",
        action="store_true",
        help="Include preprocessor (and H2D transfer) time in per-iteration timing.",
    )
    parser.add_argument(
        "--profile_blocks",
        action="store_true",
        help="Measure attention/FC block latency with forward hooks.",
    )
    parser.add_argument(
        "--attn_regex",
        action="append",
        default=[r"\.self_attn\.", r"attention", r"attn"],
        help="Regex for attention block matching (repeatable).",
    )
    parser.add_argument(
        "--fc_regex",
        action="append",
        default=[r"\.mlp\.", r"\.mlp\.fc\d+", r"\.mlp\.(gate|up|down)_proj", r"action_.*_proj", r"state_proj"],
        help="Regex for FC/MLP block matching (repeatable).",
    )
    parser.add_argument(
        "--block_lat_csv",
        default="",
        help="Output CSV for per-iteration block latencies (default: prof/block_latencies.csv).",
    )
    parser.add_argument(
        "--block_lat_plot",
        default="",
        help="Output PNG for per-iteration block latency plot (default: prof/block_latencies.png).",
    )
    parser.add_argument(
        "--image_key_map",
        action="append",
        default=[
            "observation.images.image:observation.images.base_0_rgb",
            "observation.images.image2:observation.images.left_wrist_0_rgb",
        ],
        help="Map image keys from dataset to policy expected keys (repeatable SRC:DST).",
    )
    parser.add_argument(
        "--no_fill_missing_images",
        action="store_false",
        dest="fill_missing_images",
        default=True,
        help="Disable filling missing expected image keys by copying the first available image key.",
    )
    parser.add_argument(
        "--iter_csv",
        default="",
        help="Output CSV for per-iteration latencies (default: prof/iter_latencies.csv).",
    )
    parser.add_argument(
        "--iter_plot",
        default="",
        help="Output PNG for per-iteration latency plot (default: prof/iter_latencies.png).",
    )
    parser.add_argument(
        "--analyze_trace",
        action="store_true",
        help="Analyze the latest trace in --trace_dir for compute vs memcpy time.",
    )
    args = parser.parse_args()

    device = _infer_device(args.device)
    print(f"[SimpleInference] Using device: {device}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint_path)
    policy_cfg.device = device

    if not hasattr(policy_cfg, "freeze_vision_encoder"):
        policy_cfg.freeze_vision_encoder = False
    if not hasattr(policy_cfg, "train_expert_only"):
        policy_cfg.train_expert_only = False
    if not hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    expected_img_keys = _expected_image_keys(policy_cfg)
    image_key_map = _parse_image_key_map(args.image_key_map)

    if args.use_dummy_input:
        sample = _make_dummy_batch(policy_cfg, device, args.batch_size)
        preprocessor = None
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=args.checkpoint_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    else:
        ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
        delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
        dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        sample = next(iter(data_loader))

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(args.checkpoint_path, config=policy_cfg)
    policy = policy.to(device)
    policy.eval()

    profiler = None
    if args.profile_layers:
        profiler = LayerProfiler(device=device, leaf_only=args.profile_leaf_only)
        profiler.attach(policy.model, prefix="policy.model")

    nvtx_ranges = None
    if args.nvtx_layers:
        if nvtx is None:
            print("[NVTX] torch.cuda.nvtx not available; NVTX ranges disabled.")
        else:
            nvtx_ranges = NvtxLayerRanges(leaf_only=args.nvtx_leaf_only)
            nvtx_ranges.attach(policy.model, prefix="policy.model")

    block_profiler = None
    block_latencies: list[dict[str, float]] = []
    if args.profile_blocks:
        block_profiler = BlockLatencyProfiler(device, args.attn_regex, args.fc_regex)
        block_profiler.attach(policy.model, prefix="policy.model")

    prof_ctx = None
    if args.profile_trace:
        activities = [ProfilerActivity.CPU]
        if device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof_ctx = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=tensorboard_trace_handler(args.trace_dir),
        )

    if args.profile_iters <= 0:
        raise ValueError("--profile_iters must be > 0")
    if args.warmup_iters < 0:
        raise ValueError("--warmup_iters must be >= 0")

    timings = []

    with torch.inference_mode():
        batch = preprocessor(sample) if preprocessor is not None else sample
        batch = _apply_image_key_mapping(
            batch,
            expected_img_keys,
            image_key_map,
            fill_missing=args.fill_missing_images,
        )
        if prof_ctx is not None:
            prof_ctx.__enter__()

        for _ in range(args.warmup_iters):
            if args.fresh_batch_each_iter:
                if args.use_dummy_input:
                    sample = _make_dummy_batch(policy_cfg, device, args.batch_size)
                else:
                    sample = next(iter(data_loader))
                batch = preprocessor(sample) if preprocessor is not None else sample
                batch = _apply_image_key_mapping(
                    batch,
                    expected_img_keys,
                    image_key_map,
                    fill_missing=args.fill_missing_images,
                )
            _ = policy.select_action(batch)

        for iter_idx in range(args.profile_iters):
            if device in ("cuda", "mps"):
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif device == "mps" and torch.backends.mps.is_available():
                    torch.mps.synchronize()
            start = time.perf_counter()
            if block_profiler is not None:
                block_profiler.begin_iter()
            if args.fresh_batch_each_iter:
                if args.use_dummy_input:
                    sample = _make_dummy_batch(policy_cfg, device, args.batch_size)
                else:
                    sample = next(iter(data_loader))
                if args.time_preprocessor:
                    batch = preprocessor(sample) if preprocessor is not None else sample
                    batch = _apply_image_key_mapping(
                        batch,
                        expected_img_keys,
                        image_key_map,
                        fill_missing=args.fill_missing_images,
                    )
            if not args.fresh_batch_each_iter or not args.time_preprocessor:
                batch = preprocessor(sample) if preprocessor is not None else sample
                batch = _apply_image_key_mapping(
                    batch,
                    expected_img_keys,
                    image_key_map,
                    fill_missing=args.fill_missing_images,
                )
            action = policy.select_action(batch)
            if device in ("cuda", "mps"):
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif device == "mps" and torch.backends.mps.is_available():
                    torch.mps.synchronize()
            end = time.perf_counter()
            timings.append(end - start)
            if block_profiler is not None:
                totals_ms = block_profiler.end_iter()
                batch_ms = (end - start) * 1000.0
                other_ms = max(batch_ms - totals_ms["attn"] - totals_ms["fc"], 0.0)
                block_latencies.append(
                    {
                        "iter": float(iter_idx),
                        "batch_ms": batch_ms,
                        "attn_ms": totals_ms["attn"],
                        "fc_ms": totals_ms["fc"],
                        "other_ms": other_ms,
                        "batch_size": float(args.batch_size),
                    }
                )
            if args.print_iter_lat:
                print(f"[IterLatency] iter={iter_idx} batch_latency_s={timings[-1]:.6f}")

        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

        action = postprocessor(action)

    if profiler is not None:
        profiler.detach()
        profiler.report(top_n=args.profile_top_n)
    if nvtx_ranges is not None:
        nvtx_ranges.detach()
    if block_profiler is not None:
        block_profiler.detach()

    if timings:
        avg = sum(timings) / len(timings)
        print(
            "[SimpleInference] Policy select_action time (s): "
            f"avg={avg:.6f} min={min(timings):.6f} max={max(timings):.6f} iters={len(timings)}"
        )
        # Write per-iteration latency CSV and plot if requested
        import csv as _csv
        from pathlib import Path as _Path

        import matplotlib.pyplot as _plt

        if args.iter_csv is not None:
            out_csv = _Path(args.iter_csv) if args.iter_csv else _Path("prof/iter_latencies.csv")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow(["iter", "batch_latency_s", "per_sample_latency_s", "batch_size"])
                for i, t in enumerate(timings):
                    writer.writerow([i, f"{t:.6f}", f"{t / args.batch_size:.6f}", args.batch_size])
            print(f"[IterLatency] Wrote {out_csv}")

        if args.iter_plot is not None:
            out_png = _Path(args.iter_plot) if args.iter_plot else _Path("prof/iter_latencies.png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            xs = list(range(len(timings)))
            per_sample = [t / args.batch_size for t in timings]
            fig, ax1 = _plt.subplots(figsize=(6, 3))
            ax1.plot(xs, timings, marker="o", linewidth=1, label="batch_latency_s")
            ax1.set_xlabel("iter")
            ax1.set_ylabel("batch_latency_s")

            ax2 = ax1.twinx()
            ax2.plot(xs, per_sample, color="orange", linewidth=1, label="per_sample_latency_s")
            ax2.set_ylabel("per_sample_latency_s")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
            _plt.title(f"PI0 per-iteration latency (batch_size={args.batch_size})")
            _plt.tight_layout()
            _plt.savefig(out_png, dpi=150)
            print(f"[IterLatency] Wrote {out_png}")

        if args.profile_blocks and block_latencies:
            out_csv = _Path(args.block_lat_csv) if args.block_lat_csv else _Path("prof/block_latencies.csv")
            out_png = _Path(args.block_lat_plot) if args.block_lat_plot else _Path("prof/block_latencies.png")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow(["iter", "batch_ms", "attn_ms", "fc_ms", "other_ms", "batch_size"])
                for row in block_latencies:
                    writer.writerow(
                        [
                            int(row["iter"]),
                            f"{row['batch_ms']:.6f}",
                            f"{row['attn_ms']:.6f}",
                            f"{row['fc_ms']:.6f}",
                            f"{row['other_ms']:.6f}",
                            int(row["batch_size"]),
                        ]
                    )
            out_png.parent.mkdir(parents=True, exist_ok=True)
            xs = [int(r["iter"]) for r in block_latencies]
            attn = [r["attn_ms"] for r in block_latencies]
            fc = [r["fc_ms"] for r in block_latencies]
            other = [r["other_ms"] for r in block_latencies]
            fig, ax = _plt.subplots(figsize=(7.5, 4.5))
            ax.bar(xs, fc, label="FC", color="#4c78a8")
            ax.bar(xs, attn, bottom=fc, label="Attention", color="#f58518")
            ax.bar(xs, other, bottom=[f + a for f, a in zip(fc, attn)], label="Other", color="#9e9e9e")
            ax.set_xlabel("iter")
            ax.set_ylabel("latency per batch (ms)")
            ax.set_title(f"PI0 block latency (batch_size={args.batch_size})")
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            print(f"[BlockLatency] Wrote {out_csv}")
            print(f"[BlockLatency] Wrote {out_png}")
    print(f"[SimpleInference] Action shape: {tuple(action.shape)}")
    print(f"[SimpleInference] Action (unnormalized): {action}")

    if args.analyze_trace and args.profile_trace:
        _analyze_trace_dir(args.trace_dir)


if __name__ == "__main__":
    main()
