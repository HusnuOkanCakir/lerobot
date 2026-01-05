#!/usr/bin/env python

import argparse
import time
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def _infer_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


class LayerOpTracer:
    def __init__(self, leaf_only: bool = True) -> None:
        self.leaf_only = leaf_only
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._contexts: dict[int, torch.profiler.profile] = {}

    def _is_leaf(self, module: torch.nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _pre_hook(self, module: torch.nn.Module, _inputs, name: str) -> None:
        ctx = torch.profiler.record_function(f"layer::{name}")
        self._contexts[id(module)] = ctx
        ctx.__enter__()

    def _post_hook(self, module: torch.nn.Module, _inputs, _output) -> None:
        ctx = self._contexts.pop(id(module), None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    def attach(self, module: torch.nn.Module, prefix: str = "model") -> None:
        for name, child in module.named_modules():
            if name == "":
                continue
            if self.leaf_only and not self._is_leaf(child):
                continue
            full_name = f"{prefix}.{name}"
            self._handles.append(
                child.register_forward_pre_hook(
                    lambda mod, inputs, n=full_name: self._pre_hook(mod, inputs, n)
                )
            )
            self._handles.append(child.register_forward_hook(self._post_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._contexts.clear()


def _extract_correlation_id(args: dict) -> int | None:
    for key in ("correlation", "correlation_id", "External id", "external id", "external_id"):
        if key in args:
            try:
                return int(args[key])
            except (TypeError, ValueError):
                return None
    return None


def _analyze_layer_ops(
    trace_path: Path, top_n: int | None = None, output_path: Path | None = None
) -> None:
    with trace_path.open("r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    # Track active layer scopes per CPU thread.
    layer_stack: dict[tuple[int, int], list[str]] = defaultdict(list)
    cpu_op_layer: dict[int, str] = {}
    cpu_agg: dict[tuple[str, str], list[float]] = defaultdict(list)

    # Sort events by timestamp for stack tracking.
    events_sorted = sorted(
        [e for e in events if isinstance(e.get("ts"), (int, float))], key=lambda e: e["ts"]
    )

    for e in events_sorted:
        pid = e.get("pid")
        tid = e.get("tid")
        if pid is None or tid is None:
            continue
        key = (pid, tid)
        name = e.get("name", "")
        cat = (e.get("cat") or "").lower()
        ph = e.get("ph")

        if cat == "user_annotation" and isinstance(name, str) and name.startswith("layer::"):
            layer_name = name.removeprefix("layer::")
            if ph == "B":
                layer_stack[key].append(layer_name)
            elif ph == "E" and layer_stack[key]:
                layer_stack[key].pop()
            elif ph == "X":
                if layer_stack[key]:
                    layer_stack[key].append(layer_name)
                else:
                    layer_stack[key] = [layer_name]
            continue

        if cat == "cpu_op" and isinstance(e.get("dur"), (int, float)):
            layer_name = layer_stack[key][-1] if layer_stack[key] else "<no_layer>"
            op_name = name if isinstance(name, str) else "<unknown>"
            cpu_agg[(layer_name, op_name)].append(e["dur"])
            corr = _extract_correlation_id(e.get("args", {}))
            if corr is not None:
                cpu_op_layer[corr] = layer_name

    gpu_agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in events:
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy"):
            continue
        dur = e.get("dur")
        if not isinstance(dur, (int, float)):
            continue
        corr = _extract_correlation_id(e.get("args", {}))
        layer_name = cpu_op_layer.get(corr, "<no_layer>")
        op_name = e.get("name", "<unknown>")
        gpu_agg[(layer_name, op_name)].append(dur)

    lines = []

    def _print_ranked(title: str, agg: dict[tuple[str, str], list[float]]) -> None:
        rows = []
        for (layer_name, op_name), durs in agg.items():
            total = sum(durs) / 1e6
            rows.append((total, layer_name, op_name, len(durs)))
        rows.sort(reverse=True, key=lambda r: r[0])
        lines.append(f"\n[LayerOpAnalysis] {title}")
        lines.append(f"{'total_s':>10} {'count':>7}  layer  :: op")
        if top_n is None:
            limit_rows = rows
        else:
            limit_rows = rows[:top_n]
        for total, layer_name, op_name, count in limit_rows:
            lines.append(f"{total:10.6f} {count:7d}  {layer_name} :: {op_name}")

    _print_ranked("CPU ops by layer", cpu_agg)
    _print_ranked("GPU ops by layer", gpu_agg)
    output = "\n".join(lines)
    if output_path is None:
        print(output)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single SmolVLA inference on a dataset sample.")
    parser.add_argument(
        "--checkpoint_path",
        default="outputs/train/my_smolvla/checkpoints/last/pretrained_model",
        help="Path to the pretrained_model directory from a checkpoint.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        default="lerobot/svla_so101_pickplace",
        help="Dataset repo id used during training (must be available in local cache).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--profile_layers",
        action="store_true",
        help="Enable per-layer timing using forward hooks.",
    )
    parser.add_argument(
        "--profile_leaf_only",
        action="store_true",
        help="Only profile leaf modules.",
    )
    parser.add_argument(
        "--profile_top_n",
        type=int,
        default=50,
        help="Number of layers to print in the timing report.",
    )
    parser.add_argument(
        "--trace_layer_ops",
        action="store_true",
        help="Emit per-layer trace ranges for op-to-layer attribution.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=0,
        help="Number of warmup iterations before profiling.",
    )
    parser.add_argument(
        "--profile_iters",
        type=int,
        default=1,
        help="Number of iterations to time/profile.",
    )
    parser.add_argument(
        "--profile_trace",
        action="store_true",
        help="Enable torch.profiler trace export for TensorBoard.",
    )
    parser.add_argument(
        "--trace_dir",
        default="prof/trace_inference",
        help="Output directory for TensorBoard trace files.",
    )
    parser.add_argument(
        "--analyze_trace",
        action="store_true",
        help="Analyze the latest trace in --trace_dir for compute vs memcpy time.",
    )
    parser.add_argument(
        "--analyze_layer_ops",
        action="store_true",
        help="Analyze trace to attribute CPU/GPU ops to layers.",
    )
    parser.add_argument(
        "--layer_ops_top_n",
        type=int,
        default=0,
        help="Number of layer-op rows to print per device (0 prints all).",
    )
    parser.add_argument(
        "--layer_ops_output",
        type=str,
        default=None,
        help="Write layer-op report to this file path instead of stdout.",
    )
    args = parser.parse_args()

    device = _infer_device(args.device)
    print(f"[SimpleInference] Using device: {device}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint_path)
    policy_cfg.device = device

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
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
    layer_op_tracer = None
    if args.profile_layers:
        profiler = LayerProfiler(device=device, leaf_only=args.profile_leaf_only)
        profiler.attach(policy.model, prefix="policy.model")
    if args.trace_layer_ops:
        layer_op_tracer = LayerOpTracer(leaf_only=args.profile_leaf_only)
        layer_op_tracer.attach(policy.model, prefix="policy.model")

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
        batch = preprocessor(sample)
        if prof_ctx is not None:
            prof_ctx.__enter__()

        for _ in range(args.warmup_iters):
            _ = policy.select_action(batch)

        for _ in range(args.profile_iters):
            if device in ("cuda", "mps"):
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif device == "mps" and torch.backends.mps.is_available():
                    torch.mps.synchronize()
            start = time.perf_counter()
            # action = policy.select_action(batch)
            action = policy.predict_action_chunk(batch)[:, 0]

            if device in ("cuda", "mps"):
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif device == "mps" and torch.backends.mps.is_available():
                    torch.mps.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

        action = postprocessor(action)

    if profiler is not None:
        profiler.detach()
        profiler.report(top_n=args.profile_top_n)
    if layer_op_tracer is not None:
        layer_op_tracer.detach()

    if timings:
        avg = sum(timings) / len(timings)
        print(
            "[SimpleInference] Policy select_action time (s): "
            f"avg={avg:.6f} min={min(timings):.6f} max={max(timings):.6f} iters={len(timings)}"
        )
    print(f"[SimpleInference] Action shape: {tuple(action.shape)}")
    print(f"[SimpleInference] Action (unnormalized): {action}")

    if args.analyze_trace and args.profile_trace:
        _analyze_trace_dir(args.trace_dir)
    if args.analyze_layer_ops and args.profile_trace:
        trace_dir = Path(args.trace_dir)
        traces = sorted(trace_dir.rglob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
        if traces:
            top_n = None if args.layer_ops_top_n == 0 else args.layer_ops_top_n
            output_path = Path(args.layer_ops_output) if args.layer_ops_output else None
            _analyze_layer_ops(traces[-1], top_n=top_n, output_path=output_path)
        else:
            print(f"[LayerOpAnalysis] no trace files found in: {args.trace_dir}")


if __name__ == "__main__":
    main()
