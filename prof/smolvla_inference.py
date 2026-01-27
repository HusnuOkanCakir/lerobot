#!/usr/bin/env python

import argparse
import time
from collections import defaultdict
import json
import csv
from pathlib import Path

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


class NvtxLayerRanges:
    def __init__(self, leaf_only: bool = True) -> None:
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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (DataLoader batch size).",
    )
    parser.add_argument(
        "--batch_sizes",
        default="",
        help="Comma-separated batch sizes for a sweep (e.g., 1,2,4,8).",
    )
    parser.add_argument(
        "--sweep_csv",
        default="",
        help="Output CSV for batch-size sweep results (default: prof/batch_sweep.csv).",
    )
    parser.add_argument(
        "--sweep_plot",
        default="",
        help="Output PNG for batch-size sweep plot (default: prof/batch_sweep.png).",
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
        "--fresh_batch_each_iter",
        action="store_true",
        help="Fetch a new dataset sample and run preprocessor each iteration.",
    )
    parser.add_argument(
        "--time_preprocessor",
        action="store_true",
        help="Include preprocessor (and H2D transfer) time in per-iteration timing.",
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
    args = parser.parse_args()

    device = _infer_device(args.device)
    print(f"[SimpleInference] Using device: {device}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint_path)
    policy_cfg.device = device

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps)

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.batch_sizes:
        batch_sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
        if any(bs <= 0 for bs in batch_sizes):
            raise ValueError("--batch_sizes must be positive integers.")
        if not batch_sizes:
            raise ValueError("--batch_sizes must contain at least one value.")
    else:
        batch_sizes = []

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

    def _run_profile(batch_size: int, print_iter: bool) -> tuple[list[float], torch.Tensor]:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        sample = next(iter(data_loader))
        timings: list[float] = []
        data_iter = iter(data_loader)

        def _next_sample():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                return next(data_iter)

        with torch.inference_mode():
            batch = preprocessor(sample)
            if prof_ctx is not None:
                prof_ctx.__enter__()

            for _ in range(args.warmup_iters):
                if args.fresh_batch_each_iter:
                    sample = _next_sample()
                    batch = preprocessor(sample)
                _ = policy.select_action(batch)

            for i in range(args.profile_iters):
                if args.fresh_batch_each_iter and not args.time_preprocessor:
                    sample = _next_sample()
                    batch = preprocessor(sample)
                if device in ("cuda", "mps"):
                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif device == "mps" and torch.backends.mps.is_available():
                        torch.mps.synchronize()
                start = time.perf_counter()
                if args.fresh_batch_each_iter and args.time_preprocessor:
                    sample = _next_sample()
                    batch = preprocessor(sample)
                action = policy.select_action(batch)
                if device in ("cuda", "mps"):
                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif device == "mps" and torch.backends.mps.is_available():
                        torch.mps.synchronize()
                end = time.perf_counter()
                elapsed = end - start
                timings.append(elapsed)
                if print_iter:
                    per_sample = elapsed / batch_size
                    print(
                        f"[SimpleInference] iter={i} batch_latency_s={elapsed:.6f} "
                        f"per_sample_latency_s={per_sample:.6f}"
                    )

            if prof_ctx is not None:
                prof_ctx.__exit__(None, None, None)

            action = postprocessor(action)
        return timings, action

    def _write_iter_csv(timings: list[float], batch_size: int) -> None:
        if not timings:
            return
        out_path = Path(args.iter_csv) if args.iter_csv else Path("prof/iter_latencies.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "batch_latency_s", "per_sample_latency_s", "batch_size"])
            for i, t in enumerate(timings):
                writer.writerow([i, f"{t:.6f}", f"{t / batch_size:.6f}", batch_size])
        print(f"[SimpleInference] Wrote iter CSV: {out_path}")

    def _write_iter_plot(timings: list[float], batch_size: int) -> None:
        if not timings:
            return
        out_path = Path(args.iter_plot) if args.iter_plot else Path("prof/iter_latencies.png")
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[SimpleInference] Iter plot skipped (matplotlib unavailable): {exc}")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        xs = list(range(len(timings)))
        batch_vals = timings
        per_sample_vals = [t / batch_size for t in timings]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(xs, batch_vals, marker="o", label="batch_latency_s")
        ax1.set_xlabel("iter")
        ax1.set_ylabel("batch_latency_s")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(xs, per_sample_vals, color="tab:orange", label="per_sample_latency_s")
        ax2.set_ylabel("per_sample_latency_s")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[SimpleInference] Wrote iter plot: {out_path}")

    def _summarize(timings: list[float], batch_size: int) -> None:
        if timings:
            avg = sum(timings) / len(timings)
            avg_per_sample = avg / batch_size
            print(
                "[SimpleInference] Policy select_action time (s): "
                f"avg_batch={avg:.6f} avg_per_sample={avg_per_sample:.6f} "
                f"min_batch={min(timings):.6f} max_batch={max(timings):.6f} iters={len(timings)}"
            )

    def _write_sweep_csv(rows: list[dict]) -> None:
        if not rows:
            return
        out_path = Path(args.sweep_csv) if args.sweep_csv else Path("prof/batch_sweep.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SimpleInference] Wrote sweep CSV: {out_path}")

    def _write_sweep_plot(rows: list[dict]) -> None:
        out_path = Path(args.sweep_plot) if args.sweep_plot else Path("prof/batch_sweep.png")
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[SimpleInference] Plot skipped (matplotlib unavailable): {exc}")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        xs = [r["batch_size"] for r in rows]
        avg_batch = [r["avg_batch_s"] for r in rows]
        avg_per_sample = [r["avg_per_sample_s"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(xs, avg_batch, marker="o", label="avg_batch_s")
        ax1.set_xlabel("batch_size")
        ax1.set_ylabel("avg_batch_s")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(xs, avg_per_sample, marker="s", color="tab:orange", label="avg_per_sample_s")
        ax2.set_ylabel("avg_per_sample_s")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[SimpleInference] Wrote sweep plot: {out_path}")

    if batch_sizes:
        rows = []
        for bs in batch_sizes:
            timings, _ = _run_profile(bs, print_iter=False)
            avg = sum(timings) / len(timings)
            rows.append(
                {
                    "batch_size": bs,
                    "avg_batch_s": avg,
                    "avg_per_sample_s": avg / bs,
                    "min_batch_s": min(timings),
                    "max_batch_s": max(timings),
                    "iters": len(timings),
                    "warmup_iters": args.warmup_iters,
                    "profile_iters": args.profile_iters,
                    "fresh_batch_each_iter": args.fresh_batch_each_iter,
                    "time_preprocessor": args.time_preprocessor,
                }
            )
        _write_sweep_csv(rows)
        _write_sweep_plot(rows)
        return

    timings, action = _run_profile(args.batch_size, print_iter=args.print_iter_lat)

    if profiler is not None:
        profiler.detach()
        profiler.report(top_n=args.profile_top_n)
    if nvtx_ranges is not None:
        nvtx_ranges.detach()

    _summarize(timings, args.batch_size)
    _write_iter_csv(timings, args.batch_size)
    _write_iter_plot(timings, args.batch_size)
    print(f"[SimpleInference] Action shape: {tuple(action.shape)}")
    print(f"[SimpleInference] Action (unnormalized): {action}")

    if args.analyze_trace and args.profile_trace:
        _analyze_trace_dir(args.trace_dir)


if __name__ == "__main__":
    main()
