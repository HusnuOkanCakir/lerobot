#!/usr/bin/env python

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _detect_encoding(csv_path: Path) -> str:
    with csv_path.open("rb") as f:
        start = f.read(4)
    if start.startswith(b"\xff\xfe") or start.startswith(b"\xfe\xff"):
        return "utf-16"
    return "utf-8"


def _read_header(csv_path: Path, encoding: str) -> list[str]:
    with csv_path.open(newline="", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        return next(reader)


def _to_float(value: str) -> float:
    if value is None:
        return math.nan
    s = str(value).strip().strip('"')
    if not s or s.lower() in {"nan", "na", "n/a", "inf", "-inf"}:
        return math.nan
    s = s.replace(" ", "")
    if "," in s:
        if "." in s and s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        elif "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return math.nan


def _unit_scale(unit: str) -> float:
    if unit is None:
        return 1.0
    u = str(unit).strip().lower()
    if u in {"hz"}:
        return 1.0
    if u in {"khz"}:
        return 1e3
    if u in {"mhz"}:
        return 1e6
    if u in {"ghz"}:
        return 1e9
    if u in {"byte/s", "bytes/s"}:
        return 1.0
    if u in {"kbyte/s", "kbytes/s"}:
        return 1e3
    if u in {"mbyte/s", "mbytes/s"}:
        return 1e6
    if u in {"gbyte/s", "gbytes/s"}:
        return 1e9
    if u in {"tbyte/s", "tbytes/s"}:
        return 1e12
    if u in {"pbyte/s", "pbytes/s"}:
        return 1e15
    if u in {"byte/cycle", "bytes/cycle"}:
        return 1.0
    if u in {"kbyte/cycle", "kbytes/cycle"}:
        return 1e3
    if u in {"mbyte/cycle", "mbytes/cycle"}:
        return 1e6
    if u in {"gbyte/cycle", "gbytes/cycle"}:
        return 1e9
    if u in {"tbyte/cycle", "tbytes/cycle"}:
        return 1e12
    if u in {"pbyte/cycle", "pbytes/cycle"}:
        return 1e15
    if u in {"ns"}:
        return 1e-9
    if u in {"us"}:
        return 1e-6
    if u in {"ms"}:
        return 1e-3
    if u in {"s"}:
        return 1.0
    return 1.0


def _pick_columns(header: list[str], candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in header]


def _display_stem_name(stem: str) -> str:
    return re.sub(r"^\d{8}_\d{6}_", "", stem)


def _export_rep(rep_path: Path, ncu_bin: str) -> Path:
    out_dir = rep_path.parent / rep_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (rep_path.stem + ".csv")
    cmd = [ncu_bin, "--import", str(rep_path), "--page", "raw", "--csv"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        subprocess.run(cmd, stdout=f, check=True)
    return csv_path


def _load_csv(csv_path: Path):
    encoding = _detect_encoding(csv_path)
    header = _read_header(csv_path, encoding)
    kernel_name_col = "Kernel Name" if "Kernel Name" in header else None
    nvtx_cols = [
        "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
        "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
    ]
    nvtx_cols = [c for c in nvtx_cols if c in header]

    per_cycle_cols = _pick_columns(
        header,
        [
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
            "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",
        ],
    )
    if len(per_cycle_cols) < 3:
        raise RuntimeError("Missing FP32 per-cycle SASS columns required for achieved work.")

    bytes_cols = _pick_columns(
        header,
        [
            "dram__bytes.sum.per_second",
            "dram__bytes.sum",
            "dram__bytes.avg",
        ],
    )
    if not bytes_cols:
        raise RuntimeError("No DRAM bytes columns found in the CSV.")
    bytes_col = bytes_cols[0]
    bytes_is_rate = "per_second" in bytes_col

    cycles_cols = _pick_columns(
        header,
        [
            "smsp__cycles_elapsed.avg.per_second",
            "sm__cycles_elapsed.avg.per_second",
        ],
    )
    if not cycles_cols:
        raise RuntimeError("No SM(SP) cycles per second column found in the CSV.")
    cycles_per_sec_col = cycles_cols[0]

    peak_bw_col = "dram__bytes.sum.peak_sustained" if "dram__bytes.sum.peak_sustained" in header else None
    peak_work_col = "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2"
    peak_work_col = peak_work_col if peak_work_col in header else None
    peak_cycles_col = "sm__cycles_elapsed.avg.per_second" if "sm__cycles_elapsed.avg.per_second" in header else None
    peak_traffic_cycles_col = (
        "dram__cycles_elapsed.avg.per_second" if "dram__cycles_elapsed.avg.per_second" in header else None
    )

    usecols = set(per_cycle_cols + [bytes_col, cycles_per_sec_col, "gpu__time_duration.sum"])
    bw_pct_col = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    if bw_pct_col in header:
        usecols.add(bw_pct_col)
    if kernel_name_col:
        usecols.add(kernel_name_col)
    for col in nvtx_cols:
        usecols.add(col)
    if peak_bw_col:
        usecols.add(peak_bw_col)
    if peak_work_col:
        usecols.add(peak_work_col)
    if peak_cycles_col:
        usecols.add(peak_cycles_col)
    if peak_traffic_cycles_col:
        usecols.add(peak_traffic_cycles_col)

    df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding, engine="python")
    units_row = df.iloc[0] if len(df) > 0 else None
    has_units = False
    if units_row is not None:
        for col in [bytes_col, cycles_per_sec_col, "gpu__time_duration.sum"]:
            if col in df.columns:
                val = units_row.get(col)
                if isinstance(val, str) and any(ch.isalpha() for ch in val):
                    has_units = True
                    break
    if has_units:
        df = df.iloc[1:].copy()

    def _scale_col(col: str) -> float:
        if not has_units or units_row is None:
            return 1.0
        return _unit_scale(units_row.get(col))

    for col in per_cycle_cols:
        df[col] = df[col].map(_to_float)
    df[bytes_col] = df[bytes_col].map(_to_float) * _scale_col(bytes_col)
    df[cycles_per_sec_col] = df[cycles_per_sec_col].map(_to_float) * _scale_col(cycles_per_sec_col)
    if peak_bw_col:
        df[peak_bw_col] = df[peak_bw_col].map(_to_float) * _scale_col(peak_bw_col)
    if peak_work_col:
        df[peak_work_col] = df[peak_work_col].map(_to_float)
    if peak_cycles_col:
        df[peak_cycles_col] = df[peak_cycles_col].map(_to_float) * _scale_col(peak_cycles_col)
    if peak_traffic_cycles_col:
        df[peak_traffic_cycles_col] = df[peak_traffic_cycles_col].map(_to_float) * _scale_col(peak_traffic_cycles_col)
    if bw_pct_col in df.columns:
        df[bw_pct_col] = df[bw_pct_col].map(_to_float)

    ops_per_cycle = (
        df.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed", 0.0)
        + df.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed", 0.0)
        + df.get("derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2", 0.0)
    )
    cycles_per_sec = df[cycles_per_sec_col]
    df["ops_per_sec"] = ops_per_cycle * cycles_per_sec
    df["bytes_per_sec"] = df[bytes_col]
    time_col = "gpu__time_duration.sum"
    time_scale = _scale_col(time_col) if time_col in df.columns else 1.0
    df["time_s"] = df.get(time_col, pd.Series([math.nan] * len(df))).map(_to_float) * time_scale
    if not bytes_is_rate:
        df["bytes_per_sec"] = df["bytes_per_sec"] / df["time_s"].replace(0, math.nan)
    df = df[(df["ops_per_sec"] > 0) & (df["bytes_per_sec"] > 0)].copy()
    if df.empty:
        raise RuntimeError("No valid rows found after filtering.")

    df["ai"] = df["ops_per_sec"] / df["bytes_per_sec"]
    df["perf"] = df["ops_per_sec"]

    def _row_name(row) -> str:
        for col in nvtx_cols:
            val = row.get(col)
            if val is None:
                continue
            sval = str(val).strip()
            if sval and sval.lower() not in {"nan", "none"}:
                return sval
        if kernel_name_col:
            return str(row.get(kernel_name_col))
        return "kernel"

    df["row_name"] = df.apply(_row_name, axis=1)

    peaks = {
        "peak_bw_col": peak_bw_col,
        "peak_work_col": peak_work_col,
        "peak_cycles_col": peak_cycles_col,
        "peak_traffic_cycles_col": peak_traffic_cycles_col,
    }
    return df, peaks


def _aggregate_category(df: pd.DataFrame, mask: pd.Series):
    sub = df[mask].copy()
    if sub.empty:
        return None
    sub = sub[sub["time_s"].notna()].copy()
    if sub.empty:
        return None
    ops = (sub["ops_per_sec"] * sub["time_s"]).sum()
    bytes_ = (sub["bytes_per_sec"] * sub["time_s"]).sum()
    time = sub["time_s"].sum()
    if time <= 0 or bytes_ <= 0:
        return None
    ai = ops / bytes_
    perf = ops / time
    bw_pct = None
    if "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" in sub.columns:
        weights = sub["time_s"].fillna(0)
        if weights.sum() > 0:
            bw_pct = (sub["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"] * weights).sum() / weights.sum()
        else:
            bw_pct = sub["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"].mean()
    return {"ai": ai, "perf": perf, "time": time, "bw_pct": bw_pct}


def _bounds_rank_map(bounds_path: Path) -> dict[str, int]:
    if not bounds_path.exists():
        return {}
    bounds_df = pd.read_csv(bounds_path)
    if "name" not in bounds_df.columns:
        return {}
    names = bounds_df["name"].astype(str).tolist()
    return {name: idx for idx, name in enumerate(names)}


def _normalize_name(name: str) -> str:
    s = str(name).strip().strip('"')
    s = re.sub(r'^\d+\s+', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def _row_rank(row_name: str, bounds_names: list[str]) -> int | None:
    norm_row = _normalize_name(row_name)
    norm_bounds = [_normalize_name(n) for n in bounds_names]
    for idx, name in enumerate(norm_bounds):
        if norm_row == name:
            return idx
    for idx, name in enumerate(norm_bounds):
        if norm_row in name or name in norm_row:
            return idx
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multiple NCU CSVs on one roofline.")
    parser.add_argument("--rep", action="append", required=True, help="Path to .ncu-rep or exported .csv.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--ncu", default="ncu", help="Path to ncu executable.")
    parser.add_argument("--use-peak-formula", action="store_true", help="Use Nsight peak formula columns.")
    parser.add_argument("--peak-flops", type=float, default=0.0, help="Override peak compute in ops/s.")
    parser.add_argument("--peak-bw", type=float, default=0.0, help="Override peak memory bandwidth in bytes/s.")
    parser.add_argument("--perf-scale", type=float, default=1e11, help="Scale factor for Y-axis label.")
    parser.add_argument("--scale-perf", action="store_true", help="Divide performance by --perf-scale.")
    parser.add_argument("--no-scale-perf", action="store_true", help="Do not scale performance values.")
    parser.add_argument(
        "--model",
        choices=["smolvla", "pi0"],
        default="smolvla",
        help="Select default attention/FC regex presets.",
    )
    parser.add_argument(
        "--out-points",
        default="",
        help="Optional CSV output with plotted points.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Color points by bounds rank heat (single input only).",
    )
    parser.add_argument(
        "--dot-size-time",
        action="store_true",
        help="Size dots by time_s instead of memory BW percent.",
    )
    parser.add_argument(
        "--aggregate-attn-fc",
        action="store_true",
        help="Aggregate attention and FC/MLP kernels into two points per input file.",
    )
    parser.add_argument(
        "--aggregate-stage-attn-fc",
        action="store_true",
        help="Aggregate into four points per input file: sum/gen x attn/fc (requires stage-tagged NVTX names).",
    )
    parser.add_argument(
        "--attn-sub-split",
        action="store_true",
        help="When used with --aggregate-stage-attn-fc, split attention into attn_gemm vs attn_other.",
    )
    parser.add_argument(
        "--attn-regex",
        action="append",
        default=None,
        help="Regex for attention block matching (repeatable).",
    )
    parser.add_argument(
        "--fc-regex",
        action="append",
        default=None,
        help="Regex for FC/MLP block matching (repeatable).",
    )
    parser.add_argument(
        "--sum-stage-regex",
        action="append",
        default=None,
        help="Regex for PI0 summarization-stage matching in NVTX row names (repeatable).",
    )
    parser.add_argument(
        "--gen-stage-regex",
        action="append",
        default=None,
        help="Regex for PI0 generation-stage matching in NVTX row names (repeatable).",
    )
    args = parser.parse_args()

    if args.attn_regex is None:
        args.attn_regex = [r"BLOCK\.attn\."]
    if args.fc_regex is None:
        args.fc_regex = [r"BLOCK\.fc\."]
    if args.sum_stage_regex is None:
        args.sum_stage_regex = [r"\.stage\.sum(?=[:.]|$)", r"PI0\.stage\.sum(?=[:.]|$)"]
    if args.gen_stage_regex is None:
        args.gen_stage_regex = [r"\.stage\.gen(?=[:.]|$)", r"PI0\.stage\.gen(?=[:.]|$)"]

    ncu_bin = shutil.which(args.ncu) if args.ncu == "ncu" else args.ncu
    if not ncu_bin:
        raise FileNotFoundError("ncu not found on PATH. Pass --ncu /path/to/ncu.")

    series = []
    peak_bw = args.peak_bw
    peak_flops = args.peak_flops

    for rep in args.rep:
        path = Path(rep)
        if path.suffix.lower() == ".ncu-rep":
            csv_path = _export_rep(path, ncu_bin)
        else:
            csv_path = path
        df, peaks = _load_csv(csv_path)
        series.append((_display_stem_name(path.stem), df, peaks))

        if args.use_peak_formula:
            if peaks["peak_work_col"] and peaks["peak_cycles_col"]:
                peak_flops = max(peak_flops, (df[peaks["peak_work_col"]] * df[peaks["peak_cycles_col"]]).max())
            if peaks["peak_bw_col"] and peaks["peak_traffic_cycles_col"]:
                peak_bw = max(peak_bw, (df[peaks["peak_bw_col"]] * df[peaks["peak_traffic_cycles_col"]]).max())
        elif peaks["peak_bw_col"] and peak_bw <= 0:
            peak_bw = max(peak_bw, df[peaks["peak_bw_col"]].max())

    perf_scale = args.perf_scale if args.perf_scale > 0 else 1.0
    if args.no_scale_perf:
        plot_scale = 1.0
    elif args.scale_perf or perf_scale != 1.0:
        plot_scale = perf_scale
    else:
        plot_scale = 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("tab10")
    size_by_bw = False
    points_rows = []
    if args.aggregate_stage_attn_fc:
        attn_patterns = [re.compile(p) for p in args.attn_regex]
        fc_patterns = [re.compile(p) for p in args.fc_regex]
        sum_stage_patterns = [re.compile(p) for p in args.sum_stage_regex]
        gen_stage_patterns = [re.compile(p) for p in args.gen_stage_regex]
        group_styles = {
            "sum_attn": {"color": "#f58518", "marker": "^", "label": "sum attn"},
            "sum_fc": {"color": "#4c78a8", "marker": "o", "label": "sum fc"},
            "gen_attn": {"color": "#e45756", "marker": "s", "label": "gen attn"},
            "gen_fc": {"color": "#72b7b2", "marker": "D", "label": "gen fc"},
            "sum_attn_gemm": {"color": "#f58518", "marker": "^", "label": "sum attn_gemm"},
            "sum_attn_other": {"color": "#ffbf79", "marker": "v", "label": "sum attn_other"},
            "gen_attn_gemm": {"color": "#e45756", "marker": "s", "label": "gen attn_gemm"},
            "gen_attn_other": {"color": "#ff9da6", "marker": "P", "label": "gen attn_other"},
        }
        for _idx, (label, df, _peaks) in enumerate(series):
            names = df["row_name"].astype(str)
            attn_mask = names.apply(lambda s: any(p.search(s) for p in attn_patterns))
            fc_mask = names.apply(lambda s: any(p.search(s) for p in fc_patterns))
            sum_mask = names.apply(lambda s: any(p.search(s) for p in sum_stage_patterns))
            gen_mask = names.apply(lambda s: any(p.search(s) for p in gen_stage_patterns))
            kernel_names = df["Kernel Name"].astype(str) if "Kernel Name" in df.columns else None
            if args.attn_sub_split and kernel_names is None:
                print(f"[WARN] {label}: --attn-sub-split requested but 'Kernel Name' column is unavailable; ignoring split.")
            attn_gemm_mask = (
                kernel_names.str.contains(r"gemm", case=False, regex=True, na=False)
                if (args.attn_sub_split and kernel_names is not None)
                else None
            )

            matched_any = False
            if args.attn_sub_split and attn_gemm_mask is not None:
                groups = [
                    ("sum_attn_gemm", sum_mask & attn_mask & attn_gemm_mask),
                    ("sum_attn_other", sum_mask & attn_mask & ~attn_gemm_mask),
                    ("sum_fc", sum_mask & fc_mask),
                    ("gen_attn_gemm", gen_mask & attn_mask & attn_gemm_mask),
                    ("gen_attn_other", gen_mask & attn_mask & ~attn_gemm_mask),
                    ("gen_fc", gen_mask & fc_mask),
                ]
            else:
                groups = [
                    ("sum_attn", sum_mask & attn_mask),
                    ("sum_fc", sum_mask & fc_mask),
                    ("gen_attn", gen_mask & attn_mask),
                    ("gen_fc", gen_mask & fc_mask),
                ]
            for group_key, mask in groups:
                agg = _aggregate_category(df, mask)
                if not agg:
                    continue
                matched_any = True
                style = group_styles[group_key]
                parts = group_key.split("_")
                stage = parts[0]
                if len(parts) == 2:
                    block_type = parts[1]
                    attn_subtype = ""
                else:
                    block_type = "_".join(parts[1:3])
                    attn_subtype = parts[2] if parts[1] == "attn" else ""
                points_rows.append(
                    {
                        "label": label,
                        "group": group_key,
                        "stage": stage,
                        "block_type": block_type,
                        "attn_subtype": attn_subtype,
                        "ai": agg["ai"],
                        "perf": agg["perf"],
                        "time_s": agg["time"],
                        "mem_bw_pct": agg["bw_pct"],
                    }
                )
                ax.scatter(
                    [agg["ai"]],
                    [agg["perf"] / plot_scale],
                    s=140,
                    c=[style["color"]],
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{label} {style['label']}",
                )
            if not matched_any:
                print(
                    f"[WARN] {label}: no stage-tagged attn/fc kernels matched. "
                    "Expected NVTX names containing '.stage.sum' / '.stage.gen'."
                )
        if args.attn_sub_split:
            size_label = "Point size = fixed (aggregated sum/gen x (attn_gemm|attn_other|fc))"
        else:
            size_label = "Point size = fixed (aggregated sum/gen x attn/fc)"
    elif args.aggregate_attn_fc:
        attn_patterns = [re.compile(p) for p in args.attn_regex]
        fc_patterns = [re.compile(p) for p in args.fc_regex]
        for idx, (label, df, _peaks) in enumerate(series):
            names = df["row_name"].astype(str)
            attn_mask = names.apply(lambda s: any(p.search(s) for p in attn_patterns))
            fc_mask = names.apply(lambda s: any(p.search(s) for p in fc_patterns))
            attn = _aggregate_category(df, attn_mask)
            fc = _aggregate_category(df, fc_mask)
            if attn:
                points_rows.append(
                    {
                        "label": label,
                        "group": "attn",
                        "ai": attn["ai"],
                        "perf": attn["perf"],
                        "time_s": attn["time"],
                        "mem_bw_pct": attn["bw_pct"],
                    }
                )
                ax.scatter(
                    [attn["ai"]],
                    [attn["perf"] / plot_scale],
                    s=120,
                    c=[colors(idx % 10)],
                    marker="^",
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{label} attn",
                )
            if fc:
                points_rows.append(
                    {
                        "label": label,
                        "group": "fc",
                        "ai": fc["ai"],
                        "perf": fc["perf"],
                        "time_s": fc["time"],
                        "mem_bw_pct": fc["bw_pct"],
                    }
                )
                ax.scatter(
                    [fc["ai"]],
                    [fc["perf"] / plot_scale],
                    s=120,
                    c=[colors(idx % 10)],
                    marker="o",
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{label} fc",
                )
        size_label = "Point size = fixed (aggregated)"
    else:
        attn_patterns = [re.compile(p) for p in args.attn_regex]
        fc_patterns = [re.compile(p) for p in args.fc_regex]
        heat_colors = None
        if args.heatmap and len(series) == 1:
            rep_path = Path(args.rep[0])
            if rep_path.suffix == ".ncu-rep":
                bounds_path = rep_path.parent / rep_path.stem / f"{rep_path.stem}_bounds.csv"
            else:
                bounds_path = rep_path.with_name(f"{rep_path.stem}_bounds.csv")
            bounds_map = _bounds_rank_map(bounds_path)
            bounds_names = list(bounds_map.keys())
            if bounds_names:
                ranks = []
                for name in series[0][1]["row_name"].astype(str):
                    rank = _row_rank(name, bounds_names)
                    ranks.append(rank if rank is not None else len(bounds_names))
                ranks = pd.Series(ranks)
                denom = max(len(bounds_names) - 1, 1)
                heat = 1.0 - (ranks / denom)
                cmap = plt.get_cmap("hot")
                heat_colors = cmap(heat.clip(0.0, 1.0))
        for idx, (label, df, _peaks) in enumerate(series):
            names = df["row_name"].astype(str)
            attn_mask = names.apply(lambda s: any(p.search(s) for p in attn_patterns))
            fc_mask = names.apply(lambda s: any(p.search(s) for p in fc_patterns))
            other_mask = ~(attn_mask | fc_mask)
            if not args.dot_size_time and "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" in df.columns:
                bw_pct = df["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"].fillna(0)
                sizes = (bw_pct / max(bw_pct.max(), 1e-12) * 80.0) + 10.0
                size_by_bw = True
            else:
                time_s = df["time_s"].fillna(0)
                sizes = (time_s / max(time_s.max(), 1e-12) * 80.0) + 10.0
                bw_pct = pd.Series([math.nan] * len(df))
            for ai, perf, t, bw in zip(df["ai"], df["perf"], df["time_s"], bw_pct):
                points_rows.append(
                    {
                        "label": label,
                        "group": "kernel",
                        "ai": ai,
                        "perf": perf,
                        "time_s": t,
                        "mem_bw_pct": bw,
                    }
                )
            def _plot_subset(mask, marker, suffix):
                if not mask.any():
                    return
                ax.scatter(
                    df.loc[mask, "ai"],
                    df.loc[mask, "perf"] / plot_scale,
                    s=pd.Series(sizes).loc[mask],
                    c=(heat_colors[mask] if heat_colors is not None else [colors(idx % 10)]),
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.3,
                    marker=marker,
                    label=f"{label} {suffix}",
                )

            _plot_subset(attn_mask, "^", "attn")
            _plot_subset(fc_mask, "o", "fc")
            _plot_subset(other_mask, "x", "other")
        size_label = "Point size = memory BW % of peak" if size_by_bw else "Point size = duration (s)"

        if heat_colors is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("hot"), norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Bounds rank (higher = more memory bounded)")

    if peak_bw and peak_flops:
        ridge_x = peak_flops / peak_bw
        all_ai = pd.concat([d["ai"] for _, d, _ in series], ignore_index=True)
        xmin, xmax = all_ai.min(), all_ai.max()
        xmin = min(xmin / 10.0, ridge_x / 100.0)
        xmax = max(xmax * 10.0, ridge_x * 100.0)
        xs = [xmin, xmax]
        mem_line = [(peak_bw * x) / plot_scale for x in xs]
        comp_line = [peak_flops / plot_scale, peak_flops / plot_scale]
        ax.plot(xs, mem_line, "--", color="orange", linewidth=1, label="Memory roof")
        ax.plot(xs, comp_line, "--", color="red", linewidth=1, label="Compute roof")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(
            min(min(d["perf"].min() for _, d, _ in series) / plot_scale / 10.0, (peak_flops / plot_scale) / 100.0),
            (peak_flops / plot_scale) * 10.0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("HW Arithmetic Intensity [FLOP/byte]")
    ax.set_ylabel(f"HW Performance [FLOP/s] (1 = {perf_scale:.0e})")
    ax.set_title("Multi-run Roofline")
    ax.text(
        0.02,
        0.02,
        size_label,
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
        va="bottom",
    )
    ax.legend(loc="best")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")

    if args.out_points:
        out_points = Path(args.out_points)
        out_points.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(points_rows).to_csv(out_points, index=False)
        print(f"Wrote {out_points}")


if __name__ == "__main__":
    main()
