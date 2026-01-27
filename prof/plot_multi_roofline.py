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
    if u in {"byte/cycle", "bytes/cycle"}:
        return 1.0
    if u in {"kbyte/cycle", "kbytes/cycle"}:
        return 1e3
    if u in {"mbyte/cycle", "mbytes/cycle"}:
        return 1e6
    if u in {"gbyte/cycle", "gbytes/cycle"}:
        return 1e9
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
    ops = (sub["ops_per_sec"] * sub["time_s"]).sum()
    bytes_ = (sub["bytes_per_sec"] * sub["time_s"]).sum()
    time = sub["time_s"].sum()
    if time <= 0 or bytes_ <= 0:
        return None
    ai = ops / bytes_
    perf = ops / time
    bw_pct = None
    if "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" in sub.columns:
        bw_pct = sub["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"].mean()
    return {"ai": ai, "perf": perf, "time": time, "bw_pct": bw_pct}


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
        "--out-points",
        default="",
        help="Optional CSV output with plotted points.",
    )
    parser.add_argument(
        "--aggregate-attn-fc",
        action="store_true",
        help="Aggregate attention and FC/MLP kernels into two points per input file.",
    )
    parser.add_argument(
        "--attn-regex",
        action="append",
        default=[r"\.self_attn\.", r"post_attention_layernorm"],
        help="Regex for attention block matching (repeatable).",
    )
    parser.add_argument(
        "--fc-regex",
        action="append",
        default=[r"\.mlp\.", r"\.mlp\.fc\d+", r"action_.*_proj", r"state_proj"],
        help="Regex for FC/MLP block matching (repeatable).",
    )
    args = parser.parse_args()

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
        series.append((path.stem, df, peaks))

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

    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap("tab10")
    size_by_bw = False
    points_rows = []
    if args.aggregate_attn_fc:
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
                plt.scatter(
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
                plt.scatter(
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
        for idx, (label, df, _peaks) in enumerate(series):
            if "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" in df.columns:
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
            plt.scatter(
                df["ai"],
                df["perf"] / plot_scale,
                s=sizes,
                c=[colors(idx % 10)],
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                label=label,
            )
        size_label = "Point size = memory BW % of peak" if size_by_bw else "Point size = duration (s)"

    if peak_bw and peak_flops:
        ridge_x = peak_flops / peak_bw
        all_ai = pd.concat([d["ai"] for _, d, _ in series], ignore_index=True)
        xmin, xmax = all_ai.min(), all_ai.max()
        xmin = min(xmin / 10.0, ridge_x / 100.0)
        xmax = max(xmax * 10.0, ridge_x * 100.0)
        xs = [xmin, xmax]
        mem_line = [(peak_bw * x) / plot_scale for x in xs]
        comp_line = [peak_flops / plot_scale, peak_flops / plot_scale]
        plt.plot(xs, mem_line, "--", color="orange", linewidth=1, label="Memory roof")
        plt.plot(xs, comp_line, "--", color="red", linewidth=1, label="Compute roof")
        plt.xlim(xmin, xmax)
        plt.ylim(
            min(min(d["perf"].min() for _, d, _ in series) / plot_scale / 10.0, (peak_flops / plot_scale) / 100.0),
            (peak_flops / plot_scale) * 10.0,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("HW Arithmetic Intensity [FLOP/byte]")
    plt.ylabel(f"HW Performance [FLOP/s] (1 = {perf_scale:.0e})")
    plt.title("Multi-run Roofline")
    plt.gca().text(
        0.02,
        0.02,
        size_label,
        transform=plt.gca().transAxes,
        fontsize=8,
        alpha=0.8,
        va="bottom",
    )
    plt.legend(loc="best")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")

    if args.out_points:
        out_points = Path(args.out_points)
        out_points.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(points_rows).to_csv(out_points, index=False)
        print(f"Wrote {out_points}")


if __name__ == "__main__":
    main()
