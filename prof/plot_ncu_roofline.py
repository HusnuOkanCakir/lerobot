#!/usr/bin/env python

import argparse
import csv
import math
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


def _pick_columns(header: list[str], candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in header]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a roofline scatter from Nsight Compute CSV.")
    parser.add_argument("--csv", default="prof/ncu_roofline.csv", help="Path to NCU CSV export.")
    parser.add_argument("--out", default="prof/roofline_scatter.png", help="Output PNG path.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Label the top-N kernels by time. Use 0 to disable labels.",
    )
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=0.0,
        help="Override peak compute in ops/s for roofline ceiling (0 = infer).",
    )
    parser.add_argument(
        "--peak-bw",
        type=float,
        default=0.0,
        help="Override peak memory bandwidth in bytes/s for roofline ceiling (0 = infer).",
    )
    parser.add_argument(
        "--use-peak-formula",
        action="store_true",
        help="Use Nsight peak formula columns for roofline ceilings when available.",
    )
    parser.add_argument(
        "--out-list",
        default="",
        help="Optional CSV output with bound info and vertical distance to roofline.",
    )
    parser.add_argument(
        "--perf-scale",
        type=float,
        default=1e11,
        help="Scale factor for Y-axis (1 on plot equals this many FLOP/s).",
    )
    parser.add_argument(
        "--score-by",
        choices=["duration", "memory_bw"],
        default="memory_bw",
        help="Score using duration or memory bandwidth usage.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

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
    peak_fp_cols = []

    usecols = set(per_cycle_cols + [bytes_col, cycles_per_sec_col, "gpu__time_duration.sum"])
    if "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" in header:
        usecols.add("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    if peak_bw_col:
        usecols.add(peak_bw_col)
    if peak_fp_cols:
        usecols.update(peak_fp_cols)
    if peak_work_col:
        usecols.add(peak_work_col)
    if peak_cycles_col:
        usecols.add(peak_cycles_col)
    if peak_traffic_cycles_col:
        usecols.add(peak_traffic_cycles_col)
    if kernel_name_col:
        usecols.add(kernel_name_col)
    for col in nvtx_cols:
        usecols.add(col)

    df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding, engine="python")
    for col in per_cycle_cols + [bytes_col, cycles_per_sec_col]:
        df[col] = df[col].map(_to_float)
    if peak_bw_col:
        df[peak_bw_col] = df[peak_bw_col].map(_to_float)
    for col in peak_fp_cols:
        df[col] = df[col].map(_to_float)
    if peak_work_col:
        df[peak_work_col] = df[peak_work_col].map(_to_float)
    if peak_cycles_col:
        df[peak_cycles_col] = df[peak_cycles_col].map(_to_float)
    if peak_traffic_cycles_col:
        df[peak_traffic_cycles_col] = df[peak_traffic_cycles_col].map(_to_float)
    if "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" in df.columns:
        df["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"] = df[
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        ].map(_to_float)

    ops_per_cycle = (
        df.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed", 0.0)
        + df.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed", 0.0)
        + df.get("derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2", 0.0)
    )
    cycles_per_sec = df[cycles_per_sec_col]
    df["ops_per_sec"] = ops_per_cycle * cycles_per_sec
    df["bytes_per_sec"] = df[bytes_col]
    time_col = "gpu__time_duration.sum"
    df["time_ns"] = df.get(time_col, pd.Series([math.nan] * len(df))).map(_to_float)

    df = df[(df["ops_per_sec"] > 0) & (df["bytes_per_sec"] > 0)].copy()
    if df.empty:
        raise RuntimeError("No valid rows found after filtering.")

    df["ai"] = df["ops_per_sec"] / df["bytes_per_sec"]
    df["perf"] = df["ops_per_sec"]
    perf_scale = args.perf_scale if args.perf_scale > 0 else 1.0
    df["perf_plot"] = df["perf"] / perf_scale

    plt.figure(figsize=(8, 6))
    sizes = 30.0
    colors = None
    if df["time_ns"].notna().any():
        time_s = df["time_ns"].fillna(0) * 1e-9
        sizes = (time_s / max(time_s.max(), 1e-12) * 80.0) + 10.0
        colors = time_s
    if colors is not None:
        order = colors.sort_values().index
        df_plot = df.loc[order]
        sizes_plot = sizes[order]
        colors_plot = colors[order]
    else:
        df_plot = df
        sizes_plot = sizes
        colors_plot = colors
    sc = plt.scatter(
        df_plot["ai"],
        df_plot["perf_plot"],
        s=sizes_plot,
        c=colors_plot,
        cmap="inferno",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    if colors is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label("Duration (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("HW Arithmetic Intensity [FLOP/byte]")
    plt.ylabel(f"HW Performance [FLOP/s] (1 = {perf_scale:.0e})")
    title_name = csv_path.stem.replace("_", " ")
    plt.title(f"{title_name} - Floating Point Operations Roofline")

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

    peak_bw = args.peak_bw
    peak_flops = args.peak_flops
    if args.use_peak_formula and peak_work_col and peak_cycles_col and peak_bw_col and peak_traffic_cycles_col:
        peak_flops = max(peak_flops, (df[peak_work_col] * df[peak_cycles_col]).max())
        peak_bw = max(peak_bw, (df[peak_bw_col] * df[peak_traffic_cycles_col]).max())
    if peak_bw <= 0 and peak_bw_col:
        peak_bw = df[peak_bw_col].max()
    bound_rows = []
    if peak_bw and peak_flops:
        ridge_x = peak_flops / peak_bw
        xmin, xmax = df["ai"].min(), df["ai"].max()
        xmin = min(xmin/10.0, ridge_x / 100.0)
        xmax = max(xmax*10.0, ridge_x * 100.0)
        xs = [xmin, xmax]
        mem_line = [(peak_bw * x) / perf_scale for x in xs]
        comp_line = [peak_flops / perf_scale, peak_flops / perf_scale]
        plt.plot(xs, mem_line, "--", color="orange", linewidth=1, label="Memory roof")
        plt.plot(xs, comp_line, "--", color="red", linewidth=1, label="Compute roof")
        plt.xlim(xmin, xmax)
        plt.ylim(
            min(df["perf_plot"].min() / 10.0, (peak_flops / perf_scale) / 100.0),
            (peak_flops / perf_scale) * 10.0,
        )
        plt.legend(loc="best")
        for _, row in df.iterrows():
            ai = row["ai"]
            perf = row["perf"]
            mem_roof = peak_bw * ai
            if mem_roof <= peak_flops:
                bound = "memory"
                roof = mem_roof
            else:
                bound = "compute"
                roof = peak_flops
            dist = max(roof - perf, 0.0)
            name = _row_name(row)
            duration_s = row.get("time_ns", math.nan)
            duration_s = 0.0 if math.isnan(duration_s) else duration_s * 1e-9
            bw_pct = row.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", math.nan)
            bw_pct = 0.0 if math.isnan(bw_pct) else bw_pct
            if args.score_by == "memory_bw":
                score = bw_pct / dist if dist > 0 else float("inf")
            else:
                score = duration_s / dist if dist > 0 else float("inf")
            bound_rows.append((bound, dist, score, duration_s, bw_pct, name))

    if kernel_name_col and args.top_n > 0:
        top = df.nlargest(args.top_n, "time_ns")
        for _, row in top.iterrows():
            name = _row_name(row)
            plt.annotate(name, (row["ai"], row["perf"]), fontsize=7, alpha=0.8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    if args.out_list and bound_rows:
        out_list = Path(args.out_list)
        out_list.parent.mkdir(parents=True, exist_ok=True)
        mem_rows = sorted([r for r in bound_rows if r[0] == "memory"], key=lambda r: r[2], reverse=True)
        comp_rows = sorted([r for r in bound_rows if r[0] == "compute"], key=lambda r: r[2], reverse=True)
        with out_list.open("w", newline="", encoding="utf-8") as f:
            f.write("name,bound,vertical_distance,score,duration_s,mem_bw_pct\n")
            for bound, dist, score, duration_s, bw_pct, name in mem_rows + comp_rows:
                f.write(f"\"{name}\",{bound},{dist},{score},{duration_s},{bw_pct}\n")
        print(f"Wrote {out_list}")


if __name__ == "__main__":
    main()

