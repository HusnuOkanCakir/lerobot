#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def _parse_batch_size(path: Path) -> int:
    match = re.search(r"bs(\d+)", path.name)
    if match:
        return int(match.group(1))
    df = pd.read_csv(path)
    if "batch_size" in df.columns and len(df["batch_size"]) > 0:
        return int(df["batch_size"].iloc[0])
    return 1


def _summarize_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if "batch_latency_s" not in df.columns:
        raise ValueError(f"Missing batch_latency_s in {path}")
    batch_size = _parse_batch_size(path)
    avg_batch = float(df["batch_latency_s"].mean())
    avg_per_sample = avg_batch / batch_size if batch_size > 0 else avg_batch
    return {
        "file": path.name,
        "batch_size": batch_size,
        "avg_batch_s": avg_batch,
        "avg_per_sample_s": avg_per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bar chart from iter_*.csv files.")
    parser.add_argument(
        "--dir",
        default="cluster_out",
        help="Directory containing iter_*.csv files (default: cluster_out).",
    )
    parser.add_argument(
        "--out",
        default="cluster_out/iter_latency_bars.png",
        help="Output PNG path for batch latency plot.",
    )
    parser.add_argument(
        "--out_per_sample",
        default="cluster_out/iter_latency_bars_per_sample.png",
        help="Output PNG path for per-sample latency plot.",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("iter_*.csv"))
    if not files:
        raise SystemExit(f"No iter_*.csv files found in {root}")

    rows = []
    for path in files:
        rows.append(_summarize_csv(path))

    df = pd.DataFrame(rows)
    df["model"] = df["file"].apply(lambda name: "pi0" if "pi0" in name else "smolvla")
    df = df.sort_values(["batch_size", "model", "file"])
    df = (
        df.groupby(["batch_size", "model"], as_index=False)
        .agg(avg_batch_s=("avg_batch_s", "mean"), avg_per_sample_s=("avg_per_sample_s", "mean"))
    )

    batch_df = df.pivot(index="batch_size", columns="model", values="avg_batch_s")
    per_sample_df = df.pivot(index="batch_size", columns="model", values="avg_per_sample_s")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required: {exc}") from exc

    ax = batch_df.plot(kind="bar", figsize=(7.5, 4.5))
    ax.set_xlabel("batch_size")
    ax.set_ylabel("avg_batch_latency_s")
    ax.set_title("Iter latency (batch) by batch size")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Wrote {out_path}")

    ax = per_sample_df.plot(kind="bar", figsize=(7.5, 4.5))
    ax.set_xlabel("batch_size")
    ax.set_ylabel("avg_per_sample_latency_s")
    ax.set_title("Iter latency (per-sample) by batch size")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = Path(args.out_per_sample)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
