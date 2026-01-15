#!/usr/bin/env python

import argparse
import shutil
import subprocess
import sys
from os import name as os_name
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Nsight Compute .ncu-rep to CSV and plot roofline PNG."
    )
    default_ncu = "ncu"
    if os_name == "nt":
        default_ncu = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\target\windows-desktop-win7-x64\ncu.exe"
    parser.add_argument(
        "rep",
        nargs="?",
        help="Path to the .ncu-rep file.",
    )
    parser.add_argument(
        "--rep",
        dest="rep_flag",
        default=None,
        help="Path to the .ncu-rep file (same as positional).",
    )
    parser.add_argument(
        "--ncu",
        default=default_ncu,
        help="Path to ncu executable.",
    )
    parser.add_argument(
        "--plot-script",
        default="prof/plot_ncu_roofline.py",
        help="Path to the roofline plotting script.",
    )
    parser.add_argument(
        "--no-peak-formula",
        action="store_true",
        help="Disable --use-peak-formula when calling the plotting script.",
    )
    args = parser.parse_args()

    rep_value = args.rep_flag or args.rep
    if not rep_value:
        parser.error("the following arguments are required: rep")
    rep_path = Path(rep_value)
    if rep_path.suffix.lower() != ".ncu-rep":
        raise ValueError(f"--rep must be a .ncu-rep file: {rep_path}")

    out_dir = rep_path.parent / rep_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (rep_path.stem + ".csv")
    png_path = out_dir / (rep_path.stem + ".png")

    ncu_bin = shutil.which(args.ncu) if args.ncu == "ncu" else args.ncu
    if not ncu_bin:
        raise FileNotFoundError("ncu not found on PATH. Pass --ncu /path/to/ncu.")
    ncu_cmd = [
        ncu_bin,
        "--import",
        str(rep_path),
        "--page",
        "raw",
        "--csv",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        subprocess.run(ncu_cmd, stdout=f, check=True)

    plot_cmd = [
        sys.executable,
        args.plot_script,
        "--csv",
        str(csv_path),
        "--out",
        str(png_path),
    ]
    if not args.no_peak_formula:
        plot_cmd.append("--use-peak-formula")
    plot_cmd.extend(["--out-list", str(out_dir / (rep_path.stem + "_bounds.csv"))])
    subprocess.run(plot_cmd, check=True)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
