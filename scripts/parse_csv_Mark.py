#!/usr/bin/env python3
"""Convert ROI trace export from TXT to CSV and plot all ROI traces.

This script reads a tab-delimited ROI export file containing frame, time,
Green, Red, and Trace values for many ROIs. It writes:

1. a CSV copy of the original file
2. a CSV with only Frame, Time, and all Trace columns
3. a PNG overview plot of all trace columns
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ROI txt export to CSV, trace-only CSV, and overview plot."
    )
    parser.add_argument(
        "input_file",
        type=pathlib.Path,
        help="Path to the input TXT file containing ROI trace export.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to write outputs. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Field separator for the input file. Default is tab.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Generate overview plot of all traces.",
    )
    parser.add_argument(
        "--visualise-normalized",
        action="store_true",
        help="Generate overview plot of normalized traces.",
    )
    parser.add_argument(
        "--estimate-noise",
        action="store_true",
        help="Estimate the noise level for each ROI trace.",
    )
    parser.add_argument(
        "--noise-output",
        type=pathlib.Path,
        default=None,
        help="Optional path to save the estimated noise levels CSV.",
    )
    return parser.parse_args()


def load_txt(input_path: pathlib.Path, sep: str = "\t") -> pd.DataFrame:
    return pd.read_csv(
        input_path,
        sep=sep,
        dtype=str,
        na_values=["N/A", "NA"],
        keep_default_na=False,
        engine="python",
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip() for col in df.columns]
    return df


def filter_trace_columns(df: pd.DataFrame) -> list[str]:
    trace_columns = [col for col in df.columns if "Trace" in col]
    valid_columns = []
    for col in trace_columns:
        values = pd.to_numeric(df[col], errors='coerce')
        if values.isna().all() or (values == 0).all():
            continue
        valid_columns.append(col)
    if not valid_columns:
        raise ValueError("No valid trace columns found in the input file (all contain N/A or only zeros).")
    return valid_columns


def add_normalized_trace_columns(
    df: pd.DataFrame,
    trace_columns: list[str],
    time_column: str = "Time",
    baseline_duration: float = 3.0,
) -> pd.DataFrame:
    """Add a normalized trace column for each ROI based on the first 3 seconds."""
    time_values = pd.to_numeric(df[time_column], errors="coerce")
    if time_values.isna().any():
        raise ValueError(f"Non-numeric values found in {time_column} column.")

    baseline_mask = time_values <= baseline_duration
    if not baseline_mask.any():
        raise ValueError(
            f"No data points found in the first {baseline_duration} seconds for baseline normalization."
        )

    normalized_df = df.copy()
    for trace_col in trace_columns:
        values = pd.to_numeric(df[trace_col], errors="coerce")
        baseline_mean = values[baseline_mask].mean()
        if baseline_mean == 0 or np.isnan(baseline_mean):
            normalized_df[f"{trace_col}_normalized"] = np.nan
        else:
            normalized_df[f"{trace_col}_normalized"] = values / baseline_mean

    return normalized_df


def find_green_column_for_trace(
    df: pd.DataFrame,
    trace_column: str,
    fallback_green_column: str = "Green",
) -> str:
    """Return the raw green fluorescence column matching a trace column."""
    roi_match = re.search(r"(ROI\d+)$", trace_column)
    if roi_match:
        roi_label = roi_match.group(1)
        candidates = [
            f"Green_Mean_{roi_label}",
            f"Green_{roi_label}",
            trace_column.replace("Trace", "Green_Mean", 1),
            trace_column.replace("Trace", "Green", 1),
        ]
    else:
        candidates = [
            trace_column.replace("Trace", "Green_Mean", 1),
            trace_column.replace("Trace", "Green", 1),
            fallback_green_column,
        ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not find green fluorescence column for {trace_column}. "
        f"Tried: {', '.join(dict.fromkeys(candidates))}"
    )


def estimate_noise_level_from_green(green_values: pd.Series, fps: float, green_column: str) -> float:
    green_values = pd.to_numeric(green_values, errors="coerce")
    if green_values.isna().any():
        raise ValueError(f"Non-numeric values found in {green_column} column.")

    f_next = green_values.shift(-1)
    f_curr = green_values
    delta_f = f_next - f_curr

    delta_f_over_f_curr = delta_f / f_curr.replace(0, np.nan)
    delta_f_over_f_next = delta_f / f_next.replace(0, np.nan)

    diff_ratio = (delta_f_over_f_next - delta_f_over_f_curr).abs()
    median_diff = diff_ratio.median()

    return median_diff / np.sqrt(fps)


def estimate_noise_levels(
    df: pd.DataFrame,
    trace_columns: list[str],
    time_column: str = "Time",
    green_column: str = "Green",
) -> pd.Series:
    """
    Estimate noise level for each ROI using the matching raw green fluorescence.

    For Trace_ROI1, Trace_ROI2, etc., noise is calculated from the corresponding
    Green_Mean_ROI1, Green_Mean_ROI2, etc. column.
    """
    if time_column not in df.columns:
        raise ValueError(f"Time column not found: {time_column}")

    # Compute frame rate
    time_values = pd.to_numeric(df[time_column], errors="coerce")
    if time_values.isna().any():
        raise ValueError(f"Non-numeric values found in {time_column} column.")

    time_diffs = time_values.diff().iloc[1:]
    median_dt = time_diffs[time_diffs > 0].median()
    fps = 1.0 / median_dt if median_dt and median_dt > 0 else 1.0
    print(f">> Frame rate: {fps}")

    noise_levels = {}
    for trace_col in trace_columns:
        matching_green_column = find_green_column_for_trace(
            df,
            trace_col,
            fallback_green_column=green_column,
        )
        noise_levels[trace_col] = estimate_noise_level_from_green(
            df[matching_green_column],
            fps,
            matching_green_column,
        )

    return pd.Series(noise_levels, name="noise_level")


def save_noise_estimates(
    noise_levels: pd.Series,
    path: pathlib.Path,
    overwrite: bool = False,
) -> None:
    noise_df = noise_levels.rename_axis("Trace").reset_index(name="NoiseLevel")
    save_csv(noise_df, path, overwrite=overwrite)


def save_csv(df: pd.DataFrame, path: pathlib.Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")
    df.to_csv(path, index=False)


def save_trace_csv(df: pd.DataFrame, frame_time_cols: list[str], trace_columns: list[str], path: pathlib.Path, overwrite: bool = False) -> None:
    norm_columns = [f"{col}_normalized" for col in trace_columns if f"{col}_normalized" in df.columns]
    trace_df = df.loc[:, frame_time_cols + trace_columns + norm_columns]
    save_csv(trace_df, path, overwrite=overwrite)


def plot_trace_overview(df: pd.DataFrame, time_column: str, trace_columns: list[str], path: pathlib.Path) -> None:
    plt.figure(figsize=(14, 9))
    x = df[time_column].astype(float)
    for trace_col in trace_columns:
        plt.plot(x, df[trace_col].astype(float), label=trace_col, linewidth=1, alpha=0.8)

    plt.title("ROI Trace Overview")
    plt.xlabel(time_column)
    plt.ylabel("Trace")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if len(trace_columns) <= 20:
        plt.legend(fontsize="small", ncol=2)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_normalized_trace_overview(
    df: pd.DataFrame,
    time_column: str,
    trace_columns: list[str],
    path: pathlib.Path,
) -> None:
    plt.figure(figsize=(14, 9))
    x = df[time_column].astype(float)
    normalized_columns = [f"{col}_normalized" for col in trace_columns if f"{col}_normalized" in df.columns]
    if not normalized_columns:
        raise ValueError("No normalized trace columns found for plotting.")

    # Filter to first 25 seconds
    mask = x <= 25
    x_filtered = x[mask]
    if x_filtered.empty:
        raise ValueError("No data points found in the first 25 seconds.")

    num_plots = len(normalized_columns)
    cols = 5  # Number of columns in grid
    rows = (num_plots + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(28, 18), sharex=True, sharey=True)
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, norm_col in enumerate(normalized_columns):
        ax = axes[i]
        y_filtered = df.loc[mask, norm_col].astype(float)
        ax.plot(x_filtered, y_filtered, linewidth=1.5)
        ax.set_title(norm_col.replace("_normalized", ""), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Normalized ROI Trace Overview (First 25 Seconds)", fontsize=14)
    fig.text(0.5, 0.04, time_column, ha='center', fontsize=12)
    fig.text(0.04, 0.5, "Normalized Trace", va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=(0.05, 0.05, 1, 0.95))
    plt.savefig(path, dpi=600)
    plt.close()


def main() -> None:
    args = parse_arguments()
    input_path = args.input_file.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_txt(input_path, sep=args.sep)
    df = normalize_columns(df)

    frame_time_candidates = ["Frame", "Time"]
    missing = [col for col in frame_time_candidates if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required column(s): {', '.join(missing)}")

    trace_columns = filter_trace_columns(df)
    df = add_normalized_trace_columns(df, trace_columns, time_column="Time", baseline_duration=3.0)

    csv_path = output_dir / f"{input_path.stem}.csv"
    trace_csv_path = output_dir / f"{input_path.stem}_trace_only.csv"
    plot_path = output_dir / f"{input_path.stem}_trace_overview.png"
    noise_path = args.noise_output or output_dir / f"{input_path.stem}_noise_estimates.csv"

    # save_csv(df, csv_path, overwrite=args.overwrite)
    save_trace_csv(df, frame_time_candidates, trace_columns, trace_csv_path, overwrite=args.overwrite)
    if args.visualise:
        plot_trace_overview(df, "Time", trace_columns, plot_path)
    if args.visualise_normalized:
        normalized_plot_path = output_dir / f"{input_path.stem}_normalized_trace_overview.png"
        plot_normalized_trace_overview(df, "Time", trace_columns, normalized_plot_path)
        print(f"Wrote normalized overview plot: {normalized_plot_path}")

    if args.estimate_noise:
        noise_levels = estimate_noise_levels(df, trace_columns, time_column="Time")
        save_noise_estimates(noise_levels, noise_path, overwrite=args.overwrite)
        print(f"Wrote noise estimates CSV: {noise_path}")

    # print(f"Wrote CSV: {csv_path}")
    print(f"Wrote trace-only CSV: {trace_csv_path}")
    if args.visualise:
        print(f"Wrote overview plot: {plot_path}")


if __name__ == "__main__":
    main()
