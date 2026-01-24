#!/usr/bin/env python3
"""
Collect and analyze results from SLURM array job hyperparameter search.

Usage:
    python examples/collect_results.py
    python examples/collect_results.py --top 20
"""

import argparse
import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("./hypersearch_results")


def collect_results() -> pd.DataFrame:
    """Collect all individual result files into a DataFrame."""
    results = []

    for result_file in sorted(RESULTS_DIR.glob("result_*.json")):
        with open(result_file) as f:
            results.append(json.load(f))

    if not results:
        print("No results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("total_score", ascending=False)
    return df


def print_analysis(df: pd.DataFrame, top_n: int = 10):
    """Print analysis of results."""

    print("=" * 70)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 70)
    print(f"Total experiments: {len(df)}")
    print(f"Successful: {len(df[df['error'].isna()])}")
    print(f"Failed: {len(df[df['error'].notna()])}")
    print()

    # Top configs
    print(f"TOP {top_n} CONFIGURATIONS:")
    print("-" * 70)
    for i, row in df.head(top_n).iterrows():
        print(f"Score: {row['total_score']:.2f} | "
              f"Acc: {row['balanced_accuracy']*100:.1f}% | "
              f"Lag: {row['avg_lag_samples']:.0f}ms | "
              f"win={row['window_size']}, ch={row['projected_channels']}, "
              f"F1={row['F1']}, D={row['D']}, drop={row['dropout']}")
    print()

    # Best by window size
    print("BEST SCORE BY WINDOW SIZE:")
    print("-" * 70)
    for ws in sorted(df["window_size"].unique()):
        ws_df = df[df["window_size"] == ws]
        best = ws_df.iloc[0]
        print(f"  window={ws:4d}: {best['total_score']:.2f} "
              f"(acc={best['balanced_accuracy']*100:.1f}%, "
              f"ch={best['projected_channels']}, F1={best['F1']})")
    print()

    # Best by projected channels
    print("BEST SCORE BY PROJECTED CHANNELS:")
    print("-" * 70)
    for pc in sorted(df["projected_channels"].unique()):
        pc_df = df[df["projected_channels"] == pc]
        best = pc_df.iloc[0]
        print(f"  channels={pc:3d}: {best['total_score']:.2f} "
              f"(acc={best['balanced_accuracy']*100:.1f}%, "
              f"win={best['window_size']})")
    print()

    # Correlation analysis
    print("CORRELATION WITH TOTAL SCORE:")
    print("-" * 70)
    numeric_cols = ["window_size", "projected_channels", "F1", "dropout"]
    for col in numeric_cols:
        corr = df[col].corr(df["total_score"])
        print(f"  {col}: {corr:+.3f}")
    print()

    # Best overall
    best = df.iloc[0]
    print("=" * 70)
    print("BEST CONFIGURATION:")
    print("=" * 70)
    print(f"  window_size: {best['window_size']}")
    print(f"  projected_channels: {best['projected_channels']}")
    print(f"  F1: {best['F1']}")
    print(f"  D: {best['D']}")
    print(f"  dropout: {best['dropout']}")
    print(f"  learning_rate: {best['learning_rate']}")
    print()
    print(f"  Total Score: {best['total_score']:.2f}")
    print(f"  Balanced Accuracy: {best['balanced_accuracy']*100:.2f}%")
    print(f"  Avg Lag: {best['avg_lag_samples']:.1f}ms")
    print(f"  Model Size: {best['model_size_bytes']/1024:.1f}KB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze hypersearch results")
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to show")
    parser.add_argument("--output", type=str, default=None, help="Save combined CSV to file")
    args = parser.parse_args()

    df = collect_results()

    if df.empty:
        return

    # Save combined results
    output_file = args.output or (RESULTS_DIR / "all_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Combined results saved to: {output_file}\n")

    # Print analysis
    print_analysis(df, args.top)


if __name__ == "__main__":
    main()
