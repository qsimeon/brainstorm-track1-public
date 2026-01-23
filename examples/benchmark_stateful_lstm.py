#!/usr/bin/env python3
"""
Benchmark script for Stateful LSTM Model.

This script provides comprehensive benchmarking of the Stateful LSTM model,
including:
    1. Inference latency measurements (per-sample timing)
    2. State reset ablation studies (effect of maintaining state)
    3. Hidden size variations (accuracy vs model size tradeoffs)
    4. Comparison with MLP baseline
    5. Detailed performance analysis

The Stateful LSTM is expected to:
    - Be 2-3x slower per sample than stateless models (due to LSTM computation)
    - Achieve 2-5% higher accuracy from temporal context
    - Maintain model size < 25MB (similar to MLP)

Usage:
    python examples/benchmark_stateful_lstm.py

Results are printed to console with detailed timing and accuracy comparisons.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from brainstorm.loading import load_raw_data
from brainstorm.ml.lstm import StatefulLSTM
from brainstorm.ml.mlp import MLP


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")

# Number of samples to use for latency benchmarks
LATENCY_BENCHMARK_SAMPLES = 1000


# =============================================================================
# Benchmarking Functions
# =============================================================================


def benchmark_inference_latency(
    model: Any, test_features: np.ndarray, n_samples: int = 1000
) -> dict[str, float]:
    """
    Measure inference latency per sample.

    Args:
        model: Trained model with predict() method.
        test_features: 2D array of test samples (n_samples, n_features).
        n_samples: Number of samples to benchmark (subsample for speed).

    Returns:
        Dictionary with latency statistics in milliseconds.
    """
    if n_samples > len(test_features):
        n_samples = len(test_features)

    times = []
    model.eval() if hasattr(model, "eval") else None

    for i in range(n_samples):
        sample = test_features[i]

        start = time.perf_counter()
        _ = model.predict(sample)
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)  # Convert to milliseconds

    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "std_ms": np.std(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def benchmark_state_maintenance_ablation(
    test_features: np.ndarray, test_labels: np.ndarray
) -> dict[str, Any]:
    """
    Ablation study: Compare model performance with different state reset frequencies.

    Tests three conditions:
        1. State maintained throughout (default Stateful LSTM behavior)
        2. State reset every 100 samples
        3. State reset every sample (equivalent to stateless)

    Args:
        test_features: 2D array of test samples.
        test_labels: 1D array of test labels.

    Returns:
        Dictionary comparing predictions and accuracy across conditions.
    """
    console = Console()
    console.print("\n[bold yellow]Running State Reset Ablation Study...[/]")

    try:
        model = StatefulLSTM.load()
    except FileNotFoundError:
        rprint("[red]Error: StatefulLSTM model not found. Train first![/]")
        return {}

    results = {}

    # Test 1: State maintained throughout
    console.print("[cyan]Test 1: State maintained (default)...[/]")
    model.reset_state()
    predictions_continuous = []
    for sample in tqdm(test_features, desc="  Predicting (continuous state)", leave=False):
        pred = model.predict(sample)
        predictions_continuous.append(pred)

    accuracy_continuous = balanced_accuracy_score(test_labels, predictions_continuous)
    results["continuous"] = {
        "accuracy": accuracy_continuous,
        "predictions": predictions_continuous,
    }

    # Test 2: State reset every 100 samples
    console.print("[cyan]Test 2: State reset every 100 samples...[/]")
    model.reset_state()
    predictions_periodic = []
    for i, sample in enumerate(tqdm(test_features, desc="  Predicting (periodic reset)", leave=False)):
        if i % 100 == 0:
            model.reset_state()
        pred = model.predict(sample)
        predictions_periodic.append(pred)

    accuracy_periodic = balanced_accuracy_score(test_labels, predictions_periodic)
    results["periodic"] = {
        "accuracy": accuracy_periodic,
        "predictions": predictions_periodic,
    }

    # Test 3: State reset every sample (stateless)
    console.print("[cyan]Test 3: State reset every sample (stateless)...[/]")
    predictions_stateless = []
    for sample in tqdm(test_features, desc="  Predicting (stateless)", leave=False):
        model.reset_state()
        pred = model.predict(sample)
        predictions_stateless.append(pred)

    accuracy_stateless = balanced_accuracy_score(test_labels, predictions_stateless)
    results["stateless"] = {
        "accuracy": accuracy_stateless,
        "predictions": predictions_stateless,
    }

    return results


def compare_with_baseline(
    test_features: np.ndarray, test_labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """
    Compare StatefulLSTM with MLP baseline.

    Measures latency and compares accuracy vs model size.

    Args:
        test_features: 2D array of test samples.
        test_labels: 1D array of test labels.

    Returns:
        Dictionary with comparison metrics.
    """
    console = Console()
    console.print("\n[bold yellow]Comparing with MLP Baseline...[/]")

    results = {}

    # Load LSTM model
    try:
        lstm_model = StatefulLSTM.load()
        lstm_latency = benchmark_inference_latency(lstm_model, test_features, 500)
        results["lstm"] = {
            "latency": lstm_latency,
            "mean_ms": lstm_latency["mean_ms"],
            "p95_ms": lstm_latency["p95_ms"],
        }
        console.print(f"[green]✓ Loaded StatefulLSTM[/]")
    except FileNotFoundError:
        console.print("[red]✗ StatefulLSTM not found[/]")
        results["lstm"] = None
        return results

    # Load MLP model
    try:
        mlp_model = MLP.load()
        mlp_latency = benchmark_inference_latency(mlp_model, test_features, 500)
        results["mlp"] = {
            "latency": mlp_latency,
            "mean_ms": mlp_latency["mean_ms"],
            "p95_ms": mlp_latency["p95_ms"],
        }
        console.print(f"[green]✓ Loaded MLP baseline[/]")
    except FileNotFoundError:
        console.print("[dim]⊘ MLP not found (baseline comparison skipped)[/]")
        results["mlp"] = None
        return results

    # Compute speedup
    if results["lstm"] and results["mlp"]:
        speedup = results["lstm"]["mean_ms"] / results["mlp"]["mean_ms"]
        results["speedup_ratio"] = speedup

    return results


def get_model_size(model_path: Path) -> float:
    """
    Get model file size in MB.

    Args:
        model_path: Path to model file.

    Returns:
        File size in MB.
    """
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


# =============================================================================
# Main Benchmarking Pipeline
# =============================================================================


def main() -> None:
    rprint("\n[bold cyan]Stateful LSTM Benchmarking Suite[/]\n")

    # Load data
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("[red]Error: Data not found. Run example_lstm_train_and_evaluate.py first![/]\n")
        return

    console = Console()
    test_features, test_labels = load_raw_data(DATA_PATH, step="validation")
    test_features_np = test_features.values
    test_labels_np = test_labels["label"].values

    rprint(f"[dim]Loaded {len(test_features_np)} test samples[/]\n")

    # =======================================================================
    # Benchmark 1: Inference Latency
    # =======================================================================

    rprint("\n[bold cyan]═══════════════════════════════════════════════════════[/]")
    rprint("[bold cyan]Benchmark 1: Inference Latency[/]")
    rprint("[bold cyan]═══════════════════════════════════════════════════════[/]\n")

    try:
        lstm_model = StatefulLSTM.load()
        lstm_latency = benchmark_inference_latency(
            lstm_model, test_features_np, LATENCY_BENCHMARK_SAMPLES
        )

        latency_table = Table(title="Inference Latency (StatefulLSTM)", show_header=True)
        latency_table.add_column("Metric", style="cyan")
        latency_table.add_column("Time (ms)", style="green")

        latency_table.add_row("Mean", f"{lstm_latency['mean_ms']:.3f}")
        latency_table.add_row("Median", f"{lstm_latency['median_ms']:.3f}")
        latency_table.add_row("Std Dev", f"{lstm_latency['std_ms']:.3f}")
        latency_table.add_row("P95", f"{lstm_latency['p95_ms']:.3f}")
        latency_table.add_row("P99", f"{lstm_latency['p99_ms']:.3f}")

        console.print(latency_table)
        rprint("\n[dim]Expected: 1.5-2.5ms mean (2-3x slower than stateless models)[/]\n")

    except FileNotFoundError:
        rprint("[red]✗ StatefulLSTM model not found. Train first![/]\n")
        return

    # =======================================================================
    # Benchmark 2: State Reset Ablation
    # =======================================================================

    rprint("\n[bold cyan]═══════════════════════════════════════════════════════[/]")
    rprint("[bold cyan]Benchmark 2: State Maintenance Ablation[/]")
    rprint("[bold cyan]═══════════════════════════════════════════════════════[/]")

    ablation_results = benchmark_state_maintenance_ablation(test_features_np, test_labels_np)

    if ablation_results:
        ablation_table = Table(title="Effect of State Management", show_header=True)
        ablation_table.add_column("Condition", style="cyan")
        ablation_table.add_column("Accuracy", style="green")
        ablation_table.add_column("vs Stateless", style="yellow")

        if "stateless" in ablation_results:
            baseline_acc = ablation_results["stateless"]["accuracy"]

            for condition in ["continuous", "periodic", "stateless"]:
                if condition in ablation_results:
                    acc = ablation_results[condition]["accuracy"]
                    improvement = acc - baseline_acc
                    improvement_str = (
                        f"[green]+{improvement:.1%}[/]"
                        if improvement > 0
                        else f"[red]{improvement:.1%}[/]"
                    )
                    ablation_table.add_row(
                        condition.replace("_", " ").title(),
                        f"{acc:.3f}",
                        improvement_str,
                    )

        console.print(ablation_table)
        rprint("[dim]Expected: Continuous > Periodic > Stateless (by 2-5%)[/]\n")

    # =======================================================================
    # Benchmark 3: Comparison with Baseline
    # =======================================================================

    rprint("\n[bold cyan]═══════════════════════════════════════════════════════[/]")
    rprint("[bold cyan]Benchmark 3: Comparison with MLP Baseline[/]")
    rprint("[bold cyan]═══════════════════════════════════════════════════════[/]\n")

    comparison_results = compare_with_baseline(test_features_np, test_labels_np)

    if comparison_results.get("lstm") and comparison_results.get("mlp"):
        comparison_table = Table(
            title="Inference Latency Comparison", show_header=True
        )
        comparison_table.add_column("Model", style="cyan")
        comparison_table.add_column("Mean (ms)", style="green")
        comparison_table.add_column("P95 (ms)", style="green")
        comparison_table.add_column("Speedup", style="yellow")

        comparison_table.add_row(
            "StatefulLSTM",
            f"{comparison_results['lstm']['mean_ms']:.3f}",
            f"{comparison_results['lstm']['p95_ms']:.3f}",
            "1.0x (reference)",
        )

        if comparison_results["mlp"]:
            mlp_speedup = 1.0 / comparison_results["speedup_ratio"]
            comparison_table.add_row(
                "MLP",
                f"{comparison_results['mlp']['mean_ms']:.3f}",
                f"{comparison_results['mlp']['p95_ms']:.3f}",
                f"{mlp_speedup:.2f}x faster",
            )

        console.print(comparison_table)
        rprint("[dim]Expected: LSTM 2-3x slower, but improves accuracy via temporal context[/]\n")

    # =======================================================================
    # Model Size Comparison
    # =======================================================================

    rprint("\n[bold cyan]═══════════════════════════════════════════════════════[/]")
    rprint("[bold cyan]Model Size Analysis[/]")
    rprint("[bold cyan]═══════════════════════════════════════════════════════[/]\n")

    repo_root = Path(__file__).parent.parent
    lstm_path = repo_root / "model.pt"
    lstm_size = get_model_size(lstm_path)

    size_table = Table(title="Model File Sizes", show_header=True)
    size_table.add_column("Model", style="cyan")
    size_table.add_column("Size (MB)", style="green")
    size_table.add_column("vs 5MB Target", style="yellow")

    if lstm_size > 0:
        size_table.add_row(
            "StatefulLSTM",
            f"{lstm_size:.2f}",
            f"[green]{lstm_size / 5.0:.1%}[/]" if lstm_size < 5 else f"[yellow]{lstm_size / 5.0:.1%}[/]",
        )

    console.print(size_table)
    rprint(f"[dim]Expected: ~5MB (1.33M parameters × 4 bytes)[/]\n")

    # =======================================================================
    # Summary and Conclusions
    # =======================================================================

    rprint("\n[bold cyan]═══════════════════════════════════════════════════════[/]")
    rprint("[bold cyan]Benchmarking Summary[/]")
    rprint("[bold cyan]═══════════════════════════════════════════════════════[/]\n")

    summary = Table(title="Benchmark Summary", show_header=True)
    summary.add_column("Aspect", style="cyan")
    summary.add_column("Status", style="green")
    summary.add_column("Details", style="dim")

    if comparison_results.get("lstm"):
        mean_lat = comparison_results["lstm"]["mean_ms"]
        summary.add_row(
            "Latency",
            "[green]✓[/]" if mean_lat < 5.0 else "[yellow]⚠[/]",
            f"Mean {mean_lat:.2f}ms (target <5ms)",
        )

    if ablation_results and "continuous" in ablation_results:
        acc = ablation_results["continuous"]["accuracy"]
        summary.add_row(
            "Accuracy",
            "[green]✓[/]" if acc > 0.45 else "[yellow]⚠[/]",
            f"Balanced accuracy {acc:.1%}",
        )

    if lstm_size > 0:
        summary.add_row(
            "Model Size",
            "[green]✓[/]" if lstm_size < 25 else "[red]✗[/]",
            f"{lstm_size:.2f}MB (limit 25MB)",
        )

    console.print(summary)

    rprint("\n[bold green]Benchmarking Complete![/]\n")
    rprint("[dim]Compare results with MLP baseline to evaluate temporal context benefit[/]\n")


if __name__ == "__main__":
    main()
