#!/usr/bin/env python3
"""
Example script: Train and Evaluate a Stateful LSTM for Streaming BCI Classification

This script demonstrates the complete workflow for:
    1. Downloading ECoG data from Hugging Face (if not already downloaded)
    2. Loading ECoG data (features and labels)
    3. Training a Stateful LSTM model that maintains hidden state across predictions
    4. Running inference and computing evaluation metrics
    5. Displaying results

The Stateful LSTM is specifically designed for online inference scenarios where
samples arrive sequentially (one at a time). The model maintains hidden state
across predict() calls, enabling temporal awareness even though each prediction
is made independently.

Usage:
    python examples/example_lstm_train_and_evaluate.py

The trained model and metadata are saved to the repository root.
"""

from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.lstm import StatefulLSTM


# =============================================================================
# Configuration
# =============================================================================

# Path to the formatted data directory
DATA_PATH = Path("./data")

# Training parameters for StatefulLSTM
EPOCHS = 3
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3


def main() -> None:
    rprint("\n[bold green]Stateful LSTM Training and Evaluation Pipeline[/]\n")

    # Download data if not already present
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    validation_features, validation_labels = load_raw_data(DATA_PATH, step="validation")

    # Create a nice table to display dataset information
    console = Console()
    table = Table(
        title="Dataset Overview", show_header=True, header_style="bold magenta"
    )

    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

    # Add training data row
    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} → {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )

    # Add validation data row
    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} → {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    rprint("\n[bold green]Training Stateful LSTM Model...[/]\n")
    rprint("[dim]Note: Sequential sample-by-sample training to match evaluation pattern[/]\n")

    # Create model with configuration
    model = StatefulLSTM(
        input_size=train_features.shape[1],
        hidden_size=HIDDEN_SIZE,
    )

    # fit() calls fit_model(), saves the model, validates it, and saves metadata
    # Pass validation data to monitor validation loss and accuracy during training
    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,  # type: ignore[union-attr]
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        verbose=True,
        X_val=validation_features.values,
        y_val=validation_labels["label"].values,  # type: ignore[union-attr]
    )

    rprint("\n[bold green]Evaluating Stateful LSTM on validation set...[/]\n")
    rprint("[dim]Note: State is maintained across sequential predict() calls[/]\n")

    # NOTE: we use validation_features and labels for local evaluation.
    # The test set is held out and not accessible for local evaluation.
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],  # type: ignore[union-attr]
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Evaluation complete![/]\n")

    # Print key insights about stateful LSTM
    rprint("\n[bold cyan]Stateful LSTM Insights:[/]\n")
    rprint("[dim]• Hidden state from previous samples influences current prediction[/]")
    rprint("[dim]• Enables temporal awareness despite sample-by-sample inference[/]")
    rprint("[dim]• State automatically reset on next inference session[/]")
    rprint("[dim]• Compare lag score with stateless baselines (MLP, LogReg) to see benefit[/]\n")


if __name__ == "__main__":
    main()
