#!/usr/bin/env python3
"""
Example script: Train and Evaluate a Continuous Classification Model

This script demonstrates the complete workflow for:
    1. Downloading ECoG data from Hugging Face (if not already downloaded)
    2. Loading ECoG data (features and labels)
    3. Training a model for continuous classification
    4. Running inference and computing evaluation metrics
    5. Displaying results

Usage:
    python examples/example_local_train_and_evaluate.py

The trained model and metadata are saved to the repository root.
"""

from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.mlp import MLP
from brainstorm.ml.logistic_regression import LogisticRegression
from brainstorm.ml.eegnet import EEGNet
from brainstorm.ml.chronos_classifier import ChronosClassifier


# =============================================================================
# Configuration
# =============================================================================

# Path to the formatted data directory
DATA_PATH = Path("./data")

# Training parameters for MLP
EPOCHS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Training parameters for Logistic Regression
MAX_ITER = 20
USE_PCA = True

# MODEL_TO_USE = "mlp"
# MODEL_TO_USE = "logreg"
# MODEL_TO_USE = "eegnet"
MODEL_TO_USE = "chronos"


def main() -> None:
    rprint("\n[bold green]Evaluating model...[/]\n")

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

    # Add test data row
    table.add_row(
        "Test",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} → {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    rprint("\n[bold green]Training model...[/]\n")
    if MODEL_TO_USE.lower() == "mlp":
        model = MLP(
            input_size=train_features.shape[1],
            hidden_size=HIDDEN_SIZE,
        )

        # fit() calls fit_model(), saves the model, validates it, and saves metadata
        model.fit(
            X=train_features.values,
            y=train_labels["label"].values,  # type: ignore[union-attr]
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            verbose=True,
        )
    elif MODEL_TO_USE == "logreg":
        model = LogisticRegression(
            input_size=train_features.shape[1],
            max_iter=MAX_ITER,
            use_pca=USE_PCA,
        )
        model.fit(
            X=train_features.values,
            y=train_labels["label"].values,  # type: ignore[union-attr]
            verbose=True,
        )

    elif MODEL_TO_USE == "eegnet":
        # Load our pretrained EEGNet model weights from the model.pt file
        model = EEGNet.load()
        # Do not train the model again, we already have a pretrained model
        # model.fit(
        #     X=train_features.values,
        #     y=train_labels["label"].values,  # type: ignore[union-attr]
        #     verbose=True,
        # )
    
    # elif MODEL_TO_USE == "chronos":
    #     model = ChronosClassifier(
    #         input_size=train_features.shape[1], 
    #         # Trying settings that worked well for EEGNet
    #         window_size=1600, 
    #         projected_channels=32, 
            
    #     )
    #     model.fit(
    #         X=train_features.values,
    #         y=train_labels["label"].values, 
    #         epochs=EPOCHS,
    #         batch_size=BATCH_SIZE,
    #         learning_rate=LEARNING_RATE,
    #         verbose=True,
    #     )

    rprint("\n[bold green]Evaluating model on test set...[/]\n")
    # NOTE we use validation_features and labels because the test set is held out and not accessible for local evaluation.
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],  # type: ignore[union-attr]
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
