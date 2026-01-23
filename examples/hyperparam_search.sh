#!/bin/bash
#
# Hyperparameter search for EEGNet
# Tests different window sizes and PCA projected channels
#
# Usage:
#   ./examples/hyperparam_search.sh
#
# Results are saved to /media/M2SSD/mind_meld_checkpoints/hypersearch/
#

set -e

# Base output directory
OUTPUT_DIR="/media/M2SSD/mind_meld_checkpoints/hypersearch"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/hypersearch_results.log"
echo "=== Hyperparameter Search Started: $(date) ===" | tee "$LOG_FILE"

# Hyperparameter values to test
WINDOW_SIZES=(32 64 128 256)
PCA_CHANNELS=(32 64 128)

# Fixed parameters
EPOCHS=15
BATCH_SIZE=64

# Run experiments
for window in "${WINDOW_SIZES[@]}"; do
    for pca in "${PCA_CHANNELS[@]}"; do
        exp_name="window${window}_pca${pca}"
        exp_dir="$OUTPUT_DIR/$exp_name"
        mkdir -p "$exp_dir"

        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "Experiment: $exp_name" | tee -a "$LOG_FILE"
        echo "  Window Size: $window" | tee -a "$LOG_FILE"
        echo "  PCA Channels: $pca" | tee -a "$LOG_FILE"
        echo "  Started: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"

        # Run training
        python examples/train_eegnet_hypersearch.py \
            --window-size "$window" \
            --projected-channels "$pca" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --checkpoint-dir "$exp_dir" \
            2>&1 | tee "$exp_dir/training.log"

        echo "  Completed: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

echo "=== Hyperparameter Search Completed: $(date) ===" | tee -a "$LOG_FILE"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary log: $LOG_FILE"
