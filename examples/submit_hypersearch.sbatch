#!/bin/bash
#SBATCH --job-name=eegnet_hypersearch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-119%8
#SBATCH --output=hypersearch_results/slurm_%A_%a.out
#SBATCH --error=hypersearch_results/slurm_%A_%a.err

# EEGNet Hyperparameter Search - SLURM Array Job
#
# Runs 120 configurations in parallel, 8 at a time
# Each job takes ~2min with GPU, so total ~30min wall time
#
# Usage:
#   sbatch examples/submit_hypersearch.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f hypersearch_results/slurm_*.out
#
# After completion, collect results:
#   python examples/collect_results.py

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Change to project directory
cd /home/denny-loevlie/mind_meld

# Create results directory
mkdir -p hypersearch_results

# Activate environment if needed (uncomment and modify as needed)
# source ~/.bashrc
# conda activate myenv
# source venv/bin/activate

# Run the experiment for this array task
python examples/slurm_hypersearch.py --config-id $SLURM_ARRAY_TASK_ID

echo "Job $SLURM_ARRAY_TASK_ID completed at $(date)"
