# RALPH.md — BrainStorm 2026 BCI Track 1 Decoder

## Project Goal (1 sentence)
Build the highest-scoring streaming neural decoder for auditory stimulus classification from 1024-channel ECoG recordings, balancing accuracy (50%), prediction lag (25%), and model size (25%) within hard constraints.

## Deliverable Type
Python model class inheriting `BaseModel`, saved as `model.pt` (or `model.pkl`) within the repo, registered in `model_metadata.json`. Submitted to BrainStorm 2026 hackathon for remote evaluation on a private test set.

## Audience
BrainStorm 2026 hackathon judges. The remote evaluation harness calls `model.predict(X)` once per 1ms timestep. Results are scored on a 0–100 scale.

## Scoring Formula (Critical)
```
score = 50 * balanced_accuracy
      + 25 * exp(-6 * avg_lag_samples / 500)   # <100ms lag → ~20/25 pts
      + 25 * exp(-4 * model_size_mb / 5)        # <1MB → ~20/25 pts; 5MB → ~25% pts
```
- Hard limit: model file >25MB → rejected entirely
- At 1MB size: size_score ≈ 17.3/25 (exp(-0.8) * 25)
- At 10MB size: size_score ≈ 6.6/25 (exp(-8) * 25)
- **Strategic implication**: a tiny model (1MB) scoring 60% balanced accuracy outscores a large model (10MB) scoring 80% accuracy by ~5 points

## Success Criteria
- Final score ≥ 70/100 on the leaderboard
- Model file ≤ 5MB (ideally ≤ 1MB)
- Average prediction lag ≤ 100ms (100 samples at 1kHz)
- Balanced accuracy ≥ 50% (10 classes, 67% are class 0)

## Problem Specification
**Task**: Streaming 9-class + silence (10 total) classification  
**Input**: `(1024,)` raw voltage from micro-ECoG array, one sample at a time  
**Output**: Integer label = stimulus frequency in Hz (0=silence, or 120/224/421/789/1479/2772/5195/9736)  
**Sampling rate**: 1000 Hz  
**Data**: Small dataset (few minutes) from swine auditory cortex; heavy class imbalance (~67% silence)  
**Causal constraint**: predict() called sequentially, cannot use future data; maintain internal state

## Design Philosophy
- **Tiny & accurate** beats **huge & accurate**: the scoring exponentially punishes large models
- **Spectral features** likely dominate (band power in auditory frequency bands → class 0–9736Hz)
- Keep own state in the model (sliding window buffer, EMA state) for causal streaming
- PCA projection 1024→64 is a reasonable first-stage compression before any temporal model
- Class imbalance (67% silence) must be handled: use `balanced_accuracy_score`, class-weighted loss

## Constraints (Hard Rules)
- **Do NOT modify**: `brainstorm/ml/base.py`, `brainstorm/evaluation.py`, `brainstorm/ml/metrics.py`
- **Model file** must be inside the repo directory (evaluation can only access repo files)
- **Model file** must be ≤ 25MB (hard limit in `validate_model_loadable()`)
- Must implement: `fit_model()`, `predict()`, `save()`, `load()` inheriting from `BaseModel`
- Non-causal models forbidden (no bidirectional filters, no future data)
- Use `uv run python` for all Python execution (`conda activate work_env` + `uv`)

## Current State: ~38% Done

### What's Complete ✅
- **Core framework** (100%): BaseModel, evaluation harness, metrics, data loading — locked, don't touch
- **EEGNet** (70%): Trained on full dataset (train+validation), registered as active model in `model_metadata.json`. Architecture: PCA(1024→64) + EEGNetCore (depthwise separable conv). Default params: F1=8, D=2, window=1600ms. Checkpoints in `checkpoints/`. Inference: sliding window buffer, CPU only.
- **MLP** (100%): Simple baseline, functional, too slow/large for competitive score
- **LogisticRegression** (100%): Sklearn baseline with optional PCA, functional
- **Documentation** (90%): Extensive docs in `mindmeld_docs/` and `docs/`

### What's Incomplete / Broken ❌
- **QSimeonEMANet has a critical bug**: In `predict()` (line 550 of `brainstorm/ml/qsimeon_ema_net.py`):
  ```python
  logits = outputs[0, -1, :]  # shape: (n_classes,) — 1D tensor
  predicted_idx = int(torch.argmax(logits, dim=1).item())  # BUG: dim=1 on 1D tensor!
  ```
  Fix: `torch.argmax(logits, dim=0)` or `torch.argmax(logits)`.

- **EMANet training is O(window_size) Python iterations**: `EMALayer.forward()` loops `for t in range(L)` where L=1600. Each iteration does a Linear(64, 4096) + GumbelSoftmax + bmm. This is ~1600 Python-level ops per forward pass → very slow even with GPU.

- **EMANet input_logits layer is huge**: `Linear(64, 64*64)` = `Linear(64, 4096)`. This is 4096*64*4 bytes ≈ 1MB just for this layer. Stored 30 epochs would eat memory.

- **EMANet vectorization fix exists**: Can compute `input_logits` for all timesteps at once:
  ```python
  # Instead of looping:
  all_logits = self.input_logits(layer_norm_input(x))  # (B, L, ema_nodes * input_dim)
  all_logits = all_logits.view(B, L, self.ema_nodes, self.input_dim)
  all_masks = gumbel_softmax(all_logits)  # (B, L, ema_nodes, input_dim)
  # Still need sequential EMA recurrence, but logit computation is vectorized
  ```

- **No spectral feature model**: The hint in `docs/overview.md` says "spectral power changes" are key. A simple band-power + logistic regression could be tiny (<100KB) and fast.

- **No systematic evaluation results**: No saved val scores comparing models, no results directory

- **No model compression**: EEGNet likely >10MB; no quantization/pruning applied yet

## Architecture Files (Do Not Modify)
- `brainstorm/ml/base.py` — BaseModel interface
- `brainstorm/evaluation.py` — ModelEvaluator class  
- `brainstorm/ml/metrics.py` — Scoring functions

## Key Files for Development
- `brainstorm/ml/eegnet.py` — Current deployed model (EEGNet, ~10MB)
- `brainstorm/ml/qsimeon_ema_net.py` — EMA Net (has bugs, training bottleneck)
- `brainstorm/ml/channel_projection.py` — PCAProjection utility
- `brainstorm/loading.py` — Data loading (loads train/validation .parquet from `./data/`)
- `examples/example_local_train_and_evaluate.py` — Full pipeline demo
- `model_metadata.json` — Points to current deployed model
- `checkpoints/` — Saved model checkpoints

## Human Actions Needed
1. **Download data**: `uv run python -c "from brainstorm.download import download_train_validation_data; download_train_validation_data()"`  
   Data goes to `./data/` (train + validation parquet files). Required for any training/evaluation.
2. **SLURM cluster** (ssh engaging): For GPU training, submit SLURM jobs; source `~/.secrets` for HF tokens

## Codex Delegation Guide
For coding tasks, delegate to `/codex:rescue` with this context:
- "BrainStorm 2026 BCI hackathon. Model must inherit from `BaseModel` in `brainstorm/ml/base.py`. Do NOT modify base.py, evaluation.py, metrics.py. Use uv for Python deps. Causal streaming inference."
- **EMA Net fix**: Fix `predict()` bug (argmax dim), vectorize `EMALayer.forward()` to eliminate Python loop
- **Spectral model**: Implement a new `SpectralBandPowerModel` using causal FFT windows → band power features → small sklearn or tiny MLP classifier. Target <500KB model size.
- **EEGNet compression**: Reduce D=1, F1=4, try quantization with `torch.quantization` to get under 2MB

## Priority Roadmap
1. **Fix QSimeonEMANet bugs** (1 hour): Vectorize EMALayer + fix predict() argmax bug
2. **Implement SpectralBandPowerModel** (2-3 hours): Band power features + logistic regression, target <500KB
3. **Benchmark comparison**: Run both on validation set, record balanced_accuracy + lag + size scores
4. **Optimize winner**: Tune hyperparameters of best model, compress if needed
5. **Submit** best model (update model_metadata.json + commit + push)
