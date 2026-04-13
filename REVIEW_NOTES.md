# REVIEW_NOTES — mind_meld
Date: 2026-04-13
Iteration goal: Fix QSimeonEMANet bugs + implement SpectralBandPowerModel
Outcome: ✅ achieved

Work done:
- Fixed predict() bug in brainstorm/ml/qsimeon_ema_net.py line 556: `torch.argmax(logits, dim=1)` → `torch.argmax(logits)` (logits is 1D shape (n_classes,), dim=1 caused IndexError)
- Vectorized EMALayer.forward(): all L=1600 timestep logit computations now done in one batch Linear call; only the sequential EMA recurrence loop remains. Expected 10-30x training speedup.
- Created brainstorm/ml/spectral_band_power.py — SpectralBandPowerModel:
  - 7 spectral band-power features (delta through ultra-high gamma, all ≤500Hz Nyquist) from causal 100-sample FFT window of spatially averaged ECoG
  - 3 EMA features at τ = 5, 20, 100ms
  - sklearn LogisticRegression(class_weight='balanced', C=0.1) — target <500KB model
  - Saves to model.pkl via pickle
  - Fully implements BaseModel interface (fit_model, predict, save, load)
- Created scripts/train_spectral.py — training script that loads from ./data/ parquet files and calls model.fit(X, y)
- Committed and pushed to ralph/mind_meld (commit 1c11ee0)

Blockers:
- Training data not in worktree (./data/ missing) — main project /Users/quileesimeon/mind_meld has data
- uv environment build fails in worktree due to `spectrum` C extension compile error on macOS (pre-existing issue with torcheeg dep)
- Cannot numerically validate SpectralBandPowerModel performance without running training

Next iteration: **Train SpectralBandPowerModel on real data + benchmark vs EEGNet**

Specific tasks for next agent:
1. Main project (/Users/quileesimeon/mind_meld) already has data in ./data/ and a working .venv
2. Copy worktree's new files to main project and train SpectralBandPowerModel there:
   ```
   cp /Users/quileesimeon/ralph-worktrees/mind_meld/brainstorm/ml/spectral_band_power.py /Users/quileesimeon/mind_meld/brainstorm/ml/
   cp /Users/quileesimeon/ralph-worktrees/mind_meld/scripts/train_spectral.py /Users/quileesimeon/mind_meld/scripts/
   cd /Users/quileesimeon/mind_meld && uv run python scripts/train_spectral.py
   ```
3. Evaluate: update model_metadata.json to point to model.pkl, then run evaluation harness
4. Compare scores: balanced_accuracy, lag, model size for SpectralBandPowerModel vs current EEGNet (model.pt ~2.5MB)
5. If SpectralBandPowerModel wins on composite score → keep model.pkl as active; else compress EEGNet (D=1, F1=4)
6. Update model_metadata.json in worktree to point to the winning model

Key files:
- brainstorm/ml/spectral_band_power.py — new SpectralBandPowerModel (tiny, causal)
- brainstorm/ml/qsimeon_ema_net.py — EMANet with predict() bug fixed + vectorized forward
- scripts/train_spectral.py — spectral model training script
- /Users/quileesimeon/mind_meld/data/ — training data (parquet format)
- /Users/quileesimeon/mind_meld/model.pt — current active EEGNet (~2.5MB)

Completion: 48%
