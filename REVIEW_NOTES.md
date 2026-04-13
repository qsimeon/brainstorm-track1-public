# REVIEW_NOTES — mind_meld
Date: 2026-04-13
Iteration goal: First run — orient to project, create RALPH.md with thorough analysis
Outcome: ✅ achieved

Work done:
- Performed comprehensive deep-dive of entire codebase: read base.py, eegnet.py, qsimeon_ema_net.py, metrics.py, evaluation.py, all docs, examples
- Identified current state: EEGNet trained and deployed, EMANet has bugs and training bottleneck
- Discovered critical bug in QSimeonEMANet.predict(): `torch.argmax(logits, dim=1)` on 1D tensor will throw IndexError — should be `dim=0` or no dim arg
- Identified EMALayer.forward() bottleneck: Python for-loop over L=1600 timesteps, each calling Linear(64,4096) + Gumbel-Softmax. Should vectorize logit computation across timesteps.
- Identified scoring insight: small models (<1MB) score significantly better due to exponential size penalty
- Documented all key facts in RALPH.md (project goal, constraints, scoring, architecture, bugs, roadmap)

Blockers:
- No training data downloaded locally (./data/ doesn't exist in worktree) — needed for any training/eval
- Cannot run training without data or GPU access

Next iteration: **Fix QSimeonEMANet bugs + implement SpectralBandPowerModel**

Specific tasks for next agent:
1. Fix `brainstorm/ml/qsimeon_ema_net.py` predict() bug: line ~550 `torch.argmax(logits, dim=1)` → `torch.argmax(logits)` (logits is 1D after `outputs[0, -1, :]`)
2. Vectorize `EMALayer.forward()` to compute all timestep logits in one batch matmul, keeping only the sequential EMA recurrence loop (which is necessary due to state dependency)
3. Create `brainstorm/ml/spectral_band_power.py` — a new SpectralBandPowerModel that:
   - Maintains a causal circular buffer of past N samples (e.g., 100ms = 100 samples)
   - Computes FFT on that buffer to get band power in key bands (matching stimulus frequencies: 120, 224, 421, 789, 1479, 2772, 5195, 9736 Hz)
   - Uses a small sklearn LogisticRegression or tiny Linear layer for classification
   - Target: model file <500KB, balanced accuracy competitive with EEGNet
4. Add a benchmark/compare script or update `examples/compare_models.py` to run both models on validation data and report all 3 scoring metrics

Delegate steps 1-3 to /codex:rescue with context: "BrainStorm 2026 BCI. Do not modify base.py, evaluation.py, metrics.py. Causal streaming inference. Use uv."

Completion: 38%
