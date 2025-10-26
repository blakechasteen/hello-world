# Release v1.0.1 — Draft PR

This PR prepares release `v1.0.1`.

## Summary of changes
- Finalized policy fixes in `HoloLoom/policy/unified.py`:
  - Ensure policy outputs include `value` for categorical/gaussian modes
  - Added `get_value` and deterministic sampling support in `sample_action`
- Added shared types and embeddings used by the policy (tests): `HoloLoom/embedding/spectral.py`
- Made `HoloLoom` a proper package with `HoloLoom/__init__.py` and fixed package imports
- Added `HoloLoom/README.md` instructions for running the example trainer (`train_agent`)
- Added `.gitignore` to exclude virtualenv, editor, and log artifacts

## Notable behavior
- The test harness `HoloLoom/test_unified_policy.py` runs successfully (18/18 tests passed in the local environment used for development).
- A short training run on `CartPole-v1` (2k timesteps) completes successfully and saves a checkpoint under `logs/test_run_small/checkpoints/final.pt`.

## How to test locally
1. Create and activate a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy gymnasium matplotlib
```

2. Run the unit test harness:

```bash
PYTHONPATH=HoloLoom .venv/bin/python HoloLoom/test_unified_policy.py
```

3. Run a short trainer session:

```bash
PYTHONPATH=HoloLoom .venv/bin/python -c "from HoloLoom.train_agent import PPOTrainer; t=PPOTrainer(env_name='CartPole-v1', total_timesteps=2000, steps_per_update=256, n_epochs=1, batch_size=32, log_dir='./logs/test_run_small'); t.train()"
```

## Follow-ups / Recommendations
- Consider replacing test-focused stubs with production implementations for ICM/RND/PPO when moving to production.
- Add CI to run `test_unified_policy.py` on PRs and create a release workflow for v1.0.1 (a workflow for v1.0.0 exists; can be adapted).

