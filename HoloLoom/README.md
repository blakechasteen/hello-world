holoLoom - it's TS all the way down!
## Running train_agent

Quick instructions to run the example trainer shipped with HoloLoom. It's recommended to use the project's virtual environment.

1. Create and activate a virtualenv (macOS / zsh):

```bash
# create a venv in the repository root
python3 -m venv .venv
source .venv/bin/activate
# upgrade pip and install minimal deps
pip install --upgrade pip
pip install torch numpy gymnasium matplotlib
```

2. Run a short trainer session (example uses CartPole):

```bash
# from the repository root
PYTHONPATH=HoloLoom .venv/bin/python -c "from HoloLoom.train_agent import PPOTrainer; t=PPOTrainer(env_name='CartPole-v1', total_timesteps=2000, steps_per_update=256, n_epochs=1, batch_size=32, log_dir='./logs/test_run_small'); t.train()"
```

3. Notes
- The trainer saves checkpoints under the `log_dir` passed to `PPOTrainer`.
- For longer experiments, adjust `total_timesteps`, `steps_per_update`, and `hidden_dims.

```