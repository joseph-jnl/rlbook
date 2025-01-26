# rlbook

![ci-cd](https://github.com/joseph-jnl/rlbook/actions/workflows/ci-cd.yml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Code for my walkthrough of: *Reinforcement Learning An Introduction by Richard Sutton and Andrew Barto* (http://incompleteideas.net/book/the-book.html) 

## Setup

### Install uv (for linux below):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or with pip
```
# With pip.
pip install uv
```
[Link to instructions for other OS's](https://docs.astral.sh/uv/getting-started/installation/)  

  

### Install the rlbook environment via uv:
```bash
uv sync --extra cpu
```

or if you have a gpu:
```bash
uv sync --extra gpu
```

Run commands using the rlbook environment via uv:
```bash
uv run run.py
```
or by first activating the rlbook venv (this is my preferred workflow):
```bash
source ./venv/bin/activate
```

### (Optional) Setup wandb for experiment tracking
Sign up for an account at wandb: https://app.wandb.ai/login?signup=true  

Copy the api key from: https://wandb.ai/authorize  

Login to wandb via:  
   ```bash
   wandb login
```

## Quickstart
Algorithm implementations are located in the `/src` directory while the scaffolding code/notebooks for recreating/exploring Sutton & Barto are segmented into the `experiments/` directory.  

e.g. for recreating Figure 2.3, navigate to `/experiments/ch2_bandits/` and run:
```bash
python run.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0,0.01,0.1 +bandit.random_argmax=true experiment.tag=fig2.2 experiment.upload=true
```

![image](https://github.com/user-attachments/assets/5ca29806-0cdd-4f79-9a6d-08aa1ca72417)
Figure 2.3 (rlbook): The `+bandit.random_argmax=true` flag was used to switch over to an argmax implementation that randomizes between tiebreakers rather than first occurence used in the default numpy implementation to better align with the original example.
[Link to wandb artifact](https://api.wandb.ai/links/josephjnl/53gxgbcc)

Further details on experimental setup and results can be found at corresponding chapter README's.

## Chapter Links

- [Chapter 2: Multi-armed Bandits](https://github.com/joseph-jnl/rlbook/tree/dev/experiments/ch2_bandits)
