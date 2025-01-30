# Installation

## Install uv (for linux below):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or with pip
```
# With pip.
pip install uv
```
[Link to instructions for other OS's](https://docs.astral.sh/uv/getting-started/installation/)  

  
## Install the rlbook environment via uv:
```bash
uv sync --extra cpu
```

or if you have a gpu available:
```bash
uv sync --extra gpu
```

## Run commands using the rlbook environment via uv:
```bash
uv run run.py
```
or by first activating the rlbook venv (this is my preferred workflow):
```bash
source ./venv/bin/activate
```

## (Optional) Setup wandb for experiment tracking
- Sign up for an account at wandb: https://app.wandb.ai/login?signup=true  

- Copy the api key from: https://wandb.ai/authorize  

- Login to wandb via:  
```bash
wandb login
```