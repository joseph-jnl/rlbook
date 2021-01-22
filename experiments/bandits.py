import pandas as pd
import numpy as np
from rlbook.bandits import EpsilonGreedy, init_optimistic
from rlbook.testbeds import NormalTestbed
from experiments.plotters import steps_plotter, reward_plotter
import plotnine as p9
from clearml import Task
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


N = 2000
RUNS = 10
EPSILONS = [0, 0.01, 0.1]
EXPECTED_VALUES = {
    1: {"mean": 0.5, "var": 1},
    2: {"mean": -1, "var": 1},
    3: {"mean": 2, "var": 1},
    4: {"mean": 1, "var": 1},
    5: {"mean": 1.7, "var": 1},
    6: {"mean": -2, "var": 1},
    7: {"mean": -0.5, "var": 1},
    8: {"mean": -1, "var": 1},
    9: {"mean": 1.5, "var": 1},
    10: {"mean": -1, "var": 1},
}

testbed = NormalTestbed(EXPECTED_VALUES)
testbed_drift = NormalTestbed(EXPECTED_VALUES, p_drift=1.0)


def print_scalar(bandit):
    pass


def run_bandits(bandits, run_config, plots={}):

    for b in bandits.values():
        b.run(**run_config)

    df_ar = pd.concat([b.output_df() for b in bandits.values()]).reset_index(drop=True)

    return df_ar

@hydra.main(config_path="configs/bandits", config_name="test")
def main(cfg: DictConfig):
    if cfg.upload:
        task = Task.init(project_name='rlbook', task_name='bandit')
    print(OmegaConf.to_yaml(cfg))
    testbed = instantiate(cfg.normal_testbed)
    print(testbed.expected_values)
    # bandit = EpsilonGreedy(testbed, *instantiate(cfg.normal_testbed))
    

if __name__ == "__main__":
    main()


EXPERIMENTS = {
    # "Single Run": {
    #     f"Single Run - Epsilon greedy bandit - 0 init": {
    #         "fx": steps_plotter,
    #         "config": (
    #             {e: EpsilonGreedy(testbed, epsilon=e, print_scalars=True) for e in EPSILONS},
    #             {"steps": N, "n_runs": 1, "n_jobs": 1},
    #             testbed,
    #         ),
    #     },
    # },
    # "0 initialization": {
    #     f"{RUNS} Runs - Epsilon greedy bandit - 0 init - varying 1/N step size alpha": {
    #         "fx": reward_plotter,
    #         "config": (
    #             {e: EpsilonGreedy(testbed, epsilon=e, alpha=None, print_scalars=True) for e in EPSILONS},
    #             {"steps": N, "n_runs": RUNS, "n_jobs": 8,
    #             },
    #         ),
    #     },
        # f"{RUNS} Runs - Epsilon greedy bandit - 0 init - constant step size": {
        #     "fx": reward_plotter,
        #     "config": (
        #         {e: EpsilonGreedy(testbed, epsilon=e, print_scalars=True) for e in EPSILONS},
        #         {"steps": N, "n_runs": RUNS, "n_jobs": 8},
        #     ),
        # },
        # f"{RUNS} Runs - Drifting Testbed - Epsilon greedy bandit - 0 init - constant step size": {
        #     "fx": reward_plotter,
        #     "config": (
        #         {e: EpsilonGreedy(testbed_drift, epsilon=e, print_scalars=True) for e in EPSILONS},
        #         {"steps": N, "n_runs": RUNS, "n_jobs": 8},
        #     ),
        # },
    # },
    # "optimistic initialization": {
    #     f"{RUNS} Runs - Epsilon greedy bandit - optimistic init - varying 1/N step size alpha": {
    #         "fx": reward_plotter,
    #         "config": (
    #             {e: EpsilonGreedy(testbed, epsilon=e, alpha=None, Q_init=init_optimistic) for e in EPSILONS},
    #             {"steps": N, "n_runs": RUNS, "n_jobs": 8,
    #             },
    #         ),
    #     },
    #     f"{RUNS} Runs - Epsilon greedy bandit - optimistic init - constant step size": {
    #         "fx": reward_plotter,
    #         "config": (
    #             {e: EpsilonGreedy(testbed, epsilon=e, Q_init=init_optimistic) for e in EPSILONS},
    #             {"steps": N, "n_runs": RUNS, "n_jobs": 8},
    #         ),
    #     },
    #     f"{RUNS} Runs - Drifting Testbed - Epsilon greedy bandit - optimistic init - constant step size": {
    #         "fx": reward_plotter,
    #         "config": (
    #             {e: EpsilonGreedy(testbed_drift, epsilon=e, Q_init=init_optimistic) for e in EPSILONS},
    #             {"steps": N, "n_runs": RUNS, "n_jobs": 8},
    #         ),
    #     },
    # },
}

