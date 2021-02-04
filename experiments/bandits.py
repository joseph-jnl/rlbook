import pandas as pd
import numpy as np
from rlbook.bandits import EpsilonGreedy, init_optimistic, Bandit, init_zero
from rlbook.testbeds import NormalTestbed
from experiments.plotters import steps_plotter, reward_plotter
import plotnine as p9
from clearml import Task
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict


# clearml frontend currently has a bug that truncates "." and treats
# certain series as identical. Hack: prefix invisible unicode characters to differentiate
COLORS = [
    "\u2000",
    "\u2002",
    "\u2004",
    "\u2006",
    "\u2008",
]

INIT = {"init_optimistic": init_optimistic, "init_zero": init_zero}


def average_runs(df, group=[]):
    """Average all dataframe columns across runs

    Attributes:
        group (list): Additional list of columns to group by before taking the average

    """
    return df.groupby(["step"] + group).mean().reset_index()


def optimal_action(df, group=[]):
    """Create new column "optimal_action_percent"

    Attributes:
        group (list):
            Additional list of columns to group by before calculating percent optimal action

    """
    df["optimal_action_true"] = np.where(df["action"] == df["optimal_action"], 1, 0)
    df["running_optimal_action_true"] = df.groupby(["run"] + group)[
        "optimal_action_true"
    ].cumsum()
    df["optimal_action_percent"] = np.where(
        df["running_optimal_action_true"] == 0,
        0,
        df["running_optimal_action_true"] / (df["step"] + 1),
    )

    return df


def log_scalars(df, plot: str, column: str, series_name: str):
    task = Task.init(project_name="rlbook", task_name="bandit")
    logger = task.get_logger()
    df.apply(
        lambda x: logger.report_scalar(
            title=plot,
            series=series_name,
            value=x[column],
            iteration=x.step,
        ),
        axis=1,
    )


@hydra.main(config_path="configs/bandits", config_name="test")
def main(cfg: DictConfig):

    testbed = instantiate(cfg.normal_testbed)
    logging.debug(f"Testbed expected values: {testbed.expected_values}")
    bandits = {
        e: EpsilonGreedy(
            testbed,
            epsilon=e,
            alpha=cfg.bandit["alpha"],
            Q_init=INIT[cfg.bandit["Q_init"]],
        )
        for e in cfg.bandit["epsilons"]
    }

    for b in bandits.values():
        b.run(**OmegaConf.to_container(cfg.run))

    logging.debug(f"{b.Q_init}")
    df_ar = pd.concat([b.output_df() for b in bandits.values()]).reset_index(
        drop=True
    )
    df_ar = optimal_action(df_ar, group=["epsilon"])
    df_ar = average_runs(df_ar, group=["epsilon"])
    logging.debug(f"{df_ar}")

    if cfg.upload:
        for i, e in enumerate(bandits.keys()):
            log_scalars(
                df_ar.loc[df_ar["epsilon"] == e],
                "Average Reward",
                "average_reward",
                f"{COLORS[i]}epsilon: {e}",
            )
            log_scalars(
                df_ar.loc[df_ar["epsilon"] == e],
                "% Optimal Action",
                "optimal_action_percent",
                f"{COLORS[i]}epsilon: {e}",
            )


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
