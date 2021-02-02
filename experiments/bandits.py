import pandas as pd
from rlbook.bandits import EpsilonGreedy, init_optimistic, Bandit
from rlbook.testbeds import NormalTestbed
from experiments.plotters import steps_plotter, reward_plotter
import plotnine as p9
from clearml import Task
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict


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
        e: EpsilonGreedy(testbed, epsilon=e, alpha=cfg.bandit["alpha"])
        for e in cfg.bandit["epsilons"]
    }
    logging.debug(f"alphas: {[(e, b.alpha) for e, b in bandits.items()]}")
    for b in bandits.values():
        b.run(**OmegaConf.to_container(cfg.run))
    if cfg.upload:
        df_ar = (
            pd.concat([b.output_df() for b in bandits.values()])
            .reset_index(drop=True)
            .groupby(["step", "epsilon"])
            .mean("average_reward")
            .reset_index()
        )
        shapes = [
            "\u25A0",
            "\u25B2",
            "\u25CF",
            "\u25A7",
            "\u25BC",
        ]
        for i, e in enumerate(bandits.keys()):
            log_scalars(
                df_ar.loc[df_ar["epsilon"] == e],
                "Average Reward",
                "average_reward",
                f"{shapes[i]} epsilon: {e}",
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
