import pandas as pd
import numpy as np
from rlbook.bandits import EpsilonGreedy, init_optimistic, Bandit, init_zero
from rlbook.testbeds import NormalTestbed
from experiments.plotters import steps_violin_plotter
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
local_logger = logging.getLogger("experiment")
INIT = {"init_optimistic": init_optimistic, "init_zero": init_zero}
BANDIT = {"EpsilonGreedy": EpsilonGreedy}

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


def log_scalars(df, logger, plot: str, column: str, series_name: str):
    df = average_runs(df)
    df.apply(
        lambda x: logger.report_scalar(
            title=plot,
            series=series_name,
            value=x[column],
            iteration=x.step,
        ),
        axis=1,
    )


@hydra.main(config_path="configs/bandits", config_name="defaults")
def main(cfg: DictConfig):

    testbed = instantiate(cfg.testbed)
    bandit = instantiate(cfg.bandit, Q_init=call(cfg.Q_init, testbed))
    local_logger.info(f"Running bandit: {cfg.run}")
    local_logger.debug(f"Testbed expected values: {testbed.expected_values}")
    local_logger.debug(f"bandit config: {cfg['bandit']}")
    local_logger.debug(f"run config: {cfg['run']}")
    bandit.run(testbed, **OmegaConf.to_container(cfg.run))

    df_ar = bandit.output_df()
    df_ar = optimal_action(df_ar)
    local_logger.debug(f"{df_ar[['run', 'step', 'action', 'reward']].head(15)}")

    if cfg.upload:
        local_logger.info(f"Uploading to clearml")
        task_name = f"{cfg['task']} - {cfg.bandit['Q_init']}, alpha: {cfg.bandit['alpha']}, e: {cfg.bandit['epsilon']} | testbed - p_drift: {cfg.normal_testbed['p_drift']}"
        task = Task.init(project_name=cfg["project"], task_name=task_name)
        task.add_tags(
            [
                f"{cfg.bandit['type']}",
                f"{cfg.bandit['Q_init']}",
                f"alpha: {cfg.bandit['alpha']}",
                f"p_drift: {cfg.normal_testbed['p_drift']}",
            ]
        )
        remote_logger = task.get_logger()

        for i in range(5):
            fig = steps_violin_plotter(df_ar, testbed, run=i)
            remote_logger.report_matplotlib_figure(
                figure=fig,
                title="Action Rewards across a single run",
                series="single_run",
                iteration=i,
                report_image=True,
            )

        log_scalars(
            df_ar,
            remote_logger,
            "Average Reward",
            "average_reward",
            "average_reward",
        )
        log_scalars(
            df_ar,
            remote_logger,
            "% Optimal Action",
            "optimal_action_percent",
            "optimal_action_percent",
        )
        task.close()


if __name__ == "__main__":
    main()


