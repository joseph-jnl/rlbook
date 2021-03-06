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
import plotnine as p9
from tensorboardX import SummaryWriter


local_logger = logging.getLogger("experiment")


def steps_violin_plotter(df_ar, testbed, run=0):
    df_estimate = testbed.estimate_distribution(1000)
    df_estimate = df_estimate.astype({"action": "int32"})
    df_ar = df_ar.loc[df_ar["run"] == run]
    df_ar = df_ar.astype({"action": "int32"})
    p = (
        p9.ggplot(
            p9.aes(
                x="factor(action)",
                y="reward",
            )
        )
        + p9.ggtitle(f"Action - Rewards across {df_ar.shape[0]} steps")
        + p9.xlab("k-arm")
        + p9.ylab("Reward")
        + p9.geom_violin(df_estimate, fill="#d0d3d4")
        + p9.geom_jitter(df_ar, p9.aes(color="step"))
        + p9.theme(figure_size=(20, 9))
    )
    fig = p.draw()

    return fig


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


def upload_scalars(df, logger, plot: str, column: str, series_name: str, **kwargs):
    """Upload scalars to clearml"""
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


def write_scalars(df, writer, column: str, tag: str):
    """Write scalars to local using tensorboardX"""
    df = average_runs(df)
    df.apply(
        lambda x: writer.add_scalar(
            tag,
            x[column],
            x.step,
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
        bandit_type = cfg.bandit._target_.split(".")[-1]
        Q_init = cfg.Q_init._target_.split(".")[-1]

        local_logger.info(f"Uploading to clearml")
        task_name = f"{bandit_type} - " + ", ".join(
            [
                f"{k}: {OmegaConf.select(cfg, v).split('.')[-1]}"
                if isinstance(OmegaConf.select(cfg, v), str)
                else f"{k}: {OmegaConf.select(cfg, v)}"
                for k, v in cfg.task_labels.items()
            ]
        )
        local_logger.debug(f"{cfg['project']}: {task_name}")
        task = Task.init(
            project_name=cfg["project"],
            task_name=task_name,
            auto_connect_arg_parser=False,
        )

        tags = [f"{bandit_type}"] + [
            f"{k}: {OmegaConf.select(cfg, v).split('.')[-1]}"
            if isinstance(OmegaConf.select(cfg, v), str)
            else f"{k}: {OmegaConf.select(cfg, v)}"
            for k, v in cfg.tags.items()
        ]
        task.add_tags(tags)

        remote_logger = task.get_logger()
        config = task.connect_configuration(
            OmegaConf.to_container(cfg.testbed), name="testbed parameters"
        )
        writer = SummaryWriter(comment="bandit")
        parameters = task.connect(OmegaConf.to_container(cfg.bandit))
        parameters["Q_init"] = cfg.Q_init._target_

        for i in range(5):
            fig = steps_violin_plotter(df_ar, testbed, run=i)
            remote_logger.report_matplotlib_figure(
                figure=fig,
                title="Action Rewards across a single run",
                series="single_run",
                iteration=i,
                report_image=True,
            )

        # upload_scalars(
        #     df_ar,
        #     remote_logger,
        #     "Average Reward",
        #     "reward",
        #     "reward",
        # )
        # upload_scalars(
        #     df_ar,
        #     remote_logger,
        #     "% Optimal Action",
        #     "optimal_action_percent",
        #     "optimal_action_percent",
        # )

        write_scalars(
            df_ar,
            writer,
            "reward",
            "average_reward",
        )
        write_scalars(
            df_ar,
            writer,
            "optimal_action_percent",
            "optimal_action_percent",
        )

        task.close()


if __name__ == "__main__":
    main()
