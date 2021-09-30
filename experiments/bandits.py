import pandas as pd
import numpy as np
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict
import plotnine as p9
from aim import Session
from pathlib import Path
import os


os.environ["AIM_UI_TELEMETRY_ENABLED"] = "0"
local_logger = logging.getLogger("experiment")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def steps_violin_plotter(df_ar, testbed, run=0):
    df_estimate = testbed.estimate_distribution(1000)
    df_estimate = df_estimate.astype({"action": "int32"})
    df_ar = df_ar.loc[df_ar["run"] == run]
    df_ar = df_ar.astype({"action": "int32"})
    p = (
        p9.ggplot(
            p9.aes(
                x="reorder(factor(action), action)",
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
    df["optimal_action_percent"] = df["step"].map(
        df.groupby(["step"])["optimal_action_true"].sum() / (df["run"].max() + 1)
    )

    return df


def write_scalars(df, session, column: str, tag: str, hp: dict):
    """Write scalars to local using aim

    Return
        Value of last step
    """
    df = average_runs(df)
    df.apply(
        lambda x: session.track(
            x[column],
            epoch=int(x.step),
            name=tag,
        ),
        axis=1,
    )

    return df[column].iloc[-1]


@hydra.main(config_path="configs/bandits", config_name="defaults")
def main(cfg: DictConfig):
    session = Session(
        repo=(Path.home() / "projects/rlbook/experiments/outputs/bandit").as_posix(),
        experiment=cfg.bandit["_target_"].split(".")[-1],
    )

    testbed = instantiate(cfg.testbed)
    bandit = instantiate(cfg.bandit, Q_init=call(cfg.Q_init, testbed))

    local_logger.info(f"Running bandit: {cfg.run}")
    local_logger.debug(f"Testbed expected values: {testbed.expected_values}")
    local_logger.debug(f"bandit config: {cfg['bandit']}")
    local_logger.debug(f"run config: {cfg['run']}")
    session.set_params(OmegaConf.to_container(cfg.run), "experiment")
    bandit.run(testbed, **OmegaConf.to_container(cfg.run))

    df_ar = bandit.output_df()
    df_ar = optimal_action(df_ar)
    local_logger.debug(f"\n{df_ar[['run', 'step', 'action', 'optimal_action', 'reward']].head(15)}")

    bandit_type = cfg.bandit._target_.split(".")[-1]
    Q_init = cfg.Q_init._target_.split(".")[-1]

    task_name = f"{bandit_type} - " + ", ".join(
        [
            f"{k}: {OmegaConf.select(cfg, v).split('.')[-1]}"
            if isinstance(OmegaConf.select(cfg, v), str)
            else f"{k}: {OmegaConf.select(cfg, v)}"
            for k, v in cfg.task_labels.items()
        ]
    )
    local_logger.debug(f"{task_name}")

    hp_testbed = OmegaConf.to_container(cfg.testbed)
    hp = OmegaConf.to_container(cfg.bandit)
    hp["Q_init"] = cfg.Q_init._target_
    hp["p_drift"] = hp_testbed["p_drift"]
    session.set_params(hp, "hyperparameters")

    # for i in range(min(3, cfg.run.n_runs)):
    #     fig = steps_violin_plotter(df_ar, testbed, run=i)
    #     writer.add_figure(f"run{i}", fig, global_step=cfg.run.steps)

    final_avg_reward = write_scalars(df_ar, session, "reward", "average_reward", hp)

    final_optimal_action = write_scalars(
        df_ar, session, "optimal_action_percent", "optimal_action_percent", hp
    )
    final_metrics = {
        "average_reward": final_avg_reward,
        "optimal_action_percent": final_optimal_action,
    }
    session.set_params(final_metrics, "final_metrics")
    local_logger.debug(f"final_metrics: {final_metrics}")


if __name__ == "__main__":
    main()
