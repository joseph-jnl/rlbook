import logging
import time
from datetime import timedelta

import hydra
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from plotnine import aes, geom_jitter, geom_violin, ggplot, ggtitle, theme, xlab, ylab

local_logger = logging.getLogger("experiment")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def steps_violin_plotter(df_ar, testbed, run: int = 0):
    """Return plot of reward distribution overlayed with
    actions taken over a single run

    Args:
        df_ar:
            DataFrame of actions and rewards
        testbed:
            A bandits.testbeds.Testbed instance
        run:
            An integer defining which run will have its actions overlaid
    """
    df_estimate = testbed.estimate_distribution(1000)
    df_estimate = df_estimate.astype({"action": "int32"})
    df_ar = df_ar.loc[df_ar["run"] == run]
    df_ar = df_ar.astype({"action": "int32"})
    p = (
        ggplot(
            aes(
                x="reorder(factor(action), action)",
                y="reward",
            )
        )
        + ggtitle(f"Action - Rewards across {df_ar.shape[0]} steps from Run {run}")
        + xlab("k-arm")
        + ylab("Reward")
        + geom_violin(df_estimate, fill="#d0d3d4")
        + geom_jitter(df_ar, aes(color="step"))
        + theme(figure_size=(20, 9))
    )
    fig = p.draw()

    return fig


def average_runs(df, group=None):
    """Average all dataframe columns across runs

    Args:
        df:
            DataFrame with step column
        group (list):
            Additional list of columns to group by before taking the average

    """
    if group is None:
        group = []

    return df.groupby(["step"] + group).mean().reset_index()


def optimal_action(df, group=None):
    """Calculate the percentage of runs that took the optimal action at this step,
        creates new column "optimal_action_percent"

    Args:
        group (list):
            Additional list of columns to group by before calculating percent optimal action

    """
    if group is None:
        group = []

    df["optimal_action_true"] = np.where(df["action"] == df["optimal_action"], 1, 0)
    df["optimal_action_percent"] = df["step"].map(
        df.groupby(["step"])["optimal_action_true"].sum() / (df["run"].max() + 1)
    )

    return df


def upload(df, columns: list[str]):
    """Upload selected columns from dataframe to remote wandb

    Args:
        df:
            Dataframe
        columns:
            List of column names to log to wandb

    Return
        Dict of values of last step
    """
    df = df[columns]
    rows = df.to_dict(orient="records")
    for r in rows:
        wandb.log(r)

    return r


@hydra.main(config_path="configs", config_name="defaults", version_base="1.3")
def main(cfg: DictConfig):
    local_logger.info("Run in debug mode by setting hydra.verbose=true")
    if not cfg.experiment.upload:
        local_logger.info(
            "wandb upload set to false, local run only. Set cfg.experiment.upload=true to track experiment"
        )

    hp_testbed = OmegaConf.to_container(cfg.testbed)

    bandit_type = cfg.bandit._target_.split(".")[-1]
    # Q_init = cfg.Q_init._target_.split(".")[-1]
    hp = {
        ("class" if k == "_target_" else k): (bandit_type if k == "_target_" else v)
        for k, v in OmegaConf.to_container(cfg.bandit).items()
    }
    hp["n_cpus"] = cfg.run.n_jobs
    # hp["Q_init"] = Q_init
    # hp["Q_init_value"] = cfg.Q_init.q_val
    hp["p_drift"] = hp_testbed["p_drift"]

    testbed = instantiate(cfg.testbed, _convert_="all")
    bandit = instantiate(
        cfg.bandit,
        Q_init=np.zeros(testbed.expected_values["mean"].size),
        _convert_="all",
    )

    local_logger.info(f"Running bandit: {cfg.run}")
    local_logger.debug(f"Testbed expected values: {testbed.expected_values}")
    local_logger.debug(f"bandit config: {cfg['bandit']}")
    local_logger.debug(f"run config: {cfg['run']}")

    run_start = time.monotonic()
    bandit.run(testbed, **OmegaConf.to_container(cfg.run))
    run_end = time.monotonic()

    df_ar = bandit.output_df()
    df_ar = optimal_action(df_ar)
    local_logger.debug(
        f"\n{df_ar[['run', 'step', 'action', 'optimal_action', 'reward']]}"
    )

    if cfg.experiment.upload:
        hp["tag"] = "debug" if HydraConfig.get().verbose else cfg.experiment["tag"]
        wandb.init(project="rlbook", group="bandits", config=hp, tags=[hp["tag"]])
        wandb.define_metric("reward", summary="last")
        wandb.define_metric("optimal_action_percent", summary="last")
        df_avg_ar = average_runs(df_ar)
        upload(df_avg_ar, ["reward", "optimal_action_percent"])

        wandb.log(
            {"duration (s)": timedelta(seconds=run_end - run_start).total_seconds()}
        )

        wandb.log(
            {"Reward Distribution": wandb.Image(steps_violin_plotter(df_ar, testbed))}
        )
        wandb.finish()


if __name__ == "__main__":
    main()
