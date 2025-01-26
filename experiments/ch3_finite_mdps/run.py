# ruff: noqa: F722
# https://docs.kidger.site/jaxtyping/faq/#flake8-or-ruff-are-throwing-an-error

import logging
from string import ascii_uppercase

import hydra
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from jax._src import config
from jaxtyping import Array, Float
from omegaconf import DictConfig, OmegaConf
from plotnine import (
    aes,
    arrow,
    element_blank,
    element_rect,
    geom_spoke,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_cmap,
    scale_y_reverse,
    theme,
    theme_void,
)

from rlbook.gridworlds.grids import Grid, OptimalGrid, RandomGrid
from rlbook.plots.plotnine_utils import subplot

local_logger = logging.getLogger("experiment")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
config.update("jax_platforms", "cuda")  # Assume GPU available


def plot_state_reward(v, grid: Grid, label=True):
    df = pd.DataFrame(v).reset_index().melt("index").astype({"variable": "int64"})
    df.columns = ["row", "column", "value"]
    df["color"] = pd.cut(df["value"], bins=5)

    p = (
        ggplot(df, aes("column", "row", fill="value"))
        + geom_tile()
        + geom_tile(fill=None, color="black")
        + scale_fill_cmap(cmap_name="cividis", limits=[df.value.min(), df.value.max()])
        + scale_color_manual(["white", "white", "black", "black", "black"])
        + scale_y_reverse()
        + theme_void()
        + theme(
            axis_line=element_blank(),
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            legend_position="none",
            # plot_background=pn.element_rect(fill="white"),
        )
        + labs(title="v⁎")
    )
    if label:
        p = p + geom_text(aes(label="value", color="color"), format_string="{:.1f}")
    return p


def plot_gridworld(v, grid: Grid, label=True):
    df = pd.DataFrame(v).reset_index().melt("index").astype({"variable": "int64"})
    df.columns = ["row", "column", "value"]
    df_special = pd.DataFrame(
        {
            "row": grid.special_states[0],
            "column": grid.special_states[1],
            "value": list(range(len(grid.special_states[0]))),
            "label": list(ascii_uppercase)[0 : len(grid.special_states[0])],
        }
    )
    df_special_prime = pd.DataFrame(
        {
            "row": grid.special_states_prime[0],
            "column": grid.special_states_prime[1],
            "value": list(range(len(grid.special_states_prime[0]))),
            "label": list(ascii_uppercase)[0 : len(grid.special_states_prime[0])],
        }
    )
    df_reward = pd.DataFrame(
        {
            "row": grid.special_states[0],
            "column": grid.special_states[1],
            "value": grid.special_states_rewards,
        }
    )

    p = (
        ggplot(df, aes("column", "row", fill="value"))
        + geom_tile(fill="lightgrey")
        + geom_tile(fill=None, color="black")
        + geom_tile(aes(color="value"), data=df_special)
        + geom_tile(
            aes(color="value"),
            data=df_special_prime,
        )
        + scale_fill_cmap(
            cmap_name="Pastel1", limits=[df_special.value.min(), df_special.value.max()]
        )
        + scale_y_reverse()
        + theme_void()
        + theme(
            axis_line=element_blank(),
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            legend_position="none",
            # plot_background=pn.element_rect(fill="white"),
        )
        + labs(title="Gridworld")
    )
    if label:
        p = (
            p
            + geom_text(
                aes(label="label"),
                data=df_special,
                format_string="{}",
                fontweight="bold",
            )
            + geom_text(
                aes(label="label"),
                data=df_special_prime,
                format_string="{}'",
                fontweight="bold",
            )
            + geom_text(
                aes(label="value"),
                data=df_reward,
                nudge_y=-0.25,
                format_string="(Reward: {})",
            )
        )
    return p


def v_policy(
    v: Float[Array, "n_rows n_cols"], special_states: list[list[int], list[int]]
):
    """Compute the policy from the value function
    Args:
        v: value function
        special_states: list containing special states row and columns.
            e.g. [[0, 0], [1, 3]] would correspond to special state A located at row 0 and column 1
            and special state B located at row 1 and column 3.
    """
    special_states_ij = tuple(zip(special_states[0], special_states[1]))
    policy = np.empty(v.shape, dtype=np.ndarray)
    action_map = {
        ("up", "left", "down", "right"): (90, 180, 270, 0),
        ("up", "left", "down"): (90, 180, 270, np.nan),
        ("up", "down", "right"): (90, np.nan, 270, 0),
        ("left", "down", "right"): (np.nan, 180, 270, 0),
        ("up", "left", "right"): (90, 180, 270, np.nan),
        ("up", "down"): (90, np.nan, 270, np.nan),
        ("left", "right"): (np.nan, 180, np.nan, 0),
        ("up", "left"): (90, 180, np.nan, np.nan),
        ("up", "right"): (90, np.nan, np.nan, 0),
        ("down", "right"): (np.nan, np.nan, 270, 0),
        ("left", "down"): (np.nan, 180, 270, np.nan),
        ("up",): (90, np.nan, np.nan, np.nan),
        ("left",): (np.nan, 180, np.nan, np.nan),
        ("down",): (np.nan, np.nan, 270, np.nan),
        ("right",): (np.nan, np.nan, np.nan, 0),
    }
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            actions = {
                "up": -1e9,
                "left": -1e9,
                "down": -1e9,
                "right": -1e9,
            }
            if (i, j) in special_states_ij:
                policy[i, j] = [90, 180, 270, 0]
            else:
                if i != 0:
                    actions["up"] = v[i - 1, j]
                if i != v.shape[0] - 1:
                    actions["down"] = v[i + 1, j]
                if j != 0:
                    actions["left"] = v[i, j - 1]
                if j != v.shape[0] - 1:
                    actions["right"] = v[i, j + 1]
                max_val = max(actions.values())
                max_rewards = [k for k, v in actions.items() if v == max_val]
                policy[i, j] = action_map[tuple(max_rewards)]

    return policy


def plot_policy(policy):
    df = pd.DataFrame(policy).reset_index().melt("index").astype({"variable": "string"})
    df.columns = ["row", "column", "value"]
    df[["up", "left", "down", "right"]] = pd.DataFrame(
        df["value"].to_list(), index=df.index
    ).apply(np.deg2rad)

    p = (
        ggplot(
            df,
            aes(
                "column",
                "row",
            ),
        )
        + geom_tile(fill="lightgrey")
        + geom_tile(fill=None, color="black")
        + scale_y_reverse()
        + theme_void()
        + theme(
            axis_line=element_blank(),
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            legend_position="none",
            plot_background=element_rect(fill="white"),
        )
        + labs(title="Policy")
    )
    p = p + geom_spoke(
        mapping=aes(x="column", y="row", radius=0.35, angle="up"),
        arrow=arrow(ends="last", type="closed", length=0.075),
        na_rm=True,
    )
    p = p + geom_spoke(
        mapping=aes(x="column", y="row", radius=0.35, angle="left"),
        arrow=arrow(ends="last", type="closed", length=0.075),
        na_rm=True,
    )
    p = p + geom_spoke(
        mapping=aes(x="column", y="row", radius=0.35, angle="down"),
        arrow=arrow(ends="last", type="closed", length=0.075),
        na_rm=True,
    )
    p = p + geom_spoke(
        mapping=aes(x="column", y="row", radius=0.35, angle="right"),
        arrow=arrow(ends="last", type="closed", length=0.075),
        na_rm=True,
    )
    return p


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    local_logger.info("Run in debug mode by setting hydra.verbose=true")
    if not cfg.experiment.upload:
        local_logger.info(
            "wandb upload set to false, local run only. Set experiment.upload=true to track experiment"
        )

    grid_type = cfg.grid._target_.split(".")[-1]
    grid_attrs = OmegaConf.to_container(cfg.grid)
    hp = {
        ("class" if k == "_target_" else k): (grid_type if k == "_target_" else v)
        for k, v in grid_attrs.items()
    }

    if grid_type == "RandomGrid":
        grid = RandomGrid(
            grid_attrs["special_states"],
            grid_attrs["special_states_prime"],
            jnp.array(grid_attrs["special_states_rewards"]),
            n_rows=grid_attrs["n_rows"],
            n_cols=grid_attrs["n_cols"],
        )
    elif grid_type == "OptimalGrid":
        grid = OptimalGrid(
            grid_attrs["special_states"],
            grid_attrs["special_states_prime"],
            jnp.array(grid_attrs["special_states_rewards"]),
            n_rows=grid_attrs["n_rows"],
            n_cols=grid_attrs["n_cols"],
        )
    else:
        raise ValueError(f"{grid_type} not of class RandomGrid or OptimalGrid")

    local_logger.info(f"Estimating state value function using for {grid_type}")
    v = grid.estimate_state_value(iter=grid_attrs["iter"])

    plots = []
    if cfg.plots.gridworld:
        local_logger.info("Plotting gridworld states setup")
        plots.append(plot_gridworld(v, grid, label=cfg.plots.label))
    if cfg.plots.v:
        local_logger.info("Plotting state value function")
        plots.append(plot_state_reward(v, grid, label=cfg.plots.label))
    if cfg.plots.policy:
        local_logger.info("Plotting policy")
        policy = v_policy(v, [[0, 0], [1, 3]])
        plots.append(plot_policy(policy))
    p = subplot(*plots, rows=1, cols=len(plots), figsize=tuple(cfg.plots.figsize))

    if cfg.experiment.upload:
        hp["tag"] = "debug" if HydraConfig.get().verbose else cfg.experiment["tag"]
        wandb.init(
            project="rlbook",
            dir="./logs/",
            group="gridworlds",
            config=hp,
            tags=[hp["tag"]],
        )
        wandb.log(
            {"Reward Distribution": wandb.Image(p)},
            commit=False,
        )
        wandb.finish()
    else:
        plt.show()


if __name__ == "__main__":
    main()
