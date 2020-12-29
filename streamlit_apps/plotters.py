import streamlit as st
import pandas as pd
import plotnine as p9


def reward_plotter(bandits, run_config, plot=True):
    bar_percent = [0]
    bar_increment = 1 / (len(bandits) + 1)
    bar = st.progress(sum(bar_percent))

    for b in bandits.values():
        b.run(**run_config)
        bar_percent.append(bar_increment)
        bar.progress(sum(bar_percent))

    df_ar = pd.concat([b.output_df() for b in bandits.values()]).reset_index(drop=True)
    df_ar["Average Reward"] = (
        df_ar.groupby(["epsilon", "Run"])
        .expanding()["Reward"]
        .mean()
        .reset_index(drop=True)
    )
    df_mean = df_ar.groupby(["Step", "epsilon"]).mean("Average Reward").reset_index()

    if plot:
        p = (
            p9.ggplot(
                df_mean, p9.aes(x="Step", y="Average Reward", color="factor(epsilon)")
            )
            + p9.ggtitle(
                f"Average reward across steps (n) for {run_config['n_runs']} runs over different epsilons, {b.Q_init.__name__}"
            )
            + p9.geom_line()
            + p9.theme(figure_size=(20, 9))
        )
        bar_percent.append(bar_increment)
        bar.progress(sum(bar_percent))
        st.pyplot(p9.ggplot.draw(p))
    bar.empty()

    return df_ar


def steps_plotter(bandits, run_config, testbed):
    df_ar = reward_plotter(bandits, run_config)
    df_estimate = testbed.estimate_distribution(1000)
    p = (
        p9.ggplot(
            p9.aes(
                x="factor(Action)",
                y="Reward",
            )
        )
        + p9.ggtitle(f"Action - Rewards across {run_config['steps']} steps for different epsilons")
        + p9.xlab("k-arm")
        + p9.ylab("Reward")
        + p9.geom_violin(df_estimate, fill="#d0d3d4")
        + p9.geom_jitter(df_ar, p9.aes(color="factor(epsilon)"))
        + p9.theme(figure_size=(20, 9))
    )
    st.pyplot(p9.ggplot.draw(p))
