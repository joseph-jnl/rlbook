import pandas as pd
import plotnine as p9


def steps_violin_plotter(df_ar, testbed, run=0):
    df_estimate = testbed.estimate_distribution(1000)
    df_ar = df_ar.loc[df_ar["run"]==run]
    p = (
        p9.ggplot(
            p9.aes(
                x="factor(action)",
                y="reward",
            )
        )
        + p9.ggtitle(
            f"Action - Rewards across {df_ar.shape[0]} steps"
        )
        + p9.xlab("k-arm")
        + p9.ylab("Reward")
        + p9.geom_violin(df_estimate, fill="#d0d3d4")
        + p9.geom_jitter(df_ar, p9.aes(color='step'))
        + p9.theme(figure_size=(20, 9))
    )
    fig = p.draw()

    return fig
