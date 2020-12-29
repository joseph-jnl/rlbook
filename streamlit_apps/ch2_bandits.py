import streamlit as st
import pandas as pd
import numpy as np
from rlbook.bandits import EpsilonGreedy, init_optimistic
from rlbook.testbeds import NormalTestbed
from streamlit_apps.plotters import steps_plotter, reward_plotter
import altair as alt
import plotnine as p9
import time

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title=None,  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

N = 2000
RUNS = 1000
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

EXPERIMENTS = {
    "Single Run": {
        f"Single Run - Epsilon greedy bandit - 0 init": {
            "fx": steps_plotter,
            "config": (
                {e: EpsilonGreedy(testbed, epsilon=e) for e in EPSILONS},
                {"steps": N, "n_runs": 1, "n_jobs": 1},
                testbed,
            ),
        },
    },
    "0 initialization": {
        f"{RUNS} Runs - Epsilon greedy bandit - 0 init - varying 1/N step size alpha": {
            "fx": reward_plotter,
            "config": (
                {e: EpsilonGreedy(testbed, epsilon=e, alpha=None) for e in EPSILONS},
                {"steps": N, "n_runs": RUNS, "n_jobs": 8,
                },
            ),
        },
        f"{RUNS} Runs - Epsilon greedy bandit - 0 init - constant step size": {
            "fx": reward_plotter,
            "config": (
                {e: EpsilonGreedy(testbed, epsilon=e) for e in EPSILONS},
                {"steps": N, "n_runs": RUNS, "n_jobs": 8},
            ),
        },
        f"{RUNS} Runs - Drifting Testbed - Epsilon greedy bandit - 0 init - constant step size": {
            "fx": reward_plotter,
            "config": (
                {e: EpsilonGreedy(testbed_drift, epsilon=e) for e in EPSILONS},
                {"steps": N, "n_runs": RUNS, "n_jobs": 8},
            ),
        },
    },
    "optimistic initialization": {
        f"{RUNS} Runs - Epsilon greedy bandit - optimistic init - varying 1/N step size alpha": {
            "fx": reward_plotter,
            "config": (
                {e: EpsilonGreedy(testbed, epsilon=e, alpha=None, Q_init=init_optimistic) for e in EPSILONS},
                {"steps": N, "n_runs": RUNS, "n_jobs": 8,
                },
            ),
        },
        f"{RUNS} Runs - Epsilon greedy bandit - optimistic init - constant step size": {
            "fx": reward_plotter,
            "config": (
                {e: EpsilonGreedy(testbed, epsilon=e, Q_init=init_optimistic) for e in EPSILONS},
                {"steps": N, "n_runs": RUNS, "n_jobs": 8},
            ),
        },
        f"{RUNS} Runs - Drifting Testbed - Epsilon greedy bandit - optimistic init - constant step size": {
            "fx": reward_plotter,
            "config": (
                {e: EpsilonGreedy(testbed_drift, epsilon=e, Q_init=init_optimistic) for e in EPSILONS},
                {"steps": N, "n_runs": RUNS, "n_jobs": 8},
            ),
        },
    },
}

st.title("Multi-armed Bandits")
for title, exp in EXPERIMENTS.items():
    st.markdown("---")
    st.header(title)
    for subtitle, sub_exp in exp.items():
        st.subheader(subtitle)
        sub_exp["fx"](*sub_exp["config"])

