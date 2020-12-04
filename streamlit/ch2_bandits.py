import streamlit as st
import pandas as pd
import numpy as np
from rlbook.bandits import EpsilonGreedy
from rlbook.testbeds import NormalTestbed
import altair as alt
import plotnine as p9
import time

st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)
st.title('Multi-armed Bandits')

N = 1000
RUNS = 100
EPSILONS = [0, 0.01, 0.1]
EXPECTED_VALUES = {
    1: {'mean': 0.5, 'var': 1},
    2: {'mean': -1, 'var': 1},
    3: {'mean': 2, 'var': 1},
    4: {'mean': 1, 'var': 1},
    5: {'mean': 1.7, 'var': 1},
    6: {'mean': -2, 'var': 1},
    7: {'mean': -0.5, 'var': 1},
    8: {'mean': -1, 'var': 1},
    9: {'mean': 1.5, 'var': 1},
    10: {'mean': -1, 'var': 1},
    }

st.header("Epsilon greedy bandit")
testbed = NormalTestbed(EXPECTED_VALUES)
e_bandits = {e: EpsilonGreedy(testbed, epsilon=e) for e in EPSILONS}
bar_percent = [0]
bar = st.progress(sum(bar_percent))
for b in e_bandits.values():
    b.run(N)
    bar_percent.append(1/(len(EPSILONS)+2))
    bar.progress(sum(bar_percent))

df_ar = pd.concat([b.output_df() for b in e_bandits.values()]).reset_index(drop=True)
df_ar['Average Reward'] = df_ar.groupby(['Run', 'epsilon']).expanding()['Reward'].mean().reset_index(drop=True)
df_ar
df_estimate = testbed.estimate_distribution(1000)

p = (
    p9.ggplot(p9.aes(x='factor(Action)', y='Reward',))
    + p9.ggtitle('Varying epsilon values for a single run of a greedy-e action-value method')
    + p9.xlab('k-arm')
    + p9.ylab('Reward')
    + p9.geom_violin(df_estimate, fill='#d0d3d4')
    + p9.geom_jitter(df_ar, p9.aes(color='factor(epsilon)'))
    + p9.theme(figure_size=(20,9))
)

p2 = (
    p9.ggplot(df_ar, p9.aes(x='Step', y='Average Reward', color='factor(epsilon)'))
    + p9.ggtitle('Average reward across steps (n) for a single run over different epsilons, realistic initialization')
    + p9.geom_line()
    + p9.theme(figure_size=(20,9))
)
st.pyplot(p9.ggplot.draw(p))
bar_percent.append(1/(len(EPSILONS)+2))
bar.progress(sum(bar_percent))
st.pyplot(p9.ggplot.draw(p2))
bar.progress(1.0)

st.header("Epsilon greedy bandit")
bar_percent = [0]
bar_increment = 1/(len(EPSILONS)+2)
bar = st.progress(sum(bar_percent))
for b in e_bandits.values():
    b.run(N, runs=RUNS)
    bar_percent.append(bar_increment)
    bar.progress(sum(bar_percent))

df_ar = pd.concat([b.output_df() for b in e_bandits.values()]).reset_index(drop=True)
df_ar
df_ar['Average Reward'] = df_ar.groupby(['Run', 'epsilon']).expanding()['Reward'].mean().reset_index(drop=True)
df_mean = df_ar.groupby(['Step', 'epsilon']).mean('Average Reward').reset_index()
bar_percent.append(bar_increment)
p3 = (
    p9.ggplot(df_mean, p9.aes(x='Step', y='Average Reward', color='factor(epsilon)'))
    + p9.ggtitle(f"Average reward across steps (n) for {RUNS} runs over different epsilons, realistic initialization")
    + p9.geom_line()
    + p9.theme(figure_size=(20,9))
)
st.pyplot(p9.ggplot.draw(p3))
bar.progress(1.0)
