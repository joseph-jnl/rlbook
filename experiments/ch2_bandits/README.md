# Chapter 2 - Multi-armed Bandits

```bash
python run.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0,0.01,0.1 +bandit.random_argmax=true experiment.tag=fig2.2 experiment.upload=true
```

Figure 2.1 (Sutton & Barto): An example bandit problem from the 10-armed testbed. The true value q*(a) of 
each of the ten actions was selected according to a normal distribution with mean zero and unit
variance, and then the actual rewards were selected according to a mean q*(a), unit-variance
normal distribution, as suggested by these gray distributions.

Figure 2.1 (rlbook): The testbed used for this experiment used similar normal distributions with recreated means and unit variance as the Sutton & Barto example. Also provided are the actions and rewards across steps for a single run- notice how exploration increases with epsilon.
![Link to wandb artifact](https://wandb.ai/josephjnl/rlbook/reports/Reward-Distribution-24-11-10-21-43-58---VmlldzoxMDExMzQwOQ?accessToken=sbrnp8aoxsj042cih0e6g8l4zdat994qf6vu5fttvdlff4ahzqlwqlnj52w0k5v4)

Figure 2.2 (Sutton & Barto): Average performance of epsilon-greedy action-value methods on the 10-armed testbed.
These data are averages over 2000 runs with different bandit problems. All methods used sample
averages as their action-value estimates.

Figure 2.2 (rlbook): We also used the `+bandit.random_argmax=true` flag to utilize a less performant implementation of argmax that randomizes between tiebreakers rather than first occurence used in the default numpy implementation.
![Link to wandb artifact](https://wandb.ai/josephjnl/rlbook/reports/optimal_action_percent-24-11-10-21-37-29---VmlldzoxMDExMzMyNQ?accessToken=1no1kyyrf8rihkbek12v8z85kjmti2z2onb0yayo208l5715zf6p9r06t7eiq8tt)