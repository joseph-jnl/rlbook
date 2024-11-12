# Chapter 2 - Multi-armed Bandits

## Figures 2.1 - 2.2 Epsilon Greedy Bandits

Recreate the following experiments using the cli command below:
```bash
python run.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0,0.01,0.1 +bandit.random_argmax=true experiment.tag=fig2.2 experiment.upload=true
```  
<br/>

![image](https://github.com/user-attachments/assets/8787f138-c012-4a7a-8460-56427470bf51)  
Figure 2.1 (Sutton & Barto): An example bandit problem from the 10-armed testbed. The true value q*(a) of 
each of the ten actions was selected according to a normal distribution with mean zero and unit
variance, and then the actual rewards were selected according to a mean q*(a), unit-variance
normal distribution, as suggested by these gray distributions.  

<br/>

![image](https://github.com/user-attachments/assets/fc381cad-4682-41ea-a156-111e4f8904d8)  
Figure 2.1 (rlbook): The testbed used for this experiment used similar normal distributions with recreated means and unit variance as the Sutton & Barto example. Also provided are the actions and rewards across steps for a single run- notice how exploration increases with epsilon. [Link to wandb artifact.](https://wandb.ai/josephjnl/rlbook/reports/Reward-Distribution-24-11-10-21-43-58---VmlldzoxMDExMzQwOQ?accessToken=sbrnp8aoxsj042cih0e6g8l4zdat994qf6vu5fttvdlff4ahzqlwqlnj52w0k5v4)

<br/>
<br/>

![image](https://github.com/user-attachments/assets/274cd38b-1010-4498-a527-d1c68bf243c2)  
Figure 2.2 (Sutton & Barto): Average performance of epsilon-greedy action-value methods on the 10-armed testbed.
These data are averages over 2000 runs with different bandit problems. All methods used sample
averages as their action-value estimates.  

<br/>

![image](https://github.com/user-attachments/assets/b6be1da5-7f4e-418e-97f2-fa2fc71c2751)
![image](https://github.com/user-attachments/assets/f52d3095-4564-44d9-8922-8effa544d0e4)  
Figure 2.2 (rlbook): The `+bandit.random_argmax=true` flag was used to switch over to an argmax implementation that randomizes between tiebreakers rather than first occurence used in the default numpy implementation to better align with the original example. 
[Link to wandb artifact.](https://wandb.ai/josephjnl/rlbook/reports/optimal_action_percent-24-11-10-21-37-29---VmlldzoxMDExMzMyNQ)

[🔼 Back to top](#chapter-2---multi-armed-bandits)

***

## Figure 2.3 Optimistic Initial Q Estimates

Recreate the following experiment using the cli command below:
```bash
python run.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0.1 +bandit.random_argmax=true bandit.alpha=0.1 bandit.Q_init=0 experiment.tag=fig2.3 experiment.upload=true
```

```bash
python run.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0 +bandit.random_argmax=true bandit.alpha=0.1 bandit.Q_init=5 experiment.tag=fig2.3 experiment.upload=true
```

![image](https://github.com/user-attachments/assets/45975df2-9207-491b-8913-6119b1617a4c)  
Figure 2.3 (Sutton & Barto): The effect of optimistic initial action-value estimates on the 10-armed testbed.
Both methods used a constant step-size parameter, alpha=0.1


<br/>

![image](https://github.com/user-attachments/assets/5ca29806-0cdd-4f79-9a6d-08aa1ca72417)
Figure 2.3 (rlbook): The `+bandit.random_argmax=true` flag was used to switch over to an argmax implementation that randomizes between tiebreakers rather than first occurence used in the default numpy implementation to better align with the original example.
[Link to wandb artifact](https://api.wandb.ai/links/josephjnl/53gxgbcc)

[🔼 Back to top](#chapter-2---multi-armed-bandits)

***

## Figure 2.4 UCL Bandits

Recreate the following experiment using the cli command below:
```bash
python run.py run.steps=1000 run.n_runs=2000 experiment.tag=fig2.4 experiment.upload=true bandit._target_=rlbook.bandits.algorithms.UCB +bandit.c=2
```

```bash
python run.py run.steps=1000 run.n_runs=2000 experiment.tag=fig2.4 experiment.upload=true bandit._target_=rlbook.bandits.algorithms.EpsilonGreedy +bandit.epsilon=0.1
```

![image](https://github.com/user-attachments/assets/d33e5b99-9b08-4feb-bf94-495f6594af68)  
Figure 2.4 (Sutton & Barto): Average performance of UCB action selection on the 10-armed testbed. As shown,
UCB generally performs better than "-greedy action selection, except in the first k steps, when
it selects randomly among the as-yet-untried actions.

<br/>

![image](https://github.com/user-attachments/assets/08f5463e-9a43-4876-b405-5d996b711955)  
Figure 2.4 (rlbook): rlbook UCB implementation. Na, the array that keeps the count of how many times an action has been chosen was initialized with 1e-100 instead of 0 to prevent a divide by zero error.  
[Link to wandb artifact](https://api.wandb.ai/links/josephjnl/ol9eknr9)

[🔼 Back to top](#chapter-2---multi-armed-bandits)
