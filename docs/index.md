Code for my walkthrough of: [*Reinforcement Learning An Introduction by Richard Sutton and Andrew Barto*](http://incompleteideas.net/book/the-book.html) 

## Quickstart
Algorithm implementations are located in the `/src` directory while the scaffolding code/notebooks for recreating/exploring Sutton & Barto are segmented into the `experiments/` directory.  

e.g. for recreating Figure 2.3, navigate to `/experiments/ch2_bandits/` and run:
```bash
python run.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0,0.01,0.1 +bandit.random_argmax=true experiment.tag=fig2.2 experiment.upload=true
```

![image](https://github.com/user-attachments/assets/5ca29806-0cdd-4f79-9a6d-08aa1ca72417)
Figure 2.3 (rlbook): The `+bandit.random_argmax=true` flag was used to switch over to an argmax implementation that randomizes between tiebreakers rather than first occurence used in the default numpy implementation to better align with the original example.
[Link to wandb artifact](https://api.wandb.ai/links/josephjnl/53gxgbcc)

Further details on experimental setup and results can be found at corresponding chapter README's.