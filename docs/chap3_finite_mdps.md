# Finite Markov Decision Processes

```bash
python run.py
```
/// caption
Figure 3.2: State-value function for a random policy (equal probability for all directions). Config uses this example as a default.
///

```bash
python run.py grid=example_3_8_optimal_grid plots=example_3_8_optimal_grid
```

/// caption
Figure 3.5: Optimal state-value function and policy for a gridworld.
///

```bash
python run.py grid.n_rows=10 grid.n_cols=10 grid.special_states=[[0,0,8,1],[1,3,7,9]] grid.special_states_prime=[[4,2,1,8],[1,3,7,1]] grid.special_states_rewards=[10,5,8,15] plots.policy=true
```
/// caption
Example of creating a new gridworld with arbitrary size and rewards.
///