# Finite Markov Decision Processes

## Figure 3.2 Random Gridworld
Recreate the following experiments using the cli command below:
```bash
python run.py
```
![image](https://github.com/user-attachments/assets/216229c6-5925-4479-9cd7-52a62f4ad4fb)
/// caption
Figure 3.2: State-value function for a random policy (equal probability for all directions). Config uses this example as a default.
///

## Figure 3.8 Optimal Gridworld
Recreate the following experiments using the cli command below:
```bash
python run.py grid=example_3_8_optimal_grid plots=example_3_8_optimal_grid
```
![image](https://github.com/user-attachments/assets/9b2a7a65-5f23-4b64-9217-f96b20dd8ebd)
/// caption
Figure 3.5: Optimal state-value function and policy for a gridworld.
///

## Arbitrary Example
```bash
python run.py grid.n_rows=10 grid.n_cols=10 grid.special_states=[[0,0,8,1],[1,3,7,9]] grid.special_states_prime=[[4,2,1,8],[1,3,7,1]] grid.special_states_rewards=[10,5,8,15] plots.policy=true
```
![image](https://github.com/user-attachments/assets/a4260336-8467-4823-a2bc-66d266569c24)
/// caption
Example of creating a new gridworld with arbitrary size and rewards.
///

## Figure 4.1 Convergence of iterative policy eval
Recreate the following experiments using the cli command below:
```bash
python run.py grid=example_4_1_random_terminal_grid plots=example_4_1_random_terminal_grid
```
![image](https://github.com/user-attachments/assets/f902c80a-080e-4330-8830-beef4a5470fc)
/// caption
Figure 4.1: Convergence of iterative policy evaluation on a small gridworld.
///