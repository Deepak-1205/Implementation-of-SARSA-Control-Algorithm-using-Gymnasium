# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA (State-Action-Reward-State-Action) control algorithm using the Gymnasium FrozenLake-v1 environment to learn an action-value function (Q-function) and determine an optimal policy for navigating the environment while avoiding holes. The performance of the agent is evaluated by analyzing the learned $Q$-table, state-value function, derived optimal policy, average cumulative reward, and learning convergence curves.

## Software Requirements

* **Python Version:** Python 3.8 or higher
* **Libraries & Packages:**
  * `gymnasium` (Environment API)
  * `numpy` (Q-table computation and numerical operations)
  * `matplotlib` (Plotting learning curves and evaluation metrics)


## Environment Description
The **FrozenLake-v1** environment models a grid world (typically 4x4) where an agent must navigate from a starting point to a goal without falling into holes.

* **Grid Types:**
  * `S`: Starting point (safe)
  * `F`: Frozen surface (safe)
  * `H`: Hole (terminal state, zero reward)
  * `G`: Goal (terminal state, target destination)

* **State Space:** Discrete state space with 16 states representing grid coordinates (for a 4x4 grid).

* **Action Space:** Discrete action space with 4 possible movements:
  * `0`: Left
  * `1`: Down
  * `2`: Right
  * `3`: Up

* **Reward System:**
  * Reaching the Goal (`G`): **+1**
  * Falling into a Hole (`H`): **0**
  * Stepping on Frozen ice (`F`): **0**

* **Dynamics:** Can be configured as deterministic (`is_slippery=False`) or stochastic (`is_slippery=True`), where slippery movement introduces randomness into action outcomes.


## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm


## Python Program

```python
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------ -
# 1. Create Custom FrozenLake Environment
# ------------------------------------------------ -
# 'S' = Start (State 4), 'G' = Goal (State 10), 'H' = Hole, 'F' = Frozen
custom_map = [
    "FFFF",  # States 0-3
    "SHFH",  # States 4-7  (Start at State 4)
    "FFGH",  # States 8-11 (Goal at State 10)
    "HFFF",  # States 12-15
]

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=True)

# ------------------------------------------------ -
# 2. Hyperparameters & Epsilon Configuration
# ------------------------------------------------ -
num_episodes = 10000
max_steps_per_episode = 100
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor

epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.05  # Minimum exploration threshold
epsilon_decay = 0.9995  # Decay multiplier per episode

# ------------------------------------------------ -
# 3. Initialize Q-Table & Action Selection Function
# ------------------------------------------------ -
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
Q = np.zeros((state_space_size, action_space_size))


def epsilon_greedy_action(state, eps):
  """Selects action using epsilon-greedy strategy."""
  if np.random.uniform(0, 1) < eps:
    return env.action_space.sample()
  return np.argmax(Q[state, :])


# ------------------------------------------------ -
# 4. SARSA Training Loop
# ------------------------------------------------ -
episode_rewards = []

for episode in range(num_episodes):
  state, info = env.reset()
  action = epsilon_greedy_action(state, epsilon)
  total_reward = 0

  for step in range(max_steps_per_episode):
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    next_action = epsilon_greedy_action(next_state, epsilon)

    # SARSA Update Rule: Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
    td_target = reward + (0 if done else gamma * Q[next_state, next_action])
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error

    state = next_state
    action = next_action
    total_reward += reward

    if done:
      break

  # Decay epsilon
  epsilon = max(epsilon_min, epsilon * epsilon_decay)
  episode_rewards.append(total_reward)

# ------------------------------------------------ -
# 5. Output Functions & Results Display
# ------------------------------------------------ -
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)


def print_value_function(values):
  print("\nEstimated State-Value Function:")
  print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
  action_symbols = {0: "L", 1: "D", 2: "R", 3: "U"}
  policy_grid = np.array([action_symbols[a] for a in policy]).reshape(4, 4)
  print("\nLearned Policy:")
  print(policy_grid)


print("\nFinal Q-table:")
print(np.round(Q, 3))
print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# ------------------------------------------------ -
# 6. Plot Learning Curve
# ------------------------------------------------ -
window = 500
moving_average = np.convolve(
    episode_rewards, np.ones(window) / window, mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - Custom FrozenLake")
plt.grid(True)
plt.show()

env.close()

```
---

## Output

<img width="1375" height="636" alt="image" src="https://github.com/user-attachments/assets/ec16d427-265d-4703-a776-643f8ba2bf31" />

<img width="1222" height="540" alt="image" src="https://github.com/user-attachments/assets/4a988d33-a14b-422b-aa5a-94d2f8150adb" />

---

## Result

The SARSA control algorithm was successfully implemented and evaluated on the `FrozenLake-v1` environment. The agent successfully updated its action-value function ($Q$-table) across training episodes, learning an optimal policy that navigates the grid world to reach the goal state while minimizing the risk of falling into holes.

---

## Inference

* **On-Policy vs. Safety:** Unlike off-policy methods like Q-learning, SARSA updates its action-value estimates based on the action actually executed under the current $\epsilon$-greedy policy. This makes the agent more conservative, as it actively takes the potential risks of exploration into account during policy updates.
* **Impact of Environment Stochasticity:** When `is_slippery=True`, the stochastic transitions require a lower learning rate and careful exploration decay to prevent policy oscillation, whereas a non-slippery setup allows faster convergence to the optimal deterministic path.
---

