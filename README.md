# Reinforcement Learning for Airport Taxi Route Management

This repo implements and compares Reinforcement Learning methods for routing aircraft through three taxi routes (A, B, C) in a simulated airport ground environment.

The controller observes queue length, route congestion, time of day, and weather, and chooses a route each step to minimize overall taxi delay and congestion.

---

## Project Structure

- **env.py**  
  Custom simulation environment `TaxiEnv` (queue arrivals, congestion buildup/decay, weather dynamics, reward).

- **dqn_agent.py**  
  Deep RL implementation: Dueling + Double DQN, replay buffer, and soft target updates (Polyak averaging).

- **model.py**  
  Tabular RL implementation: `TabularAgent` supporting **Q-Learning** and **SARSA** on a discretized state space.

- **baselines.py**  
  Simple baseline policies: Random, Always-A, Greedy-Min-Congestion.

- **main.py**  
  Runs training + evaluation for Q-Learning, SARSA, and DQN, logs results to CSV, and prints a final comparison table.

- **plot.py** *(optional)*  
  Plots learning curves (reward, taxi time, action fractions) from saved CSV logs.

---

## Environment Summary

- **Actions:** `0=A`, `1=B`, `2=C`
- **State (6D):**  
  `[queue_norm, congA(tanh), congB(tanh), congC(tanh), time_norm, weather]`
- **Reward:** negative cost (so values are typically **negative**)  
  Reward is the negative of total cost (taxi time + queue holding + congestion), so rewards are negative; less negative is better.
  Higher reward = **less negative** = better.

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run training + comparison
```
python main.py
```

### 3) Plot results
```
python plot.py
```
