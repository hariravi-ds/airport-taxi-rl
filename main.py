import csv
import numpy as np
import torch

from env import TaxiEnv
from baselines import random_policy, always_A, greedy_min_congestion
from model import TabularAgent
from dqn_agent import DQNAgent


# Global experiment settings
SEED = 0
TABULAR_EPISODES = 2000
TABULAR_EVAL_EVERY_EP = 100
DQN_STEPS = 800_000
DQN_EVAL_EVERY = 10_000
N_EVAL_EPISODES = 50


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_act_fn(env, act_fn, episodes=50):
    rewards = []
    avg_taxis = []
    action_fracs_all = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        taxis = []
        counts = np.zeros(3, dtype=int)

        while not done:
            a = int(act_fn(obs))
            counts[a] += 1

            obs, r, done, info = env.step(a)
            ep_r += r

            if info.get("dispatched", 1) == 1:
                taxis.append(info["taxi_time"])

        rewards.append(ep_r)
        avg_taxis.append(float(np.mean(taxis)) if taxis else 0.0)
        action_fracs_all.append(counts / max(1, counts.sum()))

    return (
        float(np.mean(rewards)),
        float(np.mean(avg_taxis)),
        float(np.std(rewards)),
        float(np.std(avg_taxis)),
        np.mean(action_fracs_all, axis=0),
    )


def train_tabular_with_eval_log(
    algorithm: str,
    out_csv: str,
    episodes=2000,
    eval_every_ep=100,
    n_eval_episodes=50,
    seed=123,
):
    env = TaxiEnv(seed=seed)
    eval_env = TaxiEnv(seed=seed)

    agent = TabularAgent(
        algorithm=algorithm,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        max_queue=env.max_queue,
        max_congestion=50,
        time_bins=6,
        weather_bins=3,
        seed=seed,
    )

    rows = []
    print(f"\n=== Training TABULAR {algorithm.upper()} ===")

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_reward = 0.0

        a = agent.select_action(s)

        while not done:
            ns, r, done, info = env.step(a)
            ep_reward += r

            if algorithm == "sarsa":
                na = agent.select_action(ns)
                agent.update(s, a, r, ns, next_action=na, done=done)
                s, a = ns, na
            else:  # q_learning
                agent.update(s, a, r, ns, done=done)
                s = ns
                a = agent.select_action(s)

        agent.decay_epsilon()

        if ep % 100 == 0 or ep == 1:
            print(
                f"Ep {ep}/{episodes}  train_reward={ep_reward:.2f}  eps={agent.epsilon:.3f}")

        if ep % eval_every_ep == 0:
            step_equiv = ep * env.episode_length
            eval_r, eval_taxi, r_std, taxi_std, fracs = evaluate_act_fn(
                eval_env, act_fn=lambda obs: agent.greedy_action(obs), episodes=n_eval_episodes
            )
            print(
                f"[{algorithm}] step={step_equiv}  eval_reward={eval_r:.2f}±{r_std:.2f}  "
                f"eval_avg_taxi={eval_taxi:.2f}±{taxi_std:.2f}  actions(A,B,C)={fracs}"
            )
            rows.append([
                step_equiv, eval_r, eval_taxi, r_std, taxi_std,
                agent.epsilon, "", fracs[0], fracs[1], fracs[2]
            ])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "eval_reward", "eval_avg_taxi", "reward_std", "taxi_std",
            "epsilon", "loss", "frac_A", "frac_B", "frac_C"
        ])
        w.writerows(rows)

    print(f"Saved: {out_csv}")
    return agent


def train_dqn_with_eval_log(
    out_csv: str,
    num_steps=120_000,
    eval_every=10_000,
    n_eval_episodes=50,
    seed=123,
):
    set_seed(seed)

    env = TaxiEnv(seed=seed)
    eval_env = TaxiEnv(seed=seed)

    obs_dim = env.reset().shape[0]
    agent = DQNAgent(
        obs_dim=obs_dim,
        act_dim=3,
        seed=seed,
        dueling=True,
        double_dqn=True,
        lr=3e-4,
        batch_size=128,
        learning_starts=10_000,
        eps_decay_steps=num_steps,
        eps_final=0.05,
        grad_clip=5.0,
    )

    obs = env.reset()
    rows = []

    print("\n=== Training DQN ===")
    for t in range(1, num_steps + 1):
        agent.total_steps = t

        a = agent.act(obs, greedy=False)
        next_obs, r, done, info = env.step(a)

        agent.observe(obs, a, r, next_obs, done)

        loss = None
        if t % agent.train_freq == 0:
            loss = agent.train_step()

        agent.maybe_update_target()
        obs = next_obs if not done else env.reset()

        if t >= agent.learning_starts and t % eval_every == 0:
            eval_r, eval_taxi, r_std, taxi_std, fracs = evaluate_act_fn(
                eval_env, act_fn=lambda ob: agent.act(ob, greedy=True), episodes=n_eval_episodes
            )
            eps = agent.epsilon()
            print(
                f"[DQN] step={t}  eval_reward={eval_r:.2f}±{r_std:.2f}  "
                f"eval_avg_taxi={eval_taxi:.2f}±{taxi_std:.2f}  actions(A,B,C)={fracs}  "
                f"eps={eps:.3f}  loss={loss}"
            )

            rows.append([
                t, eval_r, eval_taxi, r_std, taxi_std, eps,
                loss if loss is not None else "",
                fracs[0], fracs[1], fracs[2]
            ])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "eval_reward", "eval_avg_taxi", "reward_std", "taxi_std",
            "epsilon", "loss", "frac_A", "frac_B", "frac_C"
        ])
        w.writerows(rows)

    print(f"Saved: {out_csv}")
    return agent


def main():
    # 1. Train tabular models
    q_agent = train_tabular_with_eval_log(
        algorithm="q_learning",
        out_csv="qlearning_log.csv",
        episodes=TABULAR_EPISODES,
        eval_every_ep=TABULAR_EVAL_EVERY_EP,
        n_eval_episodes=N_EVAL_EPISODES,
        seed=SEED,
    )

    sarsa_agent = train_tabular_with_eval_log(
        algorithm="sarsa",
        out_csv="sarsa_log.csv",
        episodes=TABULAR_EPISODES,
        eval_every_ep=TABULAR_EVAL_EVERY_EP,
        n_eval_episodes=N_EVAL_EPISODES,
        seed=SEED,
    )

    # 2. Train DQN
    dqn_agent = train_dqn_with_eval_log(
        out_csv="dqn_log.csv",
        num_steps=DQN_STEPS,
        eval_every=DQN_EVAL_EVERY,
        n_eval_episodes=N_EVAL_EPISODES,
        seed=SEED,
    )

    # 3. Final evaluation
    env_eval = TaxiEnv(seed=SEED)

    dqn_r, dqn_t, dqn_rs, dqn_ts, dqn_fr = evaluate_act_fn(
        env_eval, act_fn=lambda ob: dqn_agent.act(ob, greedy=True), episodes=200
    )
    q_r, q_t, q_rs, q_ts, q_fr = evaluate_act_fn(
        env_eval, act_fn=lambda ob: q_agent.greedy_action(ob), episodes=200
    )
    s_r, s_t, s_rs, s_ts, s_fr = evaluate_act_fn(
        env_eval, act_fn=lambda ob: sarsa_agent.greedy_action(ob), episodes=200
    )

    # Baselines
    rand_r, rand_t, _, _, rand_fr = evaluate_act_fn(
        TaxiEnv(seed=SEED), act_fn=random_policy, episodes=200
    )
    aA_r, aA_t, _, _, aA_fr = evaluate_act_fn(
        TaxiEnv(seed=SEED), act_fn=always_A, episodes=200
    )
    cong_r, cong_t, _, _, cong_fr = evaluate_act_fn(
        TaxiEnv(seed=SEED), act_fn=greedy_min_congestion, episodes=200
    )

    print("\n=== FINAL COMPARISON ===")
    print(
        f"DQN:        reward={dqn_r:.1f}±{dqn_rs:.1f}, taxi={dqn_t:.2f}±{dqn_ts:.2f}, fracs={dqn_fr}")
    print(
        f"Q-Learning: reward={q_r:.1f}±{q_rs:.1f}, taxi={q_t:.2f}±{q_ts:.2f}, fracs={q_fr}")
    print(
        f"SARSA:      reward={s_r:.1f}±{s_rs:.1f}, taxi={s_t:.2f}±{s_ts:.2f}, fracs={s_fr}")
    print("--- baselines ---")
    print(
        f"Random:     reward={rand_r:.1f}, taxi={rand_t:.2f}, fracs={rand_fr}")
    print(f"Always A:   reward={aA_r:.1f}, taxi={aA_t:.2f}, fracs={aA_fr}")
    print(
        f"MinCong:    reward={cong_r:.1f}, taxi={cong_t:.2f}, fracs={cong_fr}")
    print("\nSaved logs: dqn_log.csv, qlearning_log.csv, sarsa_log.csv")


if __name__ == "__main__":
    main()
