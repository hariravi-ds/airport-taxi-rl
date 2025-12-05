import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_EVAL_EPISODES = 50

# log files
FILES = {
    "DQN": "dqn_log.csv",
    "Q-Learning": "qlearning_log.csv",
    "SARSA": "sarsa_log.csv",
}


def ci95(std):
    # 95% CI of the mean over eval episodes
    return 1.96 * std / np.sqrt(max(1, N_EVAL_EPISODES))


def plot_metric(metric, std_col, title, ylabel):
    plt.figure(figsize=(10, 4))

    for name, path in FILES.items():
        df = pd.read_csv(path)
        x = df["step"].to_numpy()
        y = df[metric].to_numpy()

        plt.plot(x, y, linewidth=2, label=name)

        if std_col in df.columns:
            band = ci95(df[std_col].to_numpy())
            plt.fill_between(x, y - band, y + band, alpha=0.2)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_action_fracs():
    # to print separate figure per model
    for name, path in FILES.items():
        df = pd.read_csv(path)
        if not {"frac_A", "frac_B", "frac_C"}.issubset(df.columns):
            continue

        x = df["step"].to_numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(x, df["frac_A"].to_numpy(), linewidth=2, label="Route A")
        plt.plot(x, df["frac_B"].to_numpy(), linewidth=2, label="Route B")
        plt.plot(x, df["frac_C"].to_numpy(), linewidth=2, label="Route C")
        plt.title(f"{name} Greedy Policy Action Fractions (Eval)")
        plt.xlabel("Step")
        plt.ylabel("Fraction of actions")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    plot_metric(
        metric="eval_reward",
        std_col="reward_std",
        title="Eval Reward (DQN vs Q-Learning vs SARSA) — 95% CI",
        ylabel="Reward (higher is better)",
    )

    plot_metric(
        metric="eval_avg_taxi",
        std_col="taxi_std",
        title="Eval Avg Taxi Time (DQN vs Q-Learning vs SARSA) — 95% CI",
        ylabel="Avg Taxi Time (min)",
    )

    plot_action_fracs()


if __name__ == "__main__":
    main()
