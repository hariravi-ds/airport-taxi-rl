import numpy as np
from collections import defaultdict


class TabularAgent:
    """
    Tabular RL agent: SARSA or Q-learning

    algorithm: "sarsa" or "q_learning"
    """

    def __init__(
        self,
        algorithm="q_learning",
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        # discretization knobs
        max_queue=60,
        max_congestion=50,
        time_bins=6,
        weather_bins=3,
        seed=123,
    ):
        assert algorithm in ("sarsa", "q_learning")
        self.algorithm = algorithm

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.max_queue = int(max_queue)
        self.max_congestion = int(max_congestion)
        self.time_bins = int(time_bins)
        self.weather_bins = int(weather_bins)

        self.action_size = 3
        self.Q = defaultdict(lambda: np.zeros(self.action_size, dtype=float))

        self.rng = np.random.default_rng(seed)

    def _unsquash_cong(self, c_squash):
        """
        c_squash = tanh(cong/10). Recover approx cong.
        """
        x = float(np.clip(c_squash, 0.0, 0.999))
        cong = 10.0 * np.arctanh(x)
        return cong

    def _discretize_state(self, state):
        """
        state = [queue_norm, cA_squash, cB_squash, cC_squash, time_norm, weather]
        """
        q_norm = float(np.clip(state[0], 0.0, 1.0))
        queue = int(np.clip(round(q_norm * self.max_queue), 0, self.max_queue))

        cA = self._unsquash_cong(state[1])
        cB = self._unsquash_cong(state[2])
        cC = self._unsquash_cong(state[3])

        cA = int(np.clip(round(cA), 0, self.max_congestion))
        cB = int(np.clip(round(cB), 0, self.max_congestion))
        cC = int(np.clip(round(cC), 0, self.max_congestion))

        time_norm = float(np.clip(state[4], 0.0, 1.0))
        weather = float(np.clip(state[5], 0.0, 1.0))

        t_bin = int(np.clip(time_norm * self.time_bins, 0, self.time_bins - 1))
        w_bin = int(np.clip(weather * self.weather_bins,
                    0, self.weather_bins - 1))

        return (queue, cA, cB, cC, t_bin, w_bin)

    def _get_Q_row(self, state):
        key = self._discretize_state(state)
        return self.Q[key], key

    def select_action(self, state):
        # epsilon-greedy with random tie-break for argmax
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))

        q_values, _ = self._get_Q_row(state)
        max_q = q_values.max()
        best = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.rng.choice(best))

    def greedy_action(self, state):
        q_values, _ = self._get_Q_row(state)
        max_q = q_values.max()
        best = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        q_values, key = self._get_Q_row(state)
        current = q_values[action]

        if done:
            target = reward
        else:
            next_q_values, _ = self._get_Q_row(next_state)
            if self.algorithm == "q_learning":
                target = reward + self.gamma * float(np.max(next_q_values))
            else:
                if next_action is None:
                    next_action = self.select_action(next_state)
                target = reward + self.gamma * \
                    float(next_q_values[next_action])

        q_values[action] += self.alpha * (target - current)
        self.Q[key] = q_values

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
