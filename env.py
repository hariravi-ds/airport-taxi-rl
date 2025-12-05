import numpy as np


class TaxiEnv:
    """
    Airport taxi routing environment.

    Action space: 0(A), 1(B), 2(C)
    Observation: [queue_norm, congestionA_squash, congestionB_squash, congestionC_squash, time_norm, weather]
      - queue_norm in [0,1]
      - cong*_squash = tanh(cong/10) in [0, ~1)
      - time_norm in [0,1]
      - weather in [0,1]

    Reward: -(taxi_time + holding_cost + congestion_cost) / 10
    """

    def __init__(
        self,
        episode_length=400,
        dt=0.1,          # hours per step (0.1h = 6 min)
        max_queue=60,
        seed=123,
    ):
        self.episode_length = int(episode_length)
        self.dt = float(dt)
        self.max_queue = int(max_queue)

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # capacities (aircraft/hour): higher => route clears faster
        self.route_capacity_per_hour = {'A': 6.0, 'B': 8.0, 'C': 10.0}

        # base taxi time (minutes) for each route
        self.base_times = np.array([3.0, 6.0, 6.5], dtype=float)

        # minutes per congestion unit on that route
        self.congestion_penalty = 6.0

        # costs
        self.queue_holding_cost = 0.15
        self.system_cong_cost = 0.05

        # spillover (choosing one route increases others a bit)
        self.spill_map = {'A': 0.35, 'B': 0.15, 'C': 0.05}

        # weather effect per route (minutes multiplier)
        self.weather_mult = {'A': 3.0, 'B': 3.0, 'C': 2.0}

        self.reset()

    def reset(self):
        self.queue_length = 10
        self.congestion = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        self.time = 0.0
        self.steps = 0

        # start weather in [0,1]
        self.weather_factor = float(self.rng.uniform(0.0, 1.0))

        return self._get_state()

    def _arrival_rate_per_hour(self, hour):
        """Piecewise arrival-rate schedule (aircraft/hour)."""
        if 6.0 <= hour < 10.0:
            return 18.0
        if 16.0 <= hour < 19.0:
            return 15.0
        return 8.0

    def _evolve_weather(self):
        """Mean-reverting weather in [0,1]."""
        mu = 0.4
        theta = 0.15
        sigma = 0.08

        w = self.weather_factor
        w = w + theta * (mu - w) + self.rng.normal(0.0, sigma)
        self.weather_factor = float(np.clip(w, 0.0, 1.0))

    def _get_state(self):
        time_normalized = (self.time % 24.0) / 24.0

        q = np.clip(self.queue_length / self.max_queue, 0.0, 1.0)

        cA = np.tanh(self.congestion['A'] / 10.0)
        cB = np.tanh(self.congestion['B'] / 10.0)
        cC = np.tanh(self.congestion['C'] / 10.0)

        return np.array([q, cA, cB, cC, time_normalized, self.weather_factor], dtype=np.float32)

    def step(self, action: int):
        assert action in [0, 1, 2]
        route = ['A', 'B', 'C'][action]

        dispatched = 0
        taxi_time = 0.0

        if self.queue_length > 0:
            self.queue_length -= 1
            dispatched = 1

            # add congestion on chosen route
            self.congestion[route] += 1.0

            # spillover to other routes
            spill = self.spill_map[route]
            for o in ['A', 'B', 'C']:
                if o != route:
                    self.congestion[o] += spill

            # compute taxi time (minutes)
            taxi_time = (
                self.base_times[action]
                + self.congestion[route] * self.congestion_penalty
                + self.weather_factor * self.weather_mult[route]
                + self.rng.normal(0.0, 0.25)
            )
            taxi_time = float(max(0.0, taxi_time))

        # decay congestion with service capacity (only if dispatched this step)
        for k in ['A', 'B', 'C']:
            served = (self.route_capacity_per_hour[k] * self.dt) * dispatched
            self.congestion[k] = float(max(0.0, self.congestion[k] - served))

        # cap for numerical sanity
        for k in ['A', 'B', 'C']:
            self.congestion[k] = float(min(self.congestion[k], 50.0))

        # arrivals: Poisson(lambda_per_hour * dt)
        hour = self.time % 24.0
        lam_per_hour = self._arrival_rate_per_hour(hour)
        arrivals = int(self.rng.poisson(lam_per_hour * self.dt))
        self.queue_length = min(self.max_queue, self.queue_length + arrivals)

        # evolve weather
        self._evolve_weather()

        # reward (negative cost)
        total_cong = self.congestion['A'] + \
            self.congestion['B'] + self.congestion['C']
        holding_cost = self.queue_holding_cost * self.queue_length
        cong_cost = self.system_cong_cost * total_cong

        reward = -(taxi_time + holding_cost + cong_cost) / 10.0

        self.time += self.dt
        self.steps += 1
        done = self.steps >= self.episode_length

        info = {
            'taxi_time': taxi_time,
            'arrivals': arrivals,
            'weather': self.weather_factor,
            'queue': self.queue_length,
            'total_congestion': total_cong,
            'dispatched': dispatched
        }

        return self._get_state(), float(reward), bool(done), info
