import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, obs_dim, size=100_000, seed=123):
        self.size = int(size)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.size,), dtype=np.int64)
        self.rews = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)

        self.ptr = 0
        self.len = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.len, size=batch_size)
        return (
            self.obs[idx],
            self.acts[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, dueling=True):
        super().__init__()
        self.dueling = dueling

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        if dueling:
            self.value = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.adv = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))
        else:
            self.head = nn.Linear(hidden, act_dim)

    def forward(self, x):
        z = self.backbone(x)
        if not self.dueling:
            return self.head(z)

        v = self.value(z)
        a = self.adv(z)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        act_dim=3,
        seed=123,
        device=None,
        dueling=True,
        double_dqn=True,
        gamma=0.99,
        lr=1e-3,
        buffer_size=100_000,
        batch_size=64,
        learning_starts=2_000,
        train_freq=1,
        target_update_freq=1_000,
        eps_start=1.0,
        eps_final=0.05,
        eps_decay_steps=50_000,
        grad_clip=10.0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.double_dqn = bool(double_dqn)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.target_update_freq = int(target_update_freq)
        self.grad_clip = float(grad_clip)
        self.tau = 0.005

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)

        self.q = QNet(obs_dim, act_dim, dueling=dueling).to(self.device)
        self.q_targ = QNet(obs_dim, act_dim, dueling=dueling).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.q_targ.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.rb = ReplayBuffer(obs_dim, size=buffer_size, seed=seed)

        # epsilon schedule
        self.eps_start = float(eps_start)
        self.eps_final = float(eps_final)
        self.eps_decay_steps = int(eps_decay_steps)

        self.total_steps = 0

    def epsilon(self):
        t = min(1.0, self.total_steps / max(1, self.eps_decay_steps))
        return self.eps_start + t * (self.eps_final - self.eps_start)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        if (not greedy) and (self.rng.random() < self.epsilon()):
            return int(self.rng.integers(0, self.act_dim))

        x = torch.tensor(obs, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        q = self.q(x)
        q_np = q.squeeze(0).detach().cpu().numpy()
        max_q = q_np.max()
        best = np.flatnonzero(np.isclose(q_np, max_q))
        return int(self.rng.choice(best))

    def observe(self, obs, act, rew, next_obs, done):
        self.rb.add(obs, act, rew, next_obs, done)

    def train_step(self):
        if self.rb.len < self.learning_starts:
            return None

        obs, acts, rews, next_obs, dones = self.rb.sample(self.batch_size)

        obs_t = torch.tensor(obs, device=self.device)
        acts_t = torch.tensor(acts, device=self.device).unsqueeze(1)
        rews_t = torch.tensor(rews, device=self.device).unsqueeze(1)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_sa = self.q(obs_t).gather(1, acts_t)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = torch.argmax(
                    self.q(next_obs_t), dim=1, keepdim=True)
                next_q = self.q_targ(next_obs_t).gather(1, next_actions)
            else:
                next_q = torch.max(self.q_targ(next_obs_t),
                                   dim=1, keepdim=True).values

            target = rews_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()

        return float(loss.item())

    def maybe_update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_targ.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)
