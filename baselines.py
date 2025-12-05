import numpy as np


def random_policy(obs):
    # choose randomly between the actions
    return np.random.randint(0, 3)


def always_A(obs):
    # always chooses A
    return 0


def greedy_min_congestion(obs):
    cA, cB, cC = obs[1], obs[2], obs[3]
    return int(np.argmin([cA, cB, cC]))
