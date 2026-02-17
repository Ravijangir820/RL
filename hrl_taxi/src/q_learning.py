import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon, seed=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next, done):
        target = r
        if not done:
            target = r + self.gamma * np.max(self.q[s_next])
        self.q[s, a] += self.alpha * (target - self.q[s, a])
