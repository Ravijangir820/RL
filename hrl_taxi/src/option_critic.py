import numpy as np


class OptionCriticAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        n_options,
        alpha,
        gamma,
        epsilon,
        beta,
        seed=0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_options = n_options
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.rng = np.random.default_rng(seed)

        # Intra-option action-value: Q_U(s, o, a)
        self.q_u = np.zeros((n_states, n_options, n_actions), dtype=np.float32)

    def _q_o(self, state):
        # Option value: Q_O(s, o) = max_a Q_U(s, o, a)
        return np.max(self.q_u[state], axis=1)

    def select_option(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_options))
        return int(np.argmax(self._q_o(state)))

    def select_action(self, state, option):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_u[state, option]))

    def should_terminate(self):
        return self.rng.random() < self.beta

    def update(self, s, option, a, r, s_next, done):
        if done:
            target = r
        else:
            q_next_same = np.max(self.q_u[s_next, option])
            q_next_options = np.max(self._q_o(s_next))
            target = r + self.gamma * ((1.0 - self.beta) * q_next_same + self.beta * q_next_options)

        self.q_u[s, option, a] += self.alpha * (target - self.q_u[s, option, a])
