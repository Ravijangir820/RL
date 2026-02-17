import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomTaxiEnv(gym.Env):
    """Custom Taxi environment with configurable grid size."""

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.num_locations = 4
        self.num_taxis = 1
        
        state_size = grid_size * grid_size * grid_size * grid_size * (self.num_locations + 1) * self.num_locations
        self.observation_space = spaces.Discrete(state_size)
        self.action_space = spaces.Discrete(6)
        
        self.locations = [
            (0, 0),
            (0, grid_size - 1),
            (grid_size - 1, 0),
            (grid_size - 1, grid_size - 1),
        ]
        
        self.reset()

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        
        self.taxi_row = self.np_random.integers(0, self.grid_size)
        self.taxi_col = self.np_random.integers(0, self.grid_size)
        self.passenger_location = self.np_random.integers(0, self.num_locations)
        self.destination = self.np_random.integers(0, self.num_locations)
        
        while self.destination == self.passenger_location:
            self.destination = self.np_random.integers(0, self.num_locations)
        
        return self._get_state(), {}

    def _get_state(self):
        return (
            self.taxi_row * (self.grid_size * self.grid_size * (self.num_locations + 1) * self.num_locations)
            + self.taxi_col * (self.grid_size * (self.num_locations + 1) * self.num_locations)
            + self.passenger_location * (self.grid_size * self.num_locations)
            + self.destination
        )

    def step(self, action):
        old_row, old_col = self.taxi_row, self.taxi_col
        
        if action == 0:
            self.taxi_row = min(self.grid_size - 1, self.taxi_row + 1)
        elif action == 1:
            self.taxi_row = max(0, self.taxi_row - 1)
        elif action == 2:
            self.taxi_col = min(self.grid_size - 1, self.taxi_col + 1)
        elif action == 3:
            self.taxi_col = max(0, self.taxi_col - 1)
        elif action == 4:
            if (self.taxi_row, self.taxi_col) == self.locations[self.passenger_location]:
                self.passenger_location = self.num_locations
            else:
                return self._get_state(), -10, False, False, {}
        elif action == 5:
            if self.passenger_location == self.num_locations and (self.taxi_row, self.taxi_col) == self.locations[self.destination]:
                return self._get_state(), 20, True, False, {}
            else:
                return self._get_state(), -10, False, False, {}
        
        reward = -1
        done = False
        
        return self._get_state(), reward, done, False, {}

    def render(self):
        pass

    def unwrapped(self):
        return self
