import gymnasium as gym
import numpy as np

from config import (
    CKPT_DIR,
    ENV_ID,
    EPISODES,
    GRID_SIZE,
    LOG_DIR,
    MAX_STEPS,
    Q_ALPHA,
    Q_EPSILON,
    Q_GAMMA,
    SEED,
)
from custom_taxi_env import CustomTaxiEnv
from q_learning import QLearningAgent
from utils import save_pickle, write_csv


def get_env():
    if GRID_SIZE == 5:
        return gym.make(ENV_ID)
    else:
        return CustomTaxiEnv(grid_size=GRID_SIZE)


def train():
    env = get_env()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=Q_ALPHA,
        gamma=Q_GAMMA,
        epsilon=Q_EPSILON,
        seed=SEED,
    )

    episode_metrics = []
    optimal_steps = 12  # Theoretical optimal for Taxi-v3
    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0
        steps = 0
        illegal_count = 0
        success = 0
        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if reward == -10:
                illegal_count += 1
            if done:
                success = 1 if terminated else 0
                break
        
        # Calculate efficiency (optimal / actual steps)
        efficiency = optimal_steps / steps if steps > 0 else 0
        efficiency = min(efficiency, 1.0)  # Cap at 100%
        
        # Illegal action ratio
        illegal_ratio = illegal_count / steps if steps > 0 else 0
        
        episode_metrics.append([ep, total_reward, steps, success, illegal_count, efficiency, illegal_ratio])

    env.close()

    save_pickle(agent.q, f"{CKPT_DIR}/flat_q.pkl")
    write_csv(
        f"{LOG_DIR}/flat_returns.csv",
        episode_metrics,
        ["episode", "return", "steps", "success", "illegal", "efficiency", "illegal_ratio"],
    )


if __name__ == "__main__":
    train()
