import gymnasium as gym

from config import (
    CKPT_DIR,
    ENV_ID,
    EPISODES,
    GRID_SIZE,
    LOG_DIR,
    MAX_STEPS,
    N_OPTIONS,
    OC_ALPHA,
    OC_BETA,
    OC_EPSILON,
    OC_GAMMA,
    SEED,
)
from custom_taxi_env import CustomTaxiEnv
from option_critic import OptionCriticAgent
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

    agent = OptionCriticAgent(
        n_states=n_states,
        n_actions=n_actions,
        n_options=N_OPTIONS,
        alpha=OC_ALPHA,
        gamma=OC_GAMMA,
        epsilon=OC_EPSILON,
        beta=OC_BETA,
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
        option = agent.select_option(state)

        for _ in range(MAX_STEPS):
            action = agent.select_action(state, option)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, option, action, reward, next_state, done)
            total_reward += reward
            steps += 1
            if reward == -10:
                illegal_count += 1

            if done:
                success = 1 if terminated else 0
                break
            if agent.should_terminate():
                option = agent.select_option(next_state)
            state = next_state

        # Calculate efficiency (optimal / actual steps)
        efficiency = optimal_steps / steps if steps > 0 else 0
        efficiency = min(efficiency, 1.0)  # Cap at 100%
        
        # Illegal action ratio
        illegal_ratio = illegal_count / steps if steps > 0 else 0
        
        episode_metrics.append([ep, total_reward, steps, success, illegal_count, efficiency, illegal_ratio])

    env.close()

    save_pickle(agent.q_u, f"{CKPT_DIR}/options_q_u.pkl")
    write_csv(
        f"{LOG_DIR}/options_returns.csv",
        episode_metrics,
        ["episode", "return", "steps", "success", "illegal", "efficiency", "illegal_ratio"],
    )


if __name__ == "__main__":
    train()
