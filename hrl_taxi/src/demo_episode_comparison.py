import argparse
import os

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from config import (
    ENV_ID,
    EPISODES,
    FIG_DIR,
    GRID_SIZE,
    MAX_STEPS,
    N_OPTIONS,
    OC_ALPHA,
    OC_BETA,
    OC_EPSILON_DECAY,
    OC_EPSILON_MIN,
    OC_EPSILON,
    OC_GAMMA,
    Q_ALPHA,
    Q_EPSILON,
    Q_GAMMA,
    SEED,
    STUCK_PATIENCE,
)
from custom_taxi_env import CustomTaxiEnv
from option_critic import OptionCriticAgent
from q_learning import QLearningAgent
from utils import ensure_dir, write_csv


OPTIMAL_STEPS = 12


def taxi_position(env, state):
    try:
        taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
        return int(taxi_row), int(taxi_col)
    except Exception:
        return int(state), -1


def get_env(render_mode=None):
    if GRID_SIZE == 5:
        if render_mode is None:
            return gym.make(ENV_ID)
        return gym.make(ENV_ID, render_mode=render_mode)
    return CustomTaxiEnv(grid_size=GRID_SIZE)


def train_flat(episodes):
    env = get_env()
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=Q_ALPHA,
        gamma=Q_GAMMA,
        epsilon=Q_EPSILON,
        seed=SEED,
    )

    for ep in range(episodes):
        state, _ = env.reset(seed=SEED + ep)
        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    env.close()
    return agent


def train_options(episodes):
    env = get_env()
    agent = OptionCriticAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        n_options=N_OPTIONS,
        alpha=OC_ALPHA,
        gamma=OC_GAMMA,
        epsilon=OC_EPSILON,
        beta=OC_BETA,
        seed=SEED,
    )

    for ep in range(episodes):
        agent.epsilon = max(OC_EPSILON_MIN, agent.epsilon * OC_EPSILON_DECAY)
        state, _ = env.reset(seed=SEED + ep)
        option = agent.select_option(state)
        for _ in range(MAX_STEPS):
            action = agent.select_action(state, option)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, option, action, reward, next_state, done)

            if done:
                break

            if agent.should_terminate():
                option = agent.select_option(next_state)
            state = next_state

    env.close()
    return agent


def to_eval_flat_agent(train_agent):
    eval_agent = QLearningAgent(
        n_states=train_agent.n_states,
        n_actions=train_agent.n_actions,
        alpha=train_agent.alpha,
        gamma=train_agent.gamma,
        epsilon=0.0,
        seed=SEED,
    )
    eval_agent.q = np.array(train_agent.q, copy=True)
    return eval_agent


def to_eval_options_agent(train_agent):
    eval_agent = OptionCriticAgent(
        n_states=train_agent.n_states,
        n_actions=train_agent.n_actions,
        n_options=train_agent.n_options,
        alpha=train_agent.alpha,
        gamma=train_agent.gamma,
        epsilon=0.0,
        beta=0.0,
        seed=SEED,
    )
    eval_agent.q_u = np.array(train_agent.q_u, copy=True)
    return eval_agent


def run_eval_episode(agent, mode, env, seed):
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    illegal = 0
    success = 0
    option = None
    stuck_steps = 0
    prev_pos = taxi_position(env, state)

    for _ in range(MAX_STEPS):
        if mode == "flat":
            action = agent.act(state)
        else:
            if option is None:
                option = agent.select_option(state)
            action = agent.select_action(state, option)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward
        steps += 1

        current_pos = taxi_position(env, state)
        if current_pos == prev_pos:
            stuck_steps += 1
        else:
            stuck_steps = 0
        prev_pos = current_pos

        if stuck_steps >= STUCK_PATIENCE:
            done = True

        if reward == -10:
            illegal += 1

        if mode == "options" and not done and agent.should_terminate():
            option = agent.select_option(state)

        if done:
            success = 1 if terminated else 0
            break

    efficiency = min(OPTIMAL_STEPS / steps, 1.0) if steps > 0 else 0.0
    illegal_ratio = (illegal / steps) if steps > 0 else 0.0

    return {
        "return": total_reward,
        "steps": steps,
        "success": success,
        "illegal": illegal,
        "efficiency": efficiency,
        "illegal_ratio": illegal_ratio,
    }


def evaluate_agent(agent, mode, eval_episodes):
    env = get_env()
    per_episode = []
    for ep in range(eval_episodes):
        metrics = run_eval_episode(agent, mode, env, seed=SEED + 10000 + ep)
        per_episode.append(metrics)
    env.close()
    return per_episode


def summarize(metrics_list):
    returns = np.array([m["return"] for m in metrics_list], dtype=np.float32)
    steps = np.array([m["steps"] for m in metrics_list], dtype=np.float32)
    success = np.array([m["success"] for m in metrics_list], dtype=np.float32)
    illegal = np.array([m["illegal"] for m in metrics_list], dtype=np.float32)
    efficiency = np.array([m["efficiency"] for m in metrics_list], dtype=np.float32)
    illegal_ratio = np.array([m["illegal_ratio"] for m in metrics_list], dtype=np.float32)

    return {
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(success)),
        "avg_illegal": float(np.mean(illegal)),
        "avg_efficiency": float(np.mean(efficiency)),
        "avg_illegal_ratio": float(np.mean(illegal_ratio)),
    }


def capture_rollout_gif(agent, mode, out_path, fps=2):
    if GRID_SIZE != 5:
        return False

    env = get_env(render_mode="rgb_array")
    state, _ = env.reset(seed=SEED + 777)
    frames = [env.render()]
    option = None
    stuck_steps = 0
    prev_pos = taxi_position(env, state)

    for _ in range(MAX_STEPS):
        if mode == "flat":
            action = agent.act(state)
        else:
            if option is None:
                option = agent.select_option(state)
            action = agent.select_action(state, option)

        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state

        current_pos = taxi_position(env, state)
        if current_pos == prev_pos:
            stuck_steps += 1
        else:
            stuck_steps = 0
        prev_pos = current_pos

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = terminated or truncated
        if stuck_steps >= STUCK_PATIENCE:
            done = True
        if mode == "options" and not done and agent.should_terminate():
            option = agent.select_option(state)
        if done:
            break

    env.close()

    if not frames:
        return False

    ensure_dir(os.path.dirname(out_path))
    imageio.mimsave(out_path, frames, fps=fps)
    return True


def plot_budget_comparison(rows, budgets, out_path):
    metrics = [
        ("avg_return", "Average Return"),
        ("success_rate", "Success Rate"),
        ("avg_steps", "Average Steps"),
        ("avg_illegal", "Average Illegal Actions"),
        ("avg_efficiency", "Average Efficiency"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    axes = axes.ravel()

    by_agent = {"flat": {}, "options": {}}
    for row in rows:
        by_agent[row[0]][row[1]] = {
            "avg_return": row[2],
            "avg_steps": row[3],
            "success_rate": row[4],
            "avg_illegal": row[5],
            "avg_efficiency": row[6],
            "avg_illegal_ratio": row[7],
        }

    for idx, (metric_key, label) in enumerate(metrics):
        ax = axes[idx]
        for agent_name in ("flat", "options"):
            y = [by_agent[agent_name][b][metric_key] for b in budgets]
            if metric_key == "success_rate":
                y = [v * 100.0 for v in y]
            ax.plot(budgets, y, marker="o", label=agent_name.capitalize())

        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_title(label)
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("%" if metric_key == "success_rate" else label)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].axis("off")
    fig.suptitle("Episode Budget Comparison: Flat vs Options", fontsize=14)
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL behavior across episode budgets and export demo artifacts."
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 10, 100, 1000,3000],
        help="Episode budgets for comparison.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=200,
        help="Number of evaluation episodes per budget.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(FIG_DIR, "demo_episode_comparison"),
        help="Directory for CSV, plot, and GIF exports.",
    )
    parser.add_argument("--gif-fps", type=int, default=2, help="Playback speed for rollout GIFs.")

    args = parser.parse_args()

    budgets = sorted(set([b for b in args.budgets if b > 0]))
    if not budgets:
        raise ValueError("At least one positive budget is required.")

    max_default = max(EPISODES, 10000)
    budgets = [b for b in budgets if b <= max_default]
    if not budgets:
        raise ValueError(f"All budgets exceed configured limit ({max_default}).")

    ensure_dir(args.output_dir)

    csv_rows = []
    for budget in budgets:
        print(f"\n=== Budget {budget} episodes ===")

        flat_train_agent = train_flat(budget)
        flat_eval_agent = to_eval_flat_agent(flat_train_agent)
        flat_eval = evaluate_agent(flat_eval_agent, mode="flat", eval_episodes=args.eval_episodes)
        flat_summary = summarize(flat_eval)

        options_train_agent = train_options(budget)
        options_eval_agent = to_eval_options_agent(options_train_agent)
        options_eval = evaluate_agent(options_eval_agent, mode="options", eval_episodes=args.eval_episodes)
        options_summary = summarize(options_eval)

        csv_rows.append(
            [
                "flat",
                budget,
                flat_summary["avg_return"],
                flat_summary["avg_steps"],
                flat_summary["success_rate"],
                flat_summary["avg_illegal"],
                flat_summary["avg_efficiency"],
                flat_summary["avg_illegal_ratio"],
            ]
        )
        csv_rows.append(
            [
                "options",
                budget,
                options_summary["avg_return"],
                options_summary["avg_steps"],
                options_summary["success_rate"],
                options_summary["avg_illegal"],
                options_summary["avg_efficiency"],
                options_summary["avg_illegal_ratio"],
            ]
        )

        flat_gif = os.path.join(args.output_dir, f"flat_budget_{budget}.gif")
        options_gif = os.path.join(args.output_dir, f"options_budget_{budget}.gif")

        flat_saved = capture_rollout_gif(flat_eval_agent, "flat", flat_gif, fps=args.gif_fps)
        options_saved = capture_rollout_gif(options_eval_agent, "options", options_gif, fps=args.gif_fps)

        if GRID_SIZE != 5:
            print("Skipping GIF generation for custom grid. GIF export supports Gym Taxi-v3 (5x5).")
        else:
            print(f"Saved: {flat_gif}" if flat_saved else "Flat GIF skipped.")
            print(f"Saved: {options_gif}" if options_saved else "Options GIF skipped.")

    csv_path = os.path.join(args.output_dir, "episode_budget_metrics.csv")
    write_csv(
        csv_path,
        csv_rows,
        [
            "agent",
            "episode_budget",
            "avg_return",
            "avg_steps",
            "success_rate",
            "avg_illegal",
            "avg_efficiency",
            "avg_illegal_ratio",
        ],
    )

    plot_path = os.path.join(args.output_dir, "episode_budget_comparison.png")
    plot_budget_comparison(csv_rows, budgets, plot_path)

    print("\nDemo artifacts created:")
    print(f"- Metrics CSV: {csv_path}")
    print(f"- Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()