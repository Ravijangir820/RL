import os
from config import GRID_SIZE, EPISODES, MAX_STEPS, Q_ALPHA, Q_GAMMA, Q_EPSILON, N_OPTIONS, OC_ALPHA, OC_GAMMA, OC_EPSILON, OC_BETA, LOG_DIR


def display_project_summary():
    """Display detailed project information and parameters."""
    
    summary = f"""
{'='*70}
                    HRL TAXI-v3 PROJECT SUMMARY
{'='*70}

PROJECT OVERVIEW:
  Title:  Hierarchical Reinforcement Learning with Options on Taxi-v3
  Type:   Comparison of Flat Q-Learning vs Option-Critic (HRL)
  Domain: Custom Configurable Taxi Environment

ENVIRONMENT CONFIGURATION:
  Grid Size:           {GRID_SIZE}x{GRID_SIZE}
  Total Locations:     4 (Pickup/Dropoff points)
  Action Space:        6 (North, South, East, West, Pickup, Dropoff)
  Max Steps/Episode:   {MAX_STEPS}

TRAINING CONFIGURATION:
  Total Episodes:      {EPISODES:,}
  Random Seed:         42

FLAT Q-LEARNING AGENT:
  Learning Rate (α):   {Q_ALPHA}
  Discount Factor (γ): {Q_GAMMA}
  Exploration (ε):     {Q_EPSILON}
  Model Type:          Tabular Q-Learning
  
HIERARCHICAL (OPTIONS) AGENT:
  Learning Rate (α):   {OC_ALPHA}
  Discount Factor (γ): {OC_GAMMA}
  Exploration (ε):     {OC_EPSILON}
  Number of Options:   {N_OPTIONS}
  Termination Prob (β):{OC_BETA}
  Model Type:          Option-Critic (Tabular)

KEY CONCEPTS:
  • Flat: Learns direct policy mapping state → primitive action
  • Options: Learns hierarchical policy with temporal abstraction
           - High-level: which option to execute
           - Low-level: which action within option
           - Termination: when to switch options

METRICS TRACKED:
  1. Return:              Cumulative reward per episode
  2. Steps:                Actions taken per episode (efficiency)
  3. Success Rate:         % of successful task completions
  4. Illegal Actions:      Rule violations (pickup/dropoff mistakes)
  5. Efficiency:           (Optimal steps / Actual steps) ratio
  6. Illegal Action Ratio: Proportion of illegal actions

EVALUATION METHOD:
  • Human Play:       Manual gameplay demonstration
  • Flat Agent Play:  Trained flat Q-learning agent gameplay
  • Options Agent:    Trained hierarchical RL agent gameplay
  • Metrics Plot:     6-subplot comparison of agent performance

EXPECTED OUTCOMES:
  • Evidence of learning (improving returns over episodes)
  • Comparison of sample efficiency between approaches
  • Analysis of learned behaviors and option specialization
  • Visualization of temporal abstraction benefits

PROJECT FILES:
  Core Algorithms:
    - q_learning.py:        Flat Q-Learning implementation
    - option_critic.py:     Option-Critic (HRL) implementation
    - custom_taxi_env.py:   Scalable Taxi environment
  
  Training & Evaluation:
    - train_flat.py:        Flat agent training script
    - train_options.py:     Options agent training script
    - taxi_ui_2d.py:        2D visualization & gameplay
    - plot_results.py:      Metrics visualization
  
  Configuration:
    - config.py:            Centralized parameters
    - utils.py:             Helper functions

OUTPUT FILES GENERATED:
  Logs:  {LOG_DIR}/flat_returns.csv
         {LOG_DIR}/options_returns.csv
  
  Checkpoints: checkpoints/flat_q.pkl
               checkpoints/options_q_u.pkl

{'='*70}
                   Press ENTER to continue...
{'='*70}
"""
    
    print(summary)
    try:
        input()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    display_project_summary()
