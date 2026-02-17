# HRL Taxi-v3 (Hierarchical Reinforcement Learning with Options)

## Overview

This project implements **Hierarchical Reinforcement Learning (HRL)** using the **Options framework (Option-Critic)** on a custom configurable Taxi environment. It compares hierarchical RL with flat Q-learning to demonstrate the benefits of temporal abstraction and improved credit assignment on long-horizon tasks.

## Key Concepts

### Flat Q-Learning
- Learns a direct policy: state → primitive action
- Single-level decision making
- Struggles with long-horizon tasks due to sparse rewards

### Hierarchical RL (Options)
- **High-level policy**: which option (sub-policy) to execute
- **Low-level policy**: which primitive action within the current option
- **Termination function**: when to switch to a new option
- Benefits: temporal abstraction, better credit assignment, reusable sub-skills

## Project Structure

```
rl/
├── main.py                          # Main orchestration pipeline
├── README.md                        # This file
├── hrl_taxi/
│   ├── src/
│   │   ├── config.py               # Configuration parameters
│   │   ├── custom_taxi_env.py      # Scalable Taxi environment
│   │   ├── q_learning.py           # Flat Q-learning agent
│   │   ├── option_critic.py        # Option-Critic (HRL) agent
│   │   ├── train_flat.py           # Flat agent training script
│   │   ├── train_options.py        # Options agent training script
│   │   ├── taxi_ui_2d.py           # 2D game UI (human/agent play)
│   │   ├── plot_results.py         # Training metrics visualization
│   │   ├── project_summary.py      # Terminal summary display
│   │   ├── project_summary_gui.py  # GUI summary window
│   │   └── utils.py                # Helper functions
│   ├── logs/                        # Training logs (generated)
│   ├── checkpoints/                 # Model checkpoints (generated)
│   └── reports/figures/             # Metric plots (generated)
├── tic-tac-toe/                    # Tic-Tac-Toe minimax agent (bonus)
└── requirements.txt                 # Python dependencies
```

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Full Pipeline

Execute the complete pipeline with one command:

```bash
python main.py
```

or specify a custom grid size:

```bash
python main.py --grid 8
```

### Pipeline Steps (6 stages):

1. **Human Play** – You control the Taxi agent (2D UI)
   - Arrow keys: move
   - `P`: pickup passenger
   - `D`: dropoff passenger

2. **Flat Q-Learning Training** – Agent learns optimal policy
   - 20,000 episodes of self-play
   - Tabular Q-learning with ε-greedy exploration

3. **Flat Agent Gameplay** – Watch trained flat agent play (2D UI)

4. **Options (HRL) Training** – Hierarchical agent learns with 4 options
   - 20,000 episodes of self-play
   - High-level: selects which option
   - Low-level: actions within option

5. **Options Agent Gameplay** – Watch trained options agent play (2D UI)

6. **Project Summary** – GUI window with all parameters and details

7. **Metrics Plot** – 6-subplot comparison of both agents

## Training Configuration

### Environment
- **Grid Size:** Configurable (default 5x5, max 10x10)
- **Action Space:** 6 (North, South, East, West, Pickup, Dropoff)
- **State Space:** Grid position + passenger location + destination
- **Max Steps/Episode:** 200
- **Reward Structure:**
  - +20 for successful dropoff
  - -1 per step (encourages efficiency)
  - -10 for illegal actions (pickup/dropoff mistakes)

### Flat Q-Learning
| Parameter | Value |
|-----------|-------|
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.99 |
| Exploration (ε) | 0.1 |
| Episodes | 20,000 |

### Hierarchical RL (Option-Critic)
| Parameter | Value |
|-----------|-------|
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.99 |
| Exploration (ε) | 0.1 |
| Number of Options | 4 |
| Termination Prob (β) | 0.2 |
| Episodes | 20,000 |

## Metrics Tracked

The system logs and plots 6 key metrics:

1. **Return** – Cumulative reward per episode (higher is better)
2. **Steps** – Actions per episode (lower is better for efficiency)
3. **Success Rate** – % of successful task completions (higher is better)
4. **Illegal Actions** – Rule violations per episode (lower is better)
5. **Efficiency** – (Optimal steps / Actual steps) ratio (closer to 1.0 is better)
6. **Illegal Action Ratio** – Proportion of illegal actions (lower is better)

## Running Individual Commands

You can also run components individually:

```bash
# Training
python hrl_taxi\src\train_flat.py
python hrl_taxi\src\train_options.py

# Gameplay (modes: human, flat, options)
python hrl_taxi\src\taxi_ui_2d.py human
python hrl_taxi\src\taxi_ui_2d.py flat
python hrl_taxi\src\taxi_ui_2d.py options

# Visualization
python hrl_taxi\src\plot_results.py

# Project Info
python hrl_taxi\src\project_summary_gui.py
```

## Output Files

### Training Logs (CSV)
- `hrl_taxi/logs/flat_returns.csv` – Flat agent metrics per episode
- `hrl_taxi/logs/options_returns.csv` – Options agent metrics per episode

### Model Checkpoints (Pickle)
- `hrl_taxi/checkpoints/flat_q.pkl` – Flat Q-table
- `hrl_taxi/checkpoints/options_q_u.pkl` – Options Q-table

### Metrics Plot
- `hrl_taxi/reports/figures/training_metrics.png` – 6-subplot comparison (interactive window)

## Expected Results

### Learning Curves
- Both agents should show increasing returns over episodes
- Options may take longer to converge but could be more sample-efficient

### Task Completion
- Successful episodes should increase as agents learn
- Illegal actions should decrease

### Efficiency
- Both agents should approach optimal play (efficiency → 1.0)
- Well-trained agents use ~12 steps (near theoretical optimum)

## Implementation Details

### Custom Taxi Environment
- Supports configurable grid sizes (5x5, 8x8, 10x10, etc.)
- Dynamic location placement for complexity
- Proper terminal state handling and reward structure

### Q-Learning (Flat)
- Tabular Q(s,a) representation
- ε-greedy action selection
- Temporal difference update rule

### Option-Critic (HRL)
- Tabular Q_U(s,o,a) for intra-option values
- High-level option selection: argmax_o max_a Q_U(s,o,a)
- Low-level action selection: argmax_a Q_U(s,option,a)
- Termination-aware bootstrapping in update

## Key Files Explained

### config.py
Central configuration hub. All hyperparameters defined here for easy tuning.

### custom_taxi_env.py
Implements Taxi environment with:
- Arbitrary grid sizes
- 4 location points with proper state encoding
- Terminal condition checking
- Reward calculation

### q_learning.py
Flat Q-learning agent:
- Maintains Q(state, action) table
- Implements ε-greedy policy
- Performs TD updates

### option_critic.py
Hierarchical RL agent:
- Maintains Q_U(state, option, action) table
- Option and action selection
- Probabilistic termination
- Termination-aware Q-learning update

### train_flat.py / train_options.py
Training loops that:
- Log metrics per episode (return, steps, success, illegal, efficiency, illegal_ratio)
- Save checkpoints after training
- Export CSV logs for analysis

### taxi_ui_2d.py
Pygame-based 2D visualization:
- Renders 5xN grid with walls
- Displays taxi, passenger, destination
- Supports human input or agent control
- Shows real-time statistics

### plot_results.py
Matplotlib visualization:
- 6 subplots for all metrics
- Moving average smoothing
- Raw + smoothed curves
- Interactive display

## Author Notes

This project demonstrates:
- ✓ Correct implementation of tabular Q-learning
- ✓ Proper hierarchical RL with options framework
- ✓ Comprehensive metric tracking and visualization
- ✓ Configurable environment for scaling complexity
- ✓ Clean, modular code architecture
- ✓ Full pipeline automation from training to visualization

## Future Enhancements

- Deep Q-Networks (DQN) for pixel input
- Multi-agent settings
- Option discovery (learning which options to use)
- Transfer learning across grid sizes
- Larger environments (ant-maze physics simulation)

## License

Use freely for educational purposes.

