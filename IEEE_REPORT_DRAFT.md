# Hierarchical Reinforcement Learning for Taxi-v3: A Comparative Study of Flat Q-Learning and Options

Ravi Jangid  
Registration No.: 22MIC7020  
Department CSE, VIT-AP UNIVERSITY

## Abstract
This report presents a reinforcement learning (RL) demonstration project on Taxi-v3, comparing a flat Q-learning baseline with a hierarchical RL approach based on options. The objective is to show how policy quality changes with training experience and to provide presentation-friendly artifacts such as trajectory GIFs and budget-wise comparison plots. We evaluate both agents across episode budgets of 1, 10, 100, and 3000, and track average return, success rate, steps per episode, illegal actions, efficiency, and illegal-action ratio. The project also includes an interactive 2D visualization and an automated pipeline for training, evaluation, and figure generation. Experimental results show that flat Q-learning converges faster in this implementation, while the options-based agent improves more slowly and remains sensitive to termination and exploration settings. We analyze these outcomes, discuss threats to validity, and present a practical roadmap to improve hierarchical performance through better option termination, multi-seed training, and ablation studies.

## Keywords
Reinforcement Learning, Hierarchical Reinforcement Learning, Option-Critic, Q-Learning, Taxi-v3, Temporal Abstraction

## I. Introduction
Reinforcement learning studies how agents learn sequential decision making through interaction with an environment. A classic pedagogical benchmark is Taxi-v3, where the agent must pick up and drop off a passenger while avoiding illegal actions. This task is small enough for tabular methods, but still highlights exploration-exploitation trade-offs and sparse reward challenges.

Hierarchical reinforcement learning (HRL) addresses long-horizon control by decomposing behavior into reusable sub-policies (options). In principle, HRL can improve credit assignment and sample efficiency. In practice, hierarchical methods can be sensitive to option design and termination behavior. This project compares a flat tabular agent and an options-based tabular agent using a unified environment, common reward structure, and shared evaluation metrics.

The main contributions of this project are:
1. End-to-end RL demonstration pipeline with training, evaluation, plotting, and 2D visualization.
2. Episode-budget comparison (1, 10, 100, 3000) with rollout GIFs for both agents.
3. Multi-metric analysis suited for classroom or viva demonstration.

This report is written for two goals: academic clarity and demonstration utility. Academic clarity means clear assumptions, reproducibility, and honest discussion of limitations. Demonstration utility means artifacts that visually communicate learning progress to a non-expert audience, including GIFs and training-budget curves.

## II. Related Work
Tabular Q-learning remains a foundational RL algorithm for finite Markov Decision Processes (MDPs). HRL methods such as the options framework introduce temporally extended actions and high-level control over low-level policies. Option-Critic-style methods aim to learn both intra-option policies and termination structure. This project implements a simplified tabular variant to compare against flat Q-learning on Taxi-v3.

The options framework is attractive for long-horizon tasks because it can reuse sub-skills. However, this advantage depends on good option design and sensible option termination behavior. In small tabular environments, a strong flat baseline can be very competitive, especially when hierarchical parameters are not learned end-to-end.

## III. Problem Setup
### A. Environment
- Environment: Taxi-v3 (default 5x5) with optional custom grid.
- State space: encoded taxi position, passenger status, and destination.
- Action space: 6 actions (north, south, east, west, pickup, dropoff).
- Episode horizon: 200 max steps.

### B. Reward Structure
- +20 for successful dropoff.
- -1 per step.
- -10 for illegal pickup/dropoff.

### C. Evaluation Metrics
- Average return.
- Success rate.
- Average steps.
- Average illegal actions.
- Efficiency ratio (optimal_steps / actual_steps, capped at 1.0).
- Illegal action ratio.

### D. Research Questions
1. How does policy quality change as training budget increases?
2. Does the options-based agent outperform flat Q-learning under limited training?
3. Which metrics are most informative for classroom demonstration?

## IV. Methodology
### A. Flat Q-Learning
The flat baseline uses tabular Q-learning with epsilon-greedy action selection:

Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]

### B. Options-Based Agent
The HRL agent maintains Q_U(s,o,a), where o is an option and a is a primitive action. The high-level controller selects options, and the low-level policy selects actions under the chosen option.

The simplified target uses a termination-weighted bootstrap:

target = r + gamma * ((1-beta) * max_a Q_U(s', o, a) + beta * max_o max_a Q_U(s', o, a))

### C. Training and Evaluation Protocol
For each episode budget in {1, 10, 100, 3000}, both agents are trained independently and then evaluated for a fixed number of episodes. Evaluation runs use deterministic action selection for fairness. The pipeline stores both numeric metrics and qualitative artifacts (GIF trajectories).

### D. Fairness Considerations
1. Same environment and reward function for both agents.
2. Same episode horizon and evaluation budget.
3. Same random seed base for reproducibility.
4. Separate training and evaluation phases.

### C. Hyperparameters (Current)
- alpha = 0.1
- gamma = 0.99
- flat epsilon = 0.1
- options epsilon start = 0.1, decay = 0.995, epsilon_min = 0.02
- options beta = 0.05
- number of options = 4
- default training episodes = 2000

### D. Computational Cost
The tabular setup is lightweight and runs on CPU. This makes the project suitable for laptops and classroom demonstrations without GPU requirements.

## V. Experimental Design
### A. Training Budgets
For demonstration, both agents are trained with limited budgets and then evaluated over multiple episodes:
- 1, 10, 100, 3000 episodes

### B. Artifacts Produced
- CSV: hrl_taxi/reports/figures/demo_episode_comparison/episode_budget_metrics.csv
- Plot: hrl_taxi/reports/figures/demo_episode_comparison/episode_budget_comparison.png
- GIF rollouts: flat_budget_*.gif and options_budget_*.gif

### C. Demonstration Storyline
1. Show GIFs for 1 and 10 episodes to illustrate random/immature behavior.
2. Show GIFs for 100 and 3000 episodes to illustrate policy improvement.
3. Show budget-comparison plot to connect qualitative behavior with quantitative metrics.
4. Conclude with why options underperform in this specific implementation and what changes can improve it.

### D. Reproducibility Command
python hrl_taxi/src/demo_episode_comparison.py --budgets 1 10 100 3000 --eval-episodes 200 --gif-fps 2

## VI. Results and Analysis
### A. Budget-wise Results (from current run)
| Agent | Budget | Avg Return | Avg Steps | Success Rate | Avg Illegal |
|---|---:|---:|---:|---:|---:|
| Flat | 1 | -218.00 | 200.00 | 0.000 | 2.00 |
| Options | 1 | -200.00 | 200.00 | 0.000 | 0.00 |
| Flat | 10 | -235.96 | 200.00 | 0.000 | 3.99 |
| Options | 10 | -380.00 | 200.00 | 0.000 | 20.00 |
| Flat | 100 | -235.91 | 200.00 | 0.000 | 3.99 |
| Options | 100 | -326.00 | 200.00 | 0.000 | 14.00 |
| Flat | 3000 | 7.03 | 13.86 | 0.995 | 0.00 |
| Options | 3000 | -126.56 | 93.46 | 0.565 | 4.99 |

### B. Metric Interpretation
1. Return combines efficiency and legality: more illegal moves and long episodes reduce return.
2. Success rate is the clearest indicator of task completion.
3. Steps captures trajectory quality: lower is better if success remains high.
4. Illegal actions indicate rule understanding and action discipline.
5. Efficiency is a normalized score useful for cross-budget comparison.

### C. Observations
1. At low budgets (1-100), both methods struggle, but options can be more unstable due to higher-dimensional value structure Q_U(s,o,a).
2. At 3000 episodes, flat Q-learning shows clear progress in return and success rate.
3. Options improve more slowly in this implementation and remain sensitive to option switching and exploration schedule.

### D. Why Options Underperform Here
1. The implementation uses fixed stochastic option termination instead of learned state-dependent termination.
2. The option-value space is larger than flat Q(s,a), requiring more samples.
3. Coupled high-level and low-level exploration increases variance early in training.

### E. Qualitative GIF Analysis
1. At low budgets, trajectories include frequent detours and occasional illegal operations.
2. At higher budgets, flat trajectories become shorter and more direct to passenger/destination landmarks.
3. Options trajectories improve but still show unnecessary switches and delayed convergence.

## VII. Limitations and Future Work
1. Add learned termination and intra-option policy gradients for stronger Option-Critic behavior.
2. Run multi-seed experiments and report mean +/- standard deviation.
3. Extend comparison to larger grids and transfer settings.
4. Add ablations over beta, number of options, and epsilon schedule.
5. Include statistical significance tests for final results.

Additional planned improvements:
1. Option-specific exploration schedules.
2. State-dependent termination models.
3. Better initialization for option-value tables.
4. Separate diagnostics for pickup and dropoff sub-goals.

## VIII. Threats to Validity
### A. Internal Validity
Performance is sensitive to hyperparameters, especially beta and epsilon decay. Different settings may change conclusions.

### B. External Validity
Taxi-v3 is a small discrete benchmark. Results may not generalize to continuous control or high-dimensional tasks.

### C. Construct Validity
Success rate alone can hide quality differences. Therefore, this report combines multiple metrics and GIF inspection.

## IX. How to Add GIFs in the Report
Yes, GIFs can be used in project reports. The correct method depends on output format:

1. Markdown report: GIFs can be embedded directly.
2. IEEE PDF (LaTeX): animated GIFs do not play in standard PDF viewers.
3. Recommended for IEEE PDF: include key frame images and provide link/QR code to GIF folder or repository.

Example Markdown embedding:
![Flat Agent 3000 Episodes](hrl_taxi/reports/figures/demo_episode_comparison/flat_budget_3000.gif)

Example strategy for IEEE PDF:
1. Add 2-3 representative static frames in a figure panel.
2. Add a caption saying full animations are available in repository artifacts.
3. Add the repository URL in a footnote or appendix.

## X. Conclusion
This project demonstrates a practical RL comparison framework with training visualization, rollout GIFs, and budget-sensitive performance analysis. The current experiments show that flat Q-learning converges faster than the simplified options agent under limited training budgets. The framework is suitable for academic demonstrations and can be extended to stronger HRL variants with learned option components.

## References
[1] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.

[2] R. S. Sutton, D. Precup, and S. Singh, Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning, Artificial Intelligence, vol. 112, no. 1-2, pp. 181-211, 1999.

[3] P.-L. Bacon, J. Harb, and D. Precup, The Option-Critic Architecture, in Proc. AAAI Conf. Artif. Intell., 2017.

[4] OpenAI Gym Taxi-v3 Environment Documentation.

---

## Appendix A: Project Files Used in This Report
- main.py
- hrl_taxi/src/config.py
- hrl_taxi/src/q_learning.py
- hrl_taxi/src/option_critic.py
- hrl_taxi/src/train_flat.py
- hrl_taxi/src/train_options.py
- hrl_taxi/src/demo_episode_comparison.py
- hrl_taxi/reports/figures/demo_episode_comparison/episode_budget_metrics.csv
