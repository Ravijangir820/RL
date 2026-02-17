import pygame
from config import GRID_SIZE, EPISODES, MAX_STEPS, Q_ALPHA, Q_GAMMA, Q_EPSILON, N_OPTIONS, OC_ALPHA, OC_GAMMA, OC_EPSILON, OC_BETA, LOG_DIR


def display_project_summary_gui():
    """Display project summary in a GUI window."""
    
    summary_text = [
        "HRL TAXI-v3 PROJECT SUMMARY",
        "",
        "PROJECT OVERVIEW:",
        "  Title:  Hierarchical Reinforcement Learning with Options on Taxi-v3",
        "  Type:   Comparison of Flat Q-Learning vs Option-Critic (HRL)",
        "  Domain: Custom Configurable Taxi Environment",
        "",
        "ENVIRONMENT CONFIGURATION:",
        f"  Grid Size:           {GRID_SIZE}x{GRID_SIZE}",
        "  Total Locations:     4 (Pickup/Dropoff points)",
        "  Action Space:        6 (North, South, East, West, Pickup, Dropoff)",
        f"  Max Steps/Episode:   {MAX_STEPS}",
        "",
        "TRAINING CONFIGURATION:",
        f"  Total Episodes:      {EPISODES:,}",
        "  Random Seed:         42",
        "",
        "FLAT Q-LEARNING AGENT:",
        f"  Learning Rate (α):   {Q_ALPHA}",
        f"  Discount Factor (γ): {Q_GAMMA}",
        f"  Exploration (ε):     {Q_EPSILON}",
        "  Model Type:          Tabular Q-Learning",
        "",
        "HIERARCHICAL (OPTIONS) AGENT:",
        f"  Learning Rate (α):   {OC_ALPHA}",
        f"  Discount Factor (γ): {OC_GAMMA}",
        f"  Exploration (ε):     {OC_EPSILON}",
        f"  Number of Options:   {N_OPTIONS}",
        f"  Termination Prob (β):{OC_BETA}",
        "  Model Type:          Option-Critic (Tabular)",
        "",
        "KEY CONCEPTS:",
        "  • Flat: Learns direct policy mapping state → primitive action",
        "  • Options: Learns hierarchical policy with temporal abstraction",
        "           - High-level: which option to execute",
        "           - Low-level: which action within option",
        "           - Termination: when to switch options",
        "",
        "METRICS TRACKED:",
        "  1. Return:              Cumulative reward per episode",
        "  2. Steps:                Actions taken per episode (efficiency)",
        "  3. Success Rate:         % of successful task completions",
        "  4. Illegal Actions:      Rule violations (pickup/dropoff mistakes)",
        "  5. Efficiency:           (Optimal steps / Actual steps) ratio",
        "  6. Illegal Action Ratio: Proportion of illegal actions",
        "",
        "EVALUATION METHOD:",
        "  • Human Play:       Manual gameplay demonstration",
        "  • Flat Agent Play:  Trained flat Q-learning agent gameplay",
        "  • Options Agent:    Trained hierarchical RL agent gameplay",
        "  • Metrics Plot:     6-subplot comparison of agent performance",
        "",
        "PROJECT FILES:",
        "  Core Algorithms:",
        "    - q_learning.py:        Flat Q-Learning implementation",
        "    - option_critic.py:     Option-Critic (HRL) implementation",
        "    - custom_taxi_env.py:   Scalable Taxi environment",
        "",
        "  Training & Evaluation:",
        "    - train_flat.py:        Flat agent training script",
        "    - train_options.py:     Options agent training script",
        "    - taxi_ui_2d.py:        2D visualization & gameplay",
        "    - plot_results.py:      Metrics visualization",
        "",
        "  Configuration:",
        "    - config.py:            Centralized parameters",
        "    - utils.py:             Helper functions",
        "",
        "OUTPUT FILES GENERATED:",
        f"  Logs:  {LOG_DIR}/flat_returns.csv",
        f"         {LOG_DIR}/options_returns.csv",
        "  Checkpoints: checkpoints/flat_q.pkl",
        "               checkpoints/options_q_u.pkl",
        "",
        "Press any key or close window to continue...",
    ]
    
    pygame.init()
    width, height = 1000, 900
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("HRL Taxi-v3 Project Summary")
    
    font_title = pygame.font.SysFont("consolas", 16, bold=True)
    font_text = pygame.font.SysFont("consolas", 12)
    
    clock = pygame.time.Clock()
    scroll_offset = 0
    scroll_speed = 20
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                scroll_offset += event.y * scroll_speed
        
        screen.fill((245, 245, 245))
        
        y = -scroll_offset + 20
        for line in summary_text:
            if line.isupper() and ":" not in line:
                surface = font_title.render(line, True, (0, 0, 0))
            else:
                surface = font_text.render(line, True, (30, 30, 30))
            
            if -100 < y < height + 100:
                screen.blit(surface, (20, y))
            y += 25
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    display_project_summary_gui()
