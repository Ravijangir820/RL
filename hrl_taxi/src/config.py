import os

GRID_SIZE = int(os.getenv("TAXI_GRID_SIZE", "5"))
ENV_ID = "Taxi-v3"
SEED = 42

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Training
EPISODES = 2000
MAX_STEPS = 200

# Flat Q-learning
Q_ALPHA = 0.1
Q_GAMMA = 0.99
Q_EPSILON = 0.1

# Options (Option-Critic)
N_OPTIONS = 4
OC_ALPHA = 0.1
OC_GAMMA = 0.99
OC_EPSILON = 0.1
OC_BETA = 0.2

# Logging
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
FIG_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
