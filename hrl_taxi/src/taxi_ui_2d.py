import sys
import time

import gymnasium as gym
import pygame

from config import CKPT_DIR, ENV_ID, GRID_SIZE, MAX_STEPS, OC_BETA
from custom_taxi_env import CustomTaxiEnv
from option_critic import OptionCriticAgent
from q_learning import QLearningAgent
from utils import load_pickle


def get_env():
    if GRID_SIZE == 5:
        return gym.make(ENV_ID)
    else:
        return CustomTaxiEnv(grid_size=GRID_SIZE)

CELL_SIZE = 60
PADDING = 20
HUD_HEIGHT = 80

# Grid size will be set based on environment (default 5)
GRID_SIZE = 5
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * PADDING
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * PADDING + HUD_HEIGHT

ACTIONS = {
    0: "South",
    1: "North",
    2: "East",
    3: "West",
    4: "Pickup",
    5: "Dropoff",
}

LOCATIONS = None  # Will be set from environment in main()

COLORS = {
    "bg": (245, 245, 245),
    "grid": (30, 30, 30),
    "wall": (220, 53, 69),
    "taxi": (255, 199, 44),
    "passenger": (46, 134, 193),
    "destination": (220, 53, 69),
    "text": (20, 20, 20),
}

DEFAULT_WALLS = {
    (0, 1),
    (1, 1),
    (3, 0),
    (3, 2),
    (4, 0),
    (4, 2),
}


def choose_mode():
    print("Choose mode: human | flat | options")
    while True:
        mode = input("> ").strip().lower()
        if mode in {"human", "flat", "options"}:
            return mode
        print("Invalid mode. Enter: human, flat, or options.")


def load_flat_agent(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = load_pickle(f"{CKPT_DIR}/flat_q.pkl")
    if q is None:
        raise RuntimeError("Missing flat_q.pkl. Train with train_flat.py first.")
    agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.0)
    agent.q = q
    return agent


def load_options_agent(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_u = load_pickle(f"{CKPT_DIR}/options_q_u.pkl")
    if q_u is None:
        raise RuntimeError("Missing options_q_u.pkl. Train with train_options.py first.")
    agent = OptionCriticAgent(
        n_states=n_states,
        n_actions=n_actions,
        n_options=q_u.shape[1],
        alpha=0.1,
        gamma=0.99,
        epsilon=0.0,
        beta=OC_BETA,
    )
    agent.q_u = q_u
    return agent


def decode_state(env, state):
    taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(state)
    return taxi_row, taxi_col, pass_loc, dest_idx


def cell_rect(row, col):
    x = PADDING + col * CELL_SIZE
    y = PADDING + row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def draw_grid(screen):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = cell_rect(r, c)
            pygame.draw.rect(screen, COLORS["grid"], rect, 2)


def _parse_walls_from_desc(desc):
    if desc is None:
        return set()

    rows = []
    if hasattr(desc, "shape") and len(desc.shape) == 2:
        for r in range(desc.shape[0]):
            row_chars = []
            for c in range(desc.shape[1]):
                ch = desc[r, c]
                if isinstance(ch, (bytes, bytearray)):
                    row_chars.append(ch.decode("utf-8"))
                else:
                    row_chars.append(str(ch))
            rows.append("".join(row_chars))
    else:
        for row in desc:
            if isinstance(row, (bytes, bytearray)):
                rows.append(row.decode("utf-8"))
            else:
                rows.append(str(row))

    walls = set()
    for r, row in enumerate(rows[:GRID_SIZE]):
        if len(row) < 2 * GRID_SIZE - 1:
            continue
        for c in range(GRID_SIZE - 1):
            idx = 2 * c + 1
            if idx < len(row) and row[idx] == "|":
                walls.add((r, c))
    return walls


def get_walls(env):
    walls = _parse_walls_from_desc(getattr(env.unwrapped, "desc", None))
    if not walls:
        walls = DEFAULT_WALLS
    return walls


def draw_walls(screen, walls):
    for r, c in walls:
        x = PADDING + (c + 1) * CELL_SIZE
        y = PADDING + r * CELL_SIZE
        pygame.draw.line(
            screen,
            COLORS["wall"],
            (x, y),
            (x, y + CELL_SIZE),
            6,
        )


def draw_entities(screen, env, state):
    taxi_row, taxi_col, pass_loc, dest_idx = decode_state(env, state)

    if pass_loc < 4:
        p_row, p_col = LOCATIONS[pass_loc]
        rect = cell_rect(p_row, p_col)
        pygame.draw.circle(
            screen,
            COLORS["passenger"],
            rect.center,
            CELL_SIZE // 6,
        )

    d_row, d_col = LOCATIONS[dest_idx]
    rect = cell_rect(d_row, d_col)
    pygame.draw.rect(screen, COLORS["destination"], rect.inflate(-CELL_SIZE * 0.6, -CELL_SIZE * 0.6), 3)

    rect = cell_rect(taxi_row, taxi_col)
    taxi_rect = rect.inflate(-CELL_SIZE * 0.3, -CELL_SIZE * 0.3)
    pygame.draw.rect(screen, COLORS["taxi"], taxi_rect)

    if pass_loc == 4:
        pygame.draw.circle(screen, COLORS["passenger"], taxi_rect.center, CELL_SIZE // 10)


def draw_hud(screen, font, mode, step, reward, last_action, done):
    y = PADDING + GRID_SIZE * CELL_SIZE + 10
    text = f"Mode: {mode} | Step: {step} | Last: {last_action} | Reward: {reward}"
    if done:
        text += " | Episode finished"
    surface = font.render(text, True, COLORS["text"])
    screen.blit(surface, (PADDING, y))


def get_grid_size(env):
    """Extract grid size from environment (custom or wrapped Gym)."""
    if hasattr(env, 'grid_size'):
        return env.grid_size
    if hasattr(env.unwrapped, 'grid_size'):
        return env.unwrapped.grid_size
    # Default to 5 for standard Taxi-v3
    return 5


def get_locations(env):
    """Extract locations from environment (custom or wrapped Gym)."""
    if hasattr(env, 'locations'):
        return env.locations
    if hasattr(env.unwrapped, 'locations'):
        return env.unwrapped.locations
    # Default locations for 5x5 Taxi
    return [(0, 0), (0, 4), (4, 0), (4, 3)]


def action_from_key(key):
    if key == pygame.K_DOWN:
        return 0
    if key == pygame.K_UP:
        return 1
    if key == pygame.K_RIGHT:
        return 2
    if key == pygame.K_LEFT:
        return 3
    if key == pygame.K_p:
        return 4
    if key == pygame.K_d:
        return 5
    return None


def main():
    global GRID_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, LOCATIONS
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in {"human", "flat", "options"}:
            print(f"Invalid mode: {mode}")
            mode = choose_mode()
    else:
        mode = choose_mode()
    env = get_env()
    
    # Set GRID_SIZE and window dimensions based on environment
    GRID_SIZE = get_grid_size(env)
    WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * PADDING
    WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * PADDING + HUD_HEIGHT
    LOCATIONS = get_locations(env)

    agent = None
    if mode == "flat":
        agent = load_flat_agent(env)
    elif mode == "options":
        agent = load_options_agent(env)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Taxi-v3 UI")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    walls = get_walls(env)

    state, _ = env.reset()
    option = None
    done = False
    step = 0
    reward = 0
    last_action = "-"

    auto_delay = 0.5
    last_auto_time = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if mode == "human" and event.type == pygame.KEYDOWN and not done:
                action = action_from_key(event.key)
                if action is not None:
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    step += 1
                    last_action = ACTIONS[action]

        if mode in {"flat", "options"} and not done:
            now = time.time()
            if now - last_auto_time >= auto_delay:
                last_auto_time = now
                if mode == "flat":
                    action = agent.act(state)
                else:
                    if option is None:
                        option = agent.select_option(state)
                    action = agent.select_action(state, option)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1
                last_action = ACTIONS[action]

                if mode == "options" and not done and agent.should_terminate():
                    option = agent.select_option(state)

        screen.fill(COLORS["bg"])
        draw_grid(screen)
        draw_walls(screen, walls)
        draw_entities(screen, env, state)
        draw_hud(screen, font, mode, step, reward, last_action, done)
        pygame.display.flip()

        if done:
            time.sleep(1)
            running = False

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
