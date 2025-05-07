# --- train.py --------------------------------------------------------------
import random, numpy as np, torch
from collections import deque
from .game_simulator import SevenWondersSimulator
from .dqn_agent import DQNAgent
from tqdm import trange
import config

LEVELS = [config.LEVEL_1, config.LEVEL_2, config.LEVEL_3,
          config.LEVEL_4, config.LEVEL_5]

env   = SevenWondersSimulator(level=LEVELS[0])   # dummy init – will reset each ep
agent = DQNAgent()

def train(n_episodes=2000, max_t=400,
          eps_start=1.0, eps_end=0.05, eps_decay=0.997):

    scores, scores_window = [], deque(maxlen=100)
    eps = eps_start

    pbar = trange(1, n_episodes + 1, desc="Episodes")
    for i_ep in pbar:
        # pick level (simple curriculum: cycle 1‑5)
        level_id = (i_ep - 1) % len(LEVELS)
        env = SevenWondersSimulator(level=LEVELS[level_id])

        state = env.get_state_tuple()
        score = 0
        for t in range(max_t):
            valid_swaps = env.get_valid_swaps()
            action = agent.act(*state, valid_swaps, eps)
            if action is None: break
            next_board, reward, done = env.step(action)
            next_state = env.get_state_tuple()
            valid_next = env.get_valid_swaps()
            agent.step(state, action, reward, next_state, done, valid_next)

            state, score = next_state, score + reward
            if done: break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        pbar.set_postfix({
            "lvl":  level_id + 1,
            "steps": t + 1,
            "score": f"{score:.0f}",
            "avg(100)": f"{np.mean(scores_window):.1f}",
            "ε": f"{eps:.3f}"
        })

        if i_ep % 200 == 0:
            torch.save(agent.q_local.state_dict(), f"q_local_ep{i_ep}.pth")

    torch.save(agent.q_local.state_dict(), 'q_local_final.pth')
    print("Training complete!")
    return scores

if __name__ == "__main__":
    train()
