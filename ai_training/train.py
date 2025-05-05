# --- train.py ---
import numpy as np
from collections import deque
from game_simulator import SevenWondersSimulator
from dqn_agent import DQNAgent

env   = SevenWondersSimulator()
agent = DQNAgent(state_shape=env.get_state_representation().shape)

def train(n_episodes=1000, max_t=500,
          eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    scores, scores_window = [], deque(maxlen=100)
    eps = eps_start
    for i_ep in range(1, n_episodes+1):
        env.reset()
        state = env.get_state_tuple()
        score = 0
        for t in range(max_t):
            valid_swaps = env.get_valid_swaps()
            action = agent.act(*state, valid_swaps, eps)
            if action is None: break               # no moves: dead board
            next_board, reward, done = env.step(action)
            next_state = env.get_state_tuple()
            valid_next = env.get_valid_swaps()     # for the *new* board
            agent.step(state, action, reward, next_state, done, valid_next)
            state, score = next_state, score + reward
            if done: break
        scores_window.append(score); scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print(f"\rEp {i_ep}\tAvg: {np.mean(scores_window):.2f}", end="")
        if i_ep % 100 == 0:
            print(f"\rEp {i_ep}\tAvg: {np.mean(scores_window):.2f}")
    return scores

if __name__ == "__main__":
    train()
