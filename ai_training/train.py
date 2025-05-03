# --- File: train.py ---

from game_simulator import SevenWondersSimulator
from dqn_agent import DQNAgent
from dqn_model import QNetwork
import config
import torch

# Setup environment, agent, hyperparameters (LR, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, UPDATE_EVERY, eps_start, eps_end, eps_decay)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = SevenWondersSimulator()
agent = DQNAgent(
    state_shape=env._get_state_representation().shape,
    action_representation_size=4,
    seed=0,
)  # Action rep size = 4 for coords


def train_dqn(
    n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            valid_swaps = env.get_valid_swaps()  # Get current valid swaps
            action = agent.act(state, valid_swaps, eps)
            if action is None:  # No valid moves
                break
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth') # Save model
    return scores


scores = train_dqn()
# Plot scores etc.
