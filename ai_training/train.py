# --- train.py ---
import numpy as np
from collections import deque
from .game_simulator import SevenWondersSimulator
from .dqn_agent import DQNAgent
from tqdm import tqdm
import cProfile
import pstats
import torch
env   = SevenWondersSimulator()
agent = DQNAgent(state_shape=env.get_state_representation().shape)

def train(n_episodes=1000, max_t=500,
          eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    scores, scores_window = [], deque(maxlen=100)
    eps = eps_start
    
    # Main progress bar for episodes
    main_pbar = tqdm(range(1, n_episodes+1), desc="Training Progress", position=0)
    
    # Display bars for metrics - position them below the main bar
    ep_bar = tqdm(total=0, bar_format='{desc}', position=1, leave=True)
    step_bar = tqdm(total=0, bar_format='{desc}', position=2, leave=True)
    score_bar = tqdm(total=0, bar_format='{desc}', position=3, leave=True)
    avg_bar = tqdm(total=0, bar_format='{desc}', position=4, leave=True)
    
    for i_ep in main_pbar:
        env.reset()
        state = env.get_state_tuple()
        score = 0
        
        # Update episode info
        ep_bar.set_description_str(f"Episode: {i_ep}/{n_episodes} | ε: {eps:.3f}")
        
        # Create step progress bar
        for t in range(max_t):
            valid_swaps = env.get_valid_swaps()
            action = agent.act(*state, valid_swaps, eps)
            if action is None: break               # no moves: dead board
            next_board, reward, done = env.step(action)
            next_state = env.get_state_tuple()
            valid_next = env.get_valid_swaps()     # for the *new* board
            agent.step(state, action, reward, next_state, done, valid_next)
            state, score = next_state, score + reward
            
            # Update step and score displays
            step_bar.set_description_str(f"Step: {t+1}/{max_t}")
            score_bar.set_description_str(f"Current Score: {score:.1f}")
            
            if done: break
        
        scores_window.append(score); scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        # Update average score display
        avg_score = np.mean(scores_window)
        avg_bar.set_description_str(f"Average Score: {avg_score:.2f}")
        
        # Update main progress bar with key info
        main_pbar.set_postfix(avg_score=f"{avg_score:.2f}", eps=f"{eps:.3f}")
        
        if i_ep % 100 == 0:
            torch.save(agent.q_local.state_dict(), 'q_local_weights.pth')
            # We can still print milestone messages that won't interfere with the progress bars
            tqdm.write(f"=== MILESTONE: Episode {i_ep}/{n_episodes} - Average Score: {avg_score:.2f} ===")

    # Save the model
    torch.save(agent.q_local.state_dict(), 'q_local_weights.pth')
    torch.save(agent.q_target.state_dict(), 'q_target_weights.pth') # Optional, can be re-init from local
    print("Saved trained model weights!")
    
    # Clean up progress bars
    main_pbar.close()
    ep_bar.close()
    step_bar.close()
    score_bar.close()
    avg_bar.close()
    
    print("\nTraining complete!")
    return scores

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    train()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative') # or 'tottime'
    stats.print_stats(20) # Print top 20 culprits
    # stats.dump_stats('training_profile.prof') # Optional: save for visualizer
