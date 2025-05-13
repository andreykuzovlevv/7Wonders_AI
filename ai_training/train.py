# train.py
import torch
import numpy as np
import time
from collections import deque
import os

import config # Your config file
from .ppo_agent import PPOAgent 
from .game_simulator import SevenWondersSimulator

def main():
    # --- Initialization ---
    env = SevenWondersSimulator(level=config.LEVEL_1) # Adjust level
    
    device = config.DEVICE
    print(f"Using device: {device}")

    agent = PPOAgent(env_rows=config.GRID_ROWS, env_cols=config.GRID_COLS, device=device)
    
    # Create a directory for saving models if it doesn't exist
    model_save_dir = "ppo_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # --- Training Loop ---
    print(f"Starting training for {config.TOTAL_TRAINING_TIMESTEPS} timesteps...")
    start_time = time.time()
    
    state_tuple = env.reset()
    board_state_np, global_features_np = state_tuple[0], state_tuple[1]

    episode_rewards = deque(maxlen=100) # For logging average reward
    episode_lengths = deque(maxlen=100) # For logging average episode length
    current_episode_reward = 0
    current_episode_length = 0
    num_episodes = 0

    for timestep in range(1, config.TOTAL_TRAINING_TIMESTEPS + 1):
        # --- Collect Rollout ---
        # In PPO, we collect N_ROLLOUT_STEPS before updating.
        # This inner loop collects one step at a time.
        
        valid_swaps = env.get_valid_swaps()

        if len(valid_swaps) == 0:
            raise Exception("No valid swaps found. Skipping this step.")

        # Action selection
        action_swap_obj, action_idx, log_prob, value = agent.select_action(
            board_state_np, global_features_np, valid_swaps
        )

        # Environment step
        next_state_tuple, reward, done = env.step(action_swap_obj)
        next_board_state_np, next_global_features_np = next_state_tuple[0], next_state_tuple[1]
        
        current_episode_reward += reward
        current_episode_length += 1

        # Store transition
        # For the last step of a rollout, we need V(s_{t+1}) for GAE.
        # We are collecting N_ROLLOUT_STEPS. The "next_value" is for the state *after* the N_ROLLOUT_STEPS'th action.
        agent.store_transition(board_state_np, global_features_np, action_idx, log_prob, reward, value, done)
        
        board_state_np, global_features_np = next_board_state_np, next_global_features_np


        # --- Episode End or Rollout Full ---
        if done or current_episode_length >= config.MAX_MOVES_PER_EPISODE or len(agent.memory["actions"]) == config.N_ROLLOUT_STEPS:
            last_value = torch.tensor(0.0, device=device) # Scalar tensor for terminal state
            if not done and len(agent.memory["actions"]) == config.N_ROLLOUT_STEPS : # Rollout full, but episode not over
                next_board_state_tensor = torch.from_numpy(next_board_state_np).float().unsqueeze(0).to(device)
                next_global_features_tensor = torch.from_numpy(next_global_features_np).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    # agent.model.forward returns (logits, state_value_raw)
                    # state_value_raw is (1, 1) for batch size 1 input
                    _, last_value_raw = agent.model(next_board_state_tensor, next_global_features_tensor)
                    last_value = last_value_raw.squeeze() # Squeeze (1,1) to scalar ()
            
            if len(agent.memory["actions"]) >= config.N_ROLLOUT_STEPS or done:
                 agent.learn(last_value) # Pass the scalar tensor last_value

        if done or current_episode_length >= config.MAX_MOVES_PER_EPISODE:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            num_episodes += 1
            
            # Reset environment
            state_tuple = env.reset()
            board_state_np, global_features_np = state_tuple[0], state_tuple[1]
            current_episode_reward = 0
            current_episode_length = 0

        # --- Logging and Saving ---
        if timestep % config.LOG_FREQ == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else float('nan')
            avg_length = np.mean(episode_lengths) if episode_lengths else float('nan')
            elapsed_time = time.time() - start_time
            print(f"Timestep: {timestep}/{config.TOTAL_TRAINING_TIMESTEPS} | Episodes: {num_episodes}")
            print(f"Avg Reward (last 100): {avg_reward:.2f} | Avg Length (last 100): {avg_length:.2f}")
            print(f"Time: {elapsed_time:.2f}s | FPS: {timestep / elapsed_time:.2f}")
            print("-" * 40)

        if timestep % config.SAVE_MODEL_FREQ == 0:
            agent.save_model(os.path.join(model_save_dir, f"ppo_7wonders_ts{timestep}.pth"))

    # Final save
    agent.save_model(os.path.join(model_save_dir, f"ppo_7wonders_final.pth"))
    print("Training finished.")

if __name__ == "__main__":
    main()