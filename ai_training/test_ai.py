# test_ai.py

import torch
import numpy as np
from .game_simulator import SevenWondersSimulator
from .dqn_agent import DQNAgent
import config

LEVELS = [config.LEVEL_1, config.LEVEL_2, config.LEVEL_3,
          config.LEVEL_4, config.LEVEL_5, config.LEVEL_6]

def test_model(model_path='q_local_final.pth', n_tests=10):
    # Initialize agent and load trained model
    agent = DQNAgent()
    agent.q_local.load_state_dict(torch.load(model_path))
    agent.q_local.eval()  # Set to evaluation mode
    
    # Dictionary to store results for each level
    level_results = {i+1: [] for i in range(len(LEVELS))}
    
    print("\nTesting model performance across levels...")
    print("-" * 50)
    
    # Test each level n_tests times
    for level_idx, level in enumerate(LEVELS):
        print(f"\nLevel {level_idx + 1}:")
        for test in range(n_tests):
            env = SevenWondersSimulator(level=level)
            state = env.get_state_tuple()
            steps = 0
            done = False
            
            while not done:
                valid_swaps = env.get_valid_swaps()
                action = agent.act(*state, valid_swaps, eps=0)  # No exploration during testing
                if action is None:
                    break
                    
                next_board, reward, done = env.step(action)
                next_state = env.get_state_tuple()
                state = next_state
                steps += 1
            
            level_results[level_idx + 1].append(steps)
            print(f"  Test {test + 1}: {steps} steps")
    
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    for level in range(1, len(LEVELS) + 1):
        avg_steps = np.mean(level_results[level])
        std_steps = np.std(level_results[level])
        print(f"Level {level}:")
        print(f"  Average steps: {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"  Min steps: {min(level_results[level])}")
        print(f"  Max steps: {max(level_results[level])}")
        print()

if __name__ == "__main__":
    test_model() 