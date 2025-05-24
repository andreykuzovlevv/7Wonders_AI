#!/usr/bin/env python3
"""
Test script to verify the expanded action space implementation.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_training'))

from ai_training.sw_gym_env import SevenWondersEnv
import config

def test_action_space():
    """Test the new action space with all 4 directions."""
    print("Testing expanded action space...")
    
    # Create environment
    env = SevenWondersEnv()
    
    # Basic info
    print(f"Grid size: {env.rows}x{env.cols}")
    print(f"Total action space size: {env.n_actions}")
    print(f"Expected max actions (4 directions per cell): {env.rows * env.cols * 4}")
    
    # Reset environment
    obs, info = env.reset()
    valid_swaps = env.sim.get_valid_swaps()
    action_mask = info["action_mask"]
    
    print(f"\nValid swaps found: {len(valid_swaps)}")
    print(f"Valid actions (masked): {action_mask.sum()}")
    
    # Show some example swaps
    print(f"\nFirst 10 swaps in action space:")
    for i, swap in enumerate(env.swap_list[:10]):
        print(f"  Action {i}: {swap[0]} → {swap[1]}")
    
    # Look for symmetric swaps
    print(f"\nChecking for symmetric swaps:")
    symmetric_pairs = 0
    for i, swap1 in enumerate(env.swap_list):
        reversed_swap = (swap1[1], swap1[0])
        if reversed_swap in env.swap_list:
            j = env.swap_list.index(reversed_swap)
            if i < j:  # Only count each pair once
                print(f"  {swap1} ↔ {reversed_swap} (actions {i}, {j})")
                symmetric_pairs += 1
                if symmetric_pairs >= 5:  # Limit output
                    break
    
    print(f"\nFound {symmetric_pairs}+ symmetric action pairs")
    
    # Test action encoding/decoding
    print(f"\nTesting action encoding/decoding:")
    test_swap = valid_swaps[0] if valid_swaps else env.swap_list[0]
    action_idx = env._encode_action(test_swap)
    decoded_swap = env._decode_action(action_idx)
    print(f"  Original: {test_swap}")
    print(f"  Encoded to action: {action_idx}")
    print(f"  Decoded back: {decoded_swap}")
    print(f"  Match: {test_swap == decoded_swap}")

if __name__ == "__main__":
    test_action_space() 