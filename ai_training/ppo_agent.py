# ppo_agent.py
import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
import torch.nn.functional as F
import config
from .ppo_model import ActorCritic
# Make sure your game_simulator.SevenWondersSimulator can be imported or is defined
# For example, if it's in game_simulator.py:
# from game_simulator import SevenWondersSimulator, Swap (if Swap type hint is used)
Swap = Tuple[Tuple[int, int], Tuple[int, int]] # Define Swap type alias


class PPOAgent:
    def __init__(self, env_rows: int, env_cols: int, device: torch.device):
        self.device = device
        self.env_rows = env_rows
        self.env_cols = env_cols

        self.num_planes = config.N_PLANES
        self.num_global_features = config.NUM_GLOBAL_FEATURES
        
        # Action mapping
        self.all_possible_swaps_list, self.swap_to_idx_map = config._generate_all_possible_swaps(env_rows, env_cols)
        self.num_actions = len(self.all_possible_swaps_list)

        self.model = ActorCritic(self.num_planes, self.num_global_features, self.num_actions, 
                                 self.env_rows, self.env_cols).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

        # Storage for rollouts
        self.memory = {
            "board_states": [], "global_features": [], "actions": [], "log_probs": [],
            "rewards": [], "values": [], "dones": [], "next_values": []
        }
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_epsilon = config.CLIP_EPSILON
        self.n_epochs = config.N_EPOCHS_PPO
        self.minibatch_size = config.MINIBATCH_SIZE_PPO
        self.entropy_coeff = config.ENTROPY_COEFF
        self.value_loss_coeff = config.VALUE_LOSS_COEFF
    
    def _clear_memory(self):
        for key in self.memory:
            self.memory[key].clear()

    def store_transition(self, board_state, global_features, action_idx, log_prob, reward, value, done, next_value=None):
        self.memory["board_states"].append(torch.from_numpy(board_state).float().to(self.device))
        self.memory["global_features"].append(torch.from_numpy(global_features).float().to(self.device))
        # action_idx is an int, torch.tensor([action_idx]) is (1,). This is fine for b_actions later.
        self.memory["actions"].append(torch.tensor([action_idx], dtype=torch.long).to(self.device))
        
        # Ensure log_prob, reward, value, done are scalar tensors for consistent stacking
        self.memory["log_probs"].append(log_prob.squeeze().detach()) # log_prob from select_action is (1,) -> ()
        self.memory["rewards"].append(torch.tensor(reward, dtype=torch.float32).to(self.device)) # reward (float) -> scalar tensor ()
        self.memory["values"].append(value.squeeze().detach())      # value from select_action is (1,) -> ()
        self.memory["dones"].append(torch.tensor(done, dtype=torch.bool).to(self.device)) # done (bool) -> scalar tensor ()


    def select_action(self, board_state_np: np.ndarray, global_features_np: np.ndarray, 
                      valid_swaps_from_env: List[Swap]) -> Tuple[Swap, int, torch.Tensor, torch.Tensor]:
        """
        Selects an action using the current policy.
        Returns the chosen Swap object, its integer index, log probability, and state value.
        """
        board_state_tensor = torch.from_numpy(board_state_np).float().unsqueeze(0).to(self.device)
        global_features_tensor = torch.from_numpy(global_features_np).float().unsqueeze(0).to(self.device)

        valid_action_indices = []
        for swap_obj in valid_swaps_from_env:
            canonical_s = config.get_canonical_swap(swap_obj)
            if canonical_s in self.swap_to_idx_map:
                valid_action_indices.append(self.swap_to_idx_map[canonical_s])





        valid_action_indices_tensor = torch.tensor(valid_action_indices, dtype=torch.long).to(self.device)

        self.model.eval() # Set model to evaluation mode for action selection
        with torch.no_grad():
            action_idx_tensor, log_prob, _, value = self.model.get_action_and_value(
                board_state_tensor, global_features_tensor, valid_action_indices_tensor
            )
        self.model.train() # Set back to train mode

        chosen_action_idx = action_idx_tensor.item()
        chosen_swap_obj = self.all_possible_swaps_list[chosen_action_idx]
        
        return chosen_swap_obj, chosen_action_idx, log_prob, value


    def compute_gae_and_returns(self, last_value_for_rollout: torch.Tensor): # last_value_for_rollout should be scalar tensor ()
        """
        Computes GAE and returns for the collected rollout.
        last_value_for_rollout: Value of the state after the last action in the rollout.
                                If the last state was terminal, this should be 0. Shape: ()
        """
        rewards_t = torch.stack(self.memory["rewards"])     # Now (num_steps,)
        values_t = torch.stack(self.memory["values"])       # Now (num_steps,)
        dones_t = torch.stack(self.memory["dones"])         # Now (num_steps,)
        
        num_steps = len(rewards_t)
        advantages = torch.zeros_like(rewards_t)            # (num_steps,)
        returns = torch.zeros_like(rewards_t)               # (num_steps,)
        
        gae = 0.0 # Scalar
        next_value = last_value_for_rollout # Should be scalar tensor ()

        for t in reversed(range(num_steps)):
            mask = 1.0 - dones_t[t].float() # dones_t[t] is scalar, mask is scalar
            
            # All terms are scalar, delta is scalar
            delta = rewards_t[t] + self.gamma * next_value * mask - values_t[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae # gae remains scalar
            advantages[t] = gae
            returns[t] = gae + values_t[t] 
            
            next_value = values_t[t] # values_t[t] is scalar
            
        return advantages, returns # Both (num_steps,)

    def learn(self, last_value_for_rollout: torch.Tensor):
        advantages, returns = self.compute_gae_and_returns(last_value_for_rollout)
        
        # Normalize advantages (optional but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Collate batch data
        b_board_states = torch.stack(self.memory["board_states"])
        b_global_features = torch.stack(self.memory["global_features"])
        b_actions = torch.stack(self.memory["actions"]).squeeze() # (N_ROLLOUT_STEPS,)
        b_log_probs_old = torch.stack(self.memory["log_probs"]).squeeze() # (N_ROLLOUT_STEPS,)
        # b_values_old = torch.stack(self.memory["values"]).squeeze() # No longer needed explicitly here

        num_samples = len(b_actions)
        indices = np.arange(num_samples)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                # Get new log_probs, entropy, values from current policy for minibatch
                # We need to pass valid_action_indices=None here because we are evaluating
                # the log_prob of actions *already taken*. The policy might have changed.
                # The original `get_action_and_value` needs to be able to re-evaluate old actions.
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    b_board_states[mb_indices],
                    b_global_features[mb_indices],
                    action=b_actions[mb_indices] # Pass the actions taken
                )
                # new_values will be (minibatch_size,)
                # new_log_probs will be (minibatch_size,)
                # entropy will be (minibatch_size,)

                # PPO Surrogate Loss
                ratio = torch.exp(new_log_probs - b_log_probs_old[mb_indices])
                surr1 = ratio * advantages[mb_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (MSE)
                value_loss = F.mse_loss(new_values, returns[mb_indices])

                # Entropy Bonus (to encourage exploration)
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Clip gradients
                self.optimizer.step()
        
        self._clear_memory() # Clear memory after learning

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device) # Ensure model is on the correct device
        print(f"Model loaded from {path}")