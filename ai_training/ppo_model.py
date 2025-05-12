# ppo_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config # Assuming config.py is in the same directory or accessible

class ActorCritic(nn.Module):
    def __init__(self, num_planes: int, num_global_features: int, num_actions: int, grid_rows: int, grid_cols: int):
        super(ActorCritic, self).__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_planes = num_planes # e.g., 17

        # CNN for board state processing
        # Input: (batch_size, num_planes, grid_rows, grid_cols)
        self.conv1 = nn.Conv2d(num_planes, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Example: for 8x8 grid, after two 3x3 convs with padding, size is still 8x8
        # Flattened CNN output size will be 128 * grid_rows * grid_cols
        
        # Calculate flattened size dynamically
        # Create a dummy input to calculate the output size of CNN layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_planes, grid_rows, grid_cols)
            cnn_out_dummy = self.conv2(self.relu1(self.conv1(dummy_input)))
            self.flattened_cnn_output_size = cnn_out_dummy.numel() # numel() gives total number of elements

        # FC layers for combined features
        # Input to fc1: flattened_cnn_output + num_global_features
        self.fc1 = nn.Linear(self.flattened_cnn_output_size + num_global_features, 512)
        self.relu_fc1 = nn.ReLU()
        
        # Actor head (outputs logits for action probabilities)
        self.actor_head = nn.Linear(512, num_actions)
        
        # Critic head (outputs state value)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, board_state: torch.Tensor, global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic.
        board_state: (batch_size, num_planes, grid_rows, grid_cols)
        global_features: (batch_size, num_global_features)
        """
        # CNN path
        x_cnn = self.relu1(self.conv1(board_state))
        x_cnn = self.relu2(self.conv2(x_cnn))
        x_cnn_flat = x_cnn.reshape(x_cnn.size(0), -1) # Flatten

        # Concatenate CNN output with global features
        combined_features = torch.cat((x_cnn_flat, global_features), dim=1)
        
        # Shared FC layer
        x_fc = self.relu_fc1(self.fc1(combined_features))
        
        # Actor output (action logits)
        action_logits = self.actor_head(x_fc)
        
        # Critic output (state value)
        state_value = self.critic_head(x_fc)
        
        return action_logits, state_value

    def get_action_and_value(self, board_state: torch.Tensor, global_features: torch.Tensor, 
                             valid_action_indices: torch.Tensor = None, action: torch.Tensor = None
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy, log probability of that action, entropy of policy, and state value.
        board_state: (1, num_planes, grid_rows, grid_cols) or (batch_size, ...)
        global_features: (1, num_global_features) or (batch_size, ...)
        valid_action_indices: Tensor of indices that are valid. If None, all actions are considered.
                              Shape (num_valid_actions,) for single state, or used for masking in batch.
        action: (Optional) If provided, compute log_prob and entropy for this specific action. (batch_size,)
        """
        action_logits, state_value = self.forward(board_state, global_features)

        if valid_action_indices is not None:
            # Mask invalid actions: set their logits to a very small number (-infinity)
            # This is crucial if the number of valid actions changes per state
            mask = torch.ones_like(action_logits) * float('-inf')
            if action_logits.dim() > 1 and valid_action_indices.dim() == 1: # Batch of states, single list of valid_indices for all (e.g. during training)
                 mask[:, valid_action_indices] = 0
            elif action_logits.dim() == 1 and valid_action_indices.dim() == 1: # Single state
                 mask[valid_action_indices] = 0
            else: # Complex case: per-sample valid_action_indices for a batch - requires careful handling
                  # For simplicity, assuming valid_action_indices is a flat list for one sample,
                  # or applies to all in batch if action_logits.dim() > 1
                  # For production: use a list of tensors for valid_action_indices if they vary per sample in a batch
                  # Here, we assume valid_action_indices applies to a single sample or all batch items uniformly if logits are batched.
                  # If action_logits is (B, A) and valid_action_indices is a list of tensors for each B:
                  # for i in range(B): mask[i, valid_action_indices[i]] = 0
                  # This example assumes a simpler case.
                  # For single sample inference with dynamic valid actions:
                  if action_logits.dim() == 1: # (num_actions)
                      mask[valid_action_indices] = 0
                  else: # (batch_size, num_actions) - assuming same valid_indices for all in batch if single tensor
                      mask[:, valid_action_indices] = 0


            masked_logits = action_logits + mask
            probs = F.softmax(masked_logits, dim=-1)
        else:
            # If no valid_action_indices, assume all actions are possible (e.g., during loss calculation on old actions)
            probs = F.softmax(action_logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample() # Sample an action
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, state_value.squeeze(-1) # Squeeze value to (batch_size,)