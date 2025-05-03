# --- File: dqn_model.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(
        self, input_channels, num_actions_placeholder=None
    ):  # num_actions not used directly in output
        super().__init__()
        # CNN layers to process the board state (input_channels depends on state representation)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Flatten and add Fully Connected layers
        # The input size to FC layers depends on grid size and conv output
        self.fc1 = nn.Linear(
            64 * config.GRID_ROWS * config.GRID_COLS, 128
        )  # Adjust size!

        # Branch to process action representation (simple example: coordinates)
        # Action representation needs to be defined - e.g., 4 numbers for (r1,c1,r2,c2)
        self.fc_action = nn.Linear(4, 32)  # Example size

        # Combine state and action features
        self.fc_combine1 = nn.Linear(128 + 32, 64)
        self.fc_output = nn.Linear(64, 1)  # Output single Q-value

    def forward(self, state_tensor, action_tensor):
        # Process state through CNN
        x_state = F.relu(self.conv1(state_tensor))
        x_state = F.relu(self.conv2(x_state))
        x_state = x_state.view(x_state.size(0), -1)  # Flatten
        x_state = F.relu(self.fc1(x_state))

        # Process action
        x_action = F.relu(self.fc_action(action_tensor))

        # Combine and produce Q-value
        x_combined = torch.cat((x_state, x_action), dim=1)
        x_combined = F.relu(self.fc_combine1(x_combined))
        q_value = self.fc_output(x_combined)
        return q_value
