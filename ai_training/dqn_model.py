# --- File: dqn_model.py ---
import torch, torch.nn as nn, torch.nn.functional as F, config


print(f"Using device: {config.DEVICE}")

class QNetwork(nn.Module):
    def __init__(
        self, input_channels, action_dim=4, num_global_features=3
    ):  # e.g., stone_norm, fragment_on_board, step_count
        super().__init__()
        # CNN for board state
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            64 * config.GRID_ROWS * config.GRID_COLS, 128
        )  # Spatial features size

        # Branch for global state features
        self.fc_global1 = nn.Linear(num_global_features, 16)  # Global features size

        # Branch for action representation
        self.fc_action1 = nn.Linear(action_dim, 32)  # Action features size

        # Combine state (spatial + global) and action features
        combined_feature_size = 128 + 16 + 32
        self.fc_combine1 = nn.Linear(combined_feature_size, 64)
        self.fc_output = nn.Linear(64, 1)  # Output single Q-value

    def forward(self, state_tensor, global_features_tensor, action_tensor):
        # Process spatial state through CNN
        x_spatial = F.relu(self.conv1(state_tensor))
        x_spatial = F.relu(self.conv2(x_spatial))
        x_spatial = x_spatial.view(x_spatial.size(0), -1)  # Flatten
        x_spatial = F.relu(self.fc1(x_spatial))

        # Process global state features
        x_global = F.relu(self.fc_global1(global_features_tensor))

        # Process action
        x_action = F.relu(self.fc_action1(action_tensor))

        # Combine and produce Q-value
        # Ensure tensors are correctly shaped (batch_size, feature_size)
        x_combined = torch.cat((x_spatial, x_global, x_action), dim=1)
        x_combined = F.relu(self.fc_combine1(x_combined))
        q_value = self.fc_output(x_combined)
        return q_value
