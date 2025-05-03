# --- File: dqn_agent.py ---
import numpy as np
import random
from collections import deque
import torch
import torch.optim as optim


class DQNAgent:
    def __init__(self, state_shape, action_representation_size, seed):
        self.state_shape = state_shape  # e.g., (channels, rows, cols)
        self.action_rep_size = action_representation_size  # e.g., 4 for coords
        self.seed = random.seed(seed)

        # Q-Network and Target Network
        self.qnetwork_local = QNetwork(state_shape[0]).to(
            device
        )  # device is 'cuda' or 'cpu'
        self.qnetwork_target = QNetwork(state_shape[0]).to(device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=LR
        )  # LR = Learning Rate

        # Replay memory
        self.memory = deque(maxlen=BUFFER_SIZE)  # BUFFER_SIZE e.g., 10000
        self.batch_size = BATCH_SIZE  # e.g., 64
        self.gamma = GAMMA  # Discount factor e.g., 0.99
        self.tau = TAU  # For soft update of target network e.g., 1e-3
        self.update_every = (
            UPDATE_EVERY  # How often to update the network e.g., 4 steps
        )

        self.t_step = 0  # Internal step counter

    def _action_to_tensor(self, action: Swap):
        # Convert swap coordinates ((r1,c1),(r2,c2)) into a flat tensor
        (r1, c1), (r2, c2) = action
        # Normalize coords? May help. e.g. / max_row or max_col
        return torch.tensor([[r1, c1, r2, c2]], dtype=torch.float32).to(device)

    def act(self, state, valid_swaps, eps=0.0):
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Exploitation: Choose best action based on Q-network
            best_q = -float("inf")
            best_action = None
            if not valid_swaps:
                return None  # Handle no valid moves case

            state_tensor = (
                torch.from_numpy(state).float().unsqueeze(0).to(device)
            )  # Add batch dim
            self.qnetwork_local.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for swap in valid_swaps:
                    action_tensor = self._action_to_tensor(swap)
                    q_value = self.qnetwork_local(state_tensor, action_tensor).item()
                    if q_value > best_q:
                        best_q = q_value
                        best_action = swap
            self.qnetwork_local.train()  # Set model back to train mode
            return best_action
        else:
            # Exploration: Choose random valid action
            return random.choice(valid_swaps) if valid_swaps else None

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # Need to store action representation suitable for buffer
        action_rep = list(action[0]) + list(action[1])  # Flatten action coords
        self.memory.append((state, action_rep, reward, next_state, done))

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.sample_from_memory()
                self.learn(experiences, self.gamma)

    def sample_from_memory(self):
        # Randomly sample a batch of experiences from memory
        return random.sample(self.memory, k=self.batch_size)

    def learn(self, experiences, gamma):
        states, action_reps, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        # Convert action reps back to tensors for network input
        action_tensors = torch.tensor(action_reps, dtype=torch.float32).to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        # --- Calculate Target Q-values ---
        # Get Q-values for NEXT state using TARGET network FOR ALL VALID next actions
        # This is the tricky part with dynamic actions!
        Q_targets_next_list = []
        with torch.no_grad():
            for i in range(self.batch_size):
                # We need the valid swaps for next_state[i] - Simulator needs a way to provide this!
                # valid_next_swaps = simulator.get_valid_swaps_for_state(next_states[i]) # Hypothetical function
                valid_next_swaps = (
                    []
                )  # FIXME: Need a way to get valid swaps for next_states in batch

                if not valid_next_swaps or dones[i]:
                    Q_targets_next_list.append(torch.tensor(0.0).to(device))
                else:
                    max_q_next = -float("inf")
                    next_state_tensor = next_states[i].unsqueeze(0)
                    for next_swap in valid_next_swaps:
                        next_action_tensor = self._action_to_tensor(next_swap)
                        q_next = self.qnetwork_target(
                            next_state_tensor, next_action_tensor
                        ).item()
                        max_q_next = max(max_q_next, q_next)
                    Q_targets_next_list.append(torch.tensor(max_q_next).to(device))

        # Unsqueeze needed if Q_targets_next_list isn't already a column vector
        Q_targets_next = torch.stack(Q_targets_next_list).unsqueeze(1)

        # Calculate target Q value: R + gamma * max_a' Q_target(s', a') * (1 - done)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # --- Calculate Predicted Q-values ---
        # Get expected Q values from LOCAL network for the *actions actually taken*
        Q_expected = self.qnetwork_local(states, action_tensors)

        # --- Calculate Loss ---
        loss = F.mse_loss(Q_expected, Q_targets)

        # --- Minimize loss ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Update target network ---
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        # Polyak averaging: θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
