# viz_qnetwork.py
import torch
from torch.utils.tensorboard import SummaryWriter
from .dqn_model import QNetwork, device

# ---- build a dummy batch that matches your real inputs ----
BATCH = 1
state  = torch.zeros((BATCH, 2, 10, 10)).to(device)   # (B, C, H, W)
global_ = torch.zeros((BATCH, 2)).to(device)          # (B, num_global_features)
action  = torch.zeros((BATCH, 4)).to(device)          # (B, action_dim)

model = QNetwork(input_channels=2).to(device)

writer = SummaryWriter("runs/qnetwork_viz")
writer.add_graph(model, (state, global_, action))
writer.close()
