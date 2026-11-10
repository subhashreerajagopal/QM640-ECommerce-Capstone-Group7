"""
QM 640 Capstone: PyTorch Deep Q-Network (DQN) Multi-Agent Environment
Author: Subhashree.R (Group 7) | Walsh College
"""

import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class DQNAgent(nn.Module):
        """
        Multi-Layer Perceptron value-approximation network for MARL dynamic dispatch.
        Input state: 6 dims [x_coord, y_coord, payload, traffic_idx, elapsed_time, sla_window]
        Output actions: 4 discrete actions [North, South, East, West]
        """
        def __init__(self, state_dim=6, action_dim=4):
            super(DQNAgent, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            
        def forward(self, state):
            return self.network(state)
            
except ImportError:
    pass
