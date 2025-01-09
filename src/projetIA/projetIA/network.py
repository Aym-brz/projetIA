import torch
import torch.nn as nn
import numpy as np
from pendulum_env import PendulumEnv
from pendulum_env import max_speed


#plt.ion()
class Policy(nn.Module):
    def __init__(self, double_pendulum:bool=True):
        """
        Initializes the policy network. This network takes the state of the pendulum as input and outputs the action to be taken. The output is scaled to the action space by multiplying with max_speed.

        The network is composed of two hidden layers of size 64 with ReLU activation, and an output layer of size 1 with Tanh activation.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7 if double_pendulum else 5, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16,1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.network(x) * max_speed  # scale to action space

def main():
    double_pendulum = False
    save_path = 'pendulum_policy.pth'
    policy = Policy(double_pendulum=double_pendulum)
    try:
        policy.load_state_dict(torch.load('best_'+save_path))
    except:
        print('No policy found')