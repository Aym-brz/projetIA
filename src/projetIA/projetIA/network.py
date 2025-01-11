import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, double_pendulum:bool=True):
        """
        Initializes the policy network. This network takes the state of the pendulum as input and outputs the action to be taken.

        The network is composed of two hidden layers of size 64 with ReLU activation, and an output layer of size 1 with Tanh activation.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7 if double_pendulum else 5, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128,1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.network(x) 
    

class DQN_NN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_NN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    
def main():
    double_pendulum = False
    save_path = 'pendulum_policy.pth'
    policy = FeedForwardNetwork(double_pendulum=double_pendulum)
    try:
        policy.load_state_dict(torch.load('best_'+save_path))
    except:
        print('No policy found')