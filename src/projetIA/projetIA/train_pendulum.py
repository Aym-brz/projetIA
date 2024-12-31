import torch
import torch.nn as nn
import torch.optim as optim
from pendulum_env import PendulumEnv
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # output between -1 and 1, scaled by action space
        )
        
    def forward(self, x):
        return self.network(x) * 100  # scale to action space

def train():
    env = PendulumEnv()
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            action = policy(state_tensor)
            state, reward, done, _ = env.step(action.detach().numpy())
            
            optimizer.zero_grad()
            loss = -reward  # Simple policy gradient
            loss.backward()
            optimizer.step()
            
            episode_reward += reward
            
        print(f"Episode {episode}, Reward: {episode_reward}")
        
        if episode % 100 == 0:
            torch.save(policy.state_dict(), f'pendulum_policy_{episode}.pth')

if __name__ == '__main__':
    train()