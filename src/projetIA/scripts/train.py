import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))      
        x = self.fc2(x)
        return x

policy = Policy(4, 128, 1)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

def get_reward(state, action):
    # reward function
    pendulum_angle = state[0]
    cart_position = state[1]
    cart_velocity = state[2]
    angle_velocity = state[3]

    reward = 0
    if abs(pendulum_angle) < 0.2 and abs(cart_position) < 2.0:
        reward += 1
    if abs(angle_velocity) < 0.2 and abs(cart_velocity) < 0.2:
        reward += 1
    if abs(pendulum_angle) > 0.2 or abs(cart_position) > 2.0:
        reward -= 1
    return reward

for episode in range(1000):
    state = np.random.uniform(-1, 1, 4)
    actions = []
    rewards = []
    for t in range(200):
        action = policy(torch.tensor(state, dtype=torch.float32))
        state, reward, done, _ = env.step(action.item())
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    loss = 0
    for t in range(len(actions)):
        loss -= rewards[t] * actions[t].log_prob()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
