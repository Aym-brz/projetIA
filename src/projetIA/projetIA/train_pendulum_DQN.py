# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt 
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network import DQN_NN
from pendulum_env import PendulumEnv


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ Sample a batch of transitions from the memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN_Agent:
    def __init__(self, env, policy_net: DQN_NN, target_net:DQN_NN, **hyperparameters):
        """
        Initialize the DQN_Agent.

        Parameters:
        - env (PendulumEnv): The environment to use for training.
        - policy_net (DQN_NN): The policy network to use for training.
        - target_net (DQN_NN): The target network to use for training.
        - hyperparameters (dict): A dictionary containing the following hyperparameters:
            - BATCH_SIZE (int): Batch size.
            - GAMMA (float): Discount factor.
            - LR (float): Learning rate.
            - MEM_SIZE (int): Memory size.
            - TARGET_UPDATE (int): Interval at which to update the target network.
        """
        self._init_hyperparameters(hyperparameters)
        self.policy_net = policy_net
        self.target_net = target_net    
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(self.MEM_SIZE)
        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []
        self.env = env

    def select_action(self, state):
        """
        Selects an action based on the current state of the environment.

        The action is chosen with epsilon-greedy policy: with probability
        `eps_threshold` the action is chosen randomly, and with probability
        `1 - eps_threshold` the action is chosen as the one with the highest
        expected reward according to the current policy model.

        Args:
        - state: a tensor of size `(BATCH_SIZE, STATE_DIM)` representing the current state of the environment.
        Returns:
        - action: a tensor of size `(BATCH_SIZE, 1)` representing the chosen action.
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def plot_durations(self, show_result=False):
        """
        Plot the durations for each episode.
        Args:
            show_result (bool): If True, show the result (default is False).
        """
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    def plot_reward(self, show_result=False):
        """
        Plot the rewards for each episode.
        Args:
        - show_result (bool): If True, show the result (default is False).
        """
        plt.figure(2)
        durations_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        
    def optimize_model(self):
        """
        Perform a single optimization step of the policy model.

        This method is the core of the Q-learning algorithm. It samples a batch of transitions from the memory, computes the expected Q values, and updates the policy network weights to minimize the difference between the expected and predicted Q values.

        If the memory contains less than BATCH_SIZE transitions, the method does nothing.
        """
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        batch = Transition(*zip(*transitions)) # Transpose the batch 

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.  Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1).values.This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # In-place gradient clipping
        self.optimizer.step()

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters        
        Args:
            hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.BATCH_SIZE = 256
        self.MAX_EPISODE_LENGTH = 800
        self.GAMMA = 0.995
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 8000
        self.TAU = 0.005
        self.LR = 0.0003
        self.MEM_SIZE = 20000
        self.NUM_SIM_STEPS = 5
        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
   


def train(policy_net: DQN_NN, target_net: DQN_NN, env:PendulumEnv, num_episodes:int=5000, save_path:str="saved_policies/DQN", **hyperparameters):
    """
    Trains the Policy model to stabilize the pendulum.
    
    Arguments:
    - policy: an instance of the Policy class (the neural network).
    - env: an instance of the environment (such as PendulumEnv).
    - num_episodes: total number of training episodes.
    - save_path: path to the policy save file.
    - hyperparameters: dictionary containing the following hyperparameters:
        - BATCH_SIZE: batch size (in episodes).
        - MAX_EPISODE_LENGTH: maximum length of an episode.
        - NUM_SIM_STEPS: number of simulation step per action
        - GAMMA: discount factor.
        - LR: learning rate.
        - MEM_SIZE: memory size.
        - STDDEV_START: initial standard deviation for action sampling.
        - STDDEV_END: final standard deviation (exponential decay over the training).

    Returns:
    - total_rewards: a list containing the total rewards for each episode.
    """
    plt.ion()
    best_reward = 0
    agent = DQN_Agent(env, policy_net, target_net, hyperparameters=hyperparameters)
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_episode_reward = 0
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item(), num_sim_steps=agent.NUM_SIM_STEPS)
            total_episode_reward += reward
            reward = torch.tensor([reward])
            done = terminated or truncated

            if t>=agent.MAX_EPISODE_LENGTH:
                done = True
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Soft update of the target network's weights  θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*agent.TAU + target_net_state_dict[key]*(1-agent.TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                agent.episode_durations.append(t + 1)
                agent.episode_rewards.append(total_episode_reward)
                agent.plot_reward()
                break
        # Save the model if best
        if total_episode_reward > best_reward:
            best_reward = total_episode_reward
            torch.save(policy_net.state_dict(), f"{save_path}/best_policy_DQN_{best_reward:.2f}.pth")
            torch.save(target_net.state_dict(), f"{save_path}/best_target_DQN_{best_reward:.2f}.pth")
        # # Save the model every 10 episodes
        # if i_episode % 10 == 0:
        #     torch.save(policy_net.state_dict(), f"{save_path}/policy_DQN_{i_episode}_{total_episode_reward:.2f}.pth")
        #     torch.save(target_net.state_dict(), f"{save_path}/target_DQN_{i_episode}_{total_episode_reward:.2f}.pth")
    print('Complete')
    # agent.plot_durations()
    agent.plot_reward(show_result=True)
    plt.savefig(f"{save_path}/DQN_training.png")
    plt.ioff()
    plt.show()
    
    torch.save(policy_net.state_dict(), f"{save_path}/final_policy_DQN.pth")
    torch.save(target_net.state_dict(), f"{save_path}/final_target_DQN.pth")
    
    return agent.episode_rewards