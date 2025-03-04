import torch
import torch.optim as optim
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt 
from network import FeedForwardNetwork
from pendulum_env import PendulumEnv

class REINFORCEAgent:
    def __init__(self, policy:FeedForwardNetwork, **hyperparameters):
        """
        Initialize the REINFORCEAgent.

        Parameters:
        - policy (FeedForwardNetwork): The policy network.
        - hyperparameters (dict): A dictionary containing the following hyperparameters:
            - BATCH_SIZE (int): Batch size.
            - MAX_EPISODE_LENGTH (int): Maximum length of an episode.
            - GAMMA (float): Discount factor.
            - LR (float): Learning rate.
            - MEM_SIZE (int): Memory size.
            - STDDEV_START (float): Initial standard deviation for action sampling.
            - STDDEV_END (float): Final standard deviation (exponential decay over the training).
        """
        self._init_hyperparameters(hyperparameters)
        self.memory = deque(maxlen=self.MEM_SIZE)
        self.policy_network = policy
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.LR)
        self.total_rewards = []
        
    def act(self, state):
        """
        Choose an action based on the current state of the environment. 

        Args:
        - state (np.ndarray): Current state

        Returns:
        - sampled_action (int): Chosen action.
        - log_prob (torch.Tensor): Log-probability of the action.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.policy_network(state_tensor)
        action_distribution = torch.distributions.Normal(action, torch.tensor(self.STDDEV))  # standard deviation
        sampled_action = action_distribution.sample()  # Obtain an action
        log_prob = action_distribution.log_prob(sampled_action)  # Log-probability of the action
            
        return sampled_action, log_prob
    
    def remember(self, state, action, reward, discounted_sum, time_step):
        """
        Stores a transition in the replay memory.

        Args:
        - state (np.ndarray): The current state of the environment.
        - action (int): The action taken at the current state.
        - reward (float): The reward received after taking the action.
        - discounted_sum (float): The discounted cumulative return for the episode.
        - time_step (int): The time step at which this transition occurred.
        """
        self.memory.append((state, action, reward, discounted_sum, time_step))


    def update(self, batch_size):
        """
        Update the policy network using a mini-batch of experiences.

        Args:
        - batch_size (int): size of the mini-batch for the training
        """
        if len(self.memory) < batch_size:
            return

        # Take a random sample from the memory
        minibatch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        returns = []

        # Preparation of mini-batch
        for state, action, reward, discounted_sum, time_step in minibatch:
            states.append(state)
            actions.append(action)
            returns.append(discounted_sum)

        # Converting data to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Compute the log-probabilities of the actions
        action_probs = self.policy_network(states_tensor)
        action_log_probs = torch.log(action_probs)

        # Extraire les log-probabilités des actions prises

        # Compute of the loss : -log(pi(a|s)) * G
        loss = -torch.mean(action_log_probs * returns_tensor)

        # Update of the model weights
        self.optimizer.zero_grad()
        loss.backward()
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
        self.BATCH_SIZE = 25
        self.GAMMA = 0.995
        self.LR = 0.0003
        self.MEM_SIZE = 10000
        self.MAX_EPISODE_LENGTH = 800
        self.STDDEV_START = 0.3
        self.STDDEV_END = 0.05
        self.STDDEV = self.STDDEV_START
        self.NUM_SIM_STEPS = 5

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))


    def plot_reward(self, show_result=False):
        plt.figure(2)
        durations_t = torch.tensor(self.total_rewards, dtype=torch.float)
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

def train(policy:FeedForwardNetwork, env:PendulumEnv, num_episodes:int=1000, save_path:str="saved_policies/reinforce2", **hyperparameters):
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
        - GAMMA: discount factor.
        - LR: learning rate.
        - MEM_SIZE: memory size.
        - STDDEV_START: initial standard deviation for action sampling.
        - STDDEV_END: final standard deviation (exponential decay over the training).

    Returns:
    - total_rewards: a list containing the total rewards for each episode.
    """
    plt.ion()
    agent = REINFORCEAgent(policy=policy, hyperparameters=hyperparameters)
    best_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_memory = []
        episode_reward = 0
        done = False
        iter = 0
        while not done and iter < agent.MAX_EPISODE_LENGTH:
            action, log_prob = agent.act(state)
            next_state, reward, done, _ , _= env.step(action, agent.NUM_SIM_STEPS)
            episode_memory.append((state, action, reward, log_prob))
            state = next_state
            episode_reward += reward
            iter += 1
        agent.total_rewards.append(episode_reward)
        agent.plot_reward()
        if episode_reward > best_reward:
            torch.save(policy.state_dict(), f"{save_path}/best_policy_REINFORCE2_{best_reward}.pth")
            best_reward = episode_reward

        agent.STDDEV = agent.STDDEV_END + (agent.STDDEV_START - agent.STDDEV_END) * np.exp(-1. * episode/(num_episodes/5))

        # Campute the discounted returns for the episode and store them in the replay memory. 
        discounted_sum = 0
        for i in reversed(range(len(episode_memory))):
            state, action, reward, log_prob = episode_memory[i]
            discounted_sum = reward + agent.GAMMA * discounted_sum
            agent.remember(state, action, reward, discounted_sum, i)

        # Update the policy 
        if len(agent.memory) >= agent.BATCH_SIZE:
            agent.update(agent.BATCH_SIZE)

        # # print(f"Épisode {episode + 1}/{num_episodes}, Récompense : {episode_reward}")
        # if (episode + 1) % 10 == 0:
        #     torch.save(policy.state_dict(), f"{save_path}/policy_REINFORCE2_{episode +1}.pth")

    # Final save
    torch.save(policy.state_dict(), f"{save_path}/final_policy_REINFORCE2.pth")
    print(f"Training finished. Model saved in {save_path}")

    agent.plot_reward(show_result=True)
    plt.savefig(f"{save_path}/reinforce2_training.png")
    plt.ioff()
    plt.show()
    
    return agent.total_rewards