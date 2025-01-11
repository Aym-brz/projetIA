"""
This module contains an attempt to implement the REINFORCE algorithm for training a policy network to stabilize a pendulum.
Note: This implementation is not used as it is not working. Another implementation replaced it.
Classes:
    REINFORCEAgent: An agent that uses the REINFORCE algorithm to learn a policy for the pendulum environment.
Functions:
    train: Trains the policy model to stabilize the pendulum using the REINFORCE algorithm.
"""
import matplotlib.pyplot as plt 
import torch
import torch.optim as optim
from collections import deque
import numpy as np
from network import FeedForwardNetwork
from pendulum_env import PendulumEnv

    
class REINFORCEAgent:
    def __init__(self, policy:FeedForwardNetwork, best:FeedForwardNetwork, **hyperparameters):
        """ 
        Initialise l'agent REINFORCE.

        Parameters:
        - discount_factor (float): Facteur d'actualisation pour les récompenses futures.
        - lr (float): Taux d'apprentissage.
        - memory_size (int): Taille maximale de la mémoire.
        """
        self._init_hyperparameters(hyperparameters)
        self.memory = deque(maxlen=self.MEM_SIZE)
        self.best = best
        self.policy_network = policy
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.LR)
        self.total_rewards = []
        
    def act(self, state):
        """
        Choisit une action en fonction de l'état actuel.

        Parameters:
        - state (np.ndarray): L'état courant.

        Returns:
        - action (int): L'action choisie.
        - log_prob (torch.Tensor): Log-probabilité de l'action.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.policy_network(state_tensor)
        action_distribution = torch.distributions.Normal(action, torch.tensor(self.STDDEV))
        sampled_action = action_distribution.sample()  # Obtenir une action
        log_prob = action_distribution.log_prob(sampled_action)  # Log-probabilité de l'action
            
        return sampled_action, log_prob
    
    def remember(self, state, sampled_action_episode, reward_episode, log_prob_episode):
        """
        Ajoute une transition à la mémoire.

        Parameters:
        - state (np.ndarray): L'état.
        - action_episode (int): L'action.
        - reward_episode (float): La récompense.
        - log_prob_episode (torch.Tensor): Log-probabilité de l'action.
        """
        self.memory.append((state, sampled_action_episode, reward_episode, log_prob_episode))
    
    def update(self, best:FeedForwardNetwork, batch_size:int=25):
        num_batches = len(self.memory) // batch_size
        #print(self.memory[:][2])
        for i in range(num_batches):
            #batch = self.memory[i * batch_size : (i + 1) * batch_size]
            batch = [self.memory[i] for i in range(i * batch_size, (i + 1) * batch_size)]            
            self._update_policy(batch)
    
        # Process leftover data
        if len(self.memory) % batch_size > 0:
            batch = self.memory[num_batches * batch_size :]
            self._update_policy(batch)
    
        self.memory.clear()
        
        
        # for idx in batch:
        #     _, _, rewards, log_probs = self.memory[idx]
        #     for reward in reversed(rewards):
        #         discounted_sum = reward + self.discount_factor * discounted_sum

        
    def _update_policy(self, batch):
        policy_loss = []
        for _, _, log_probs, rewards in batch:
            returns = []
            discounted_sum = 0
            # Compute discounted returns
            for reward in reversed(rewards):
                discounted_sum = reward + self.GAMMA * discounted_sum
                returns.append(discounted_sum)
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float32)
            #returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Compute policy loss
            log_probs = torch.tensor(log_probs, dtype=torch.float32, requires_grad=True)
            policy_loss_onebatch = -torch.sum(log_probs * returns)
            policy_loss.append(policy_loss_onebatch)
        total_policy_loss = torch.stack(policy_loss).sum()
        
        # backpropagate
        # self.policy_network.train()
        self.optimizer.zero_grad()
        total_policy_loss.backward()
        self.optimizer.step()
    
    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.BATCH_SIZE = 25
        self.GAMMA = 0.95
        self.LR = 1e-3
        self.MEM_SIZE = 10000
        self.MAX_EPISODE_LENGTH = 800
        self.STDDEV_START = 0.3
        self.STDDEV_END = 0.05
        self.NUM_SIM_STEPS = 5
        self.STDDEV = self.STDDEV_START
        
        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    def plot_reward(self, show_result=False):
        plt.figure(1)
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


def train(policy:FeedForwardNetwork, env:PendulumEnv, num_episodes:int=1000, save_path:str="saved_policies/reinforce", **hyperparameters):
    """
    Trains the Policy model to stabilize the pendulum.
    
    Arguments:
    - policy: an instance of the Policy class (the neural network).
    - env: an instance of the environment (such as PendulumEnv).
    - num_episodes: total number of training episodes.
    - save_path: path to the policy save file.
    - hyperparameters: dictionary containing the following hyperparameters:
        - BATCH_SIZE: batch size.
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
    agent = REINFORCEAgent(policy=policy, best=policy, hyperparameters=hyperparameters)
    best_reward = 0
    for episode in range(num_episodes):
        # Réinitialisation de l'environnement
        state, _ = env.reset()  # État initial (vecteur de taille 6)
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        iter = 0
        while not done and iter < agent.MAX_EPISODE_LENGTH:
            sampled_action, log_prob = agent.act(state)

            # Appliquer l'action à l'environnement
            next_state, reward, done, _, _ = env.step(sampled_action.item(), num_sim_steps=agent.NUM_SIM_STEPS)
            
            # Enregistrer la récompense et la log-probabilité
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            
            # Passer à l'état suivant
            iter += 1
            state = next_state
        # Calcul de la récompense totale pour l'épisode
        total_episode_reward = sum(episode_rewards)
        agent.total_rewards.append(total_episode_reward)
        if total_episode_reward > best_reward:
            torch.save(policy.state_dict(), f"{save_path}/best_policy_REINFORCE_{best_reward}.pth")
            agent.best = policy
            best_reward = total_episode_reward

        agent.remember(state, sampled_action, episode_rewards, episode_log_probs)
        agent.plot_reward()
        agent.STDDEV = agent.STDDEV_END + (agent.STDDEV_START - agent.STDDEV_END) * np.exp(-1. * episode/(num_episodes/5))

        if len(agent.memory) >= agent.BATCH_SIZE:
            for i in range(int(len(agent.memory)//agent.BATCH_SIZE)):
                    agent.update(agent.best, agent.BATCH_SIZE)
        
        # print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_episode_reward}")
        # Afficher le résultat périodiquement
        if (episode + 1) % 10 == 0:
            torch.save(policy.state_dict(), f"{save_path}/policy_REINFORCE_{episode+1}.pth")

    # Sauvegarde finale du modèle
    torch.save(policy.state_dict(), f"{save_path}/final_policy_REINFORCE.pth")
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")  
    print('Complete')
    
    agent.plot_reward(show_result=True)
    plt.savefig(f"{save_path}/reinforce_training.png")
    plt.ioff()
    plt.show()
    return agent.total_rewards

