import matplotlib
import matplotlib.pyplot as plt 
import torch
import torch.optim as optim
from collections import deque
import numpy as np
from network import Policy
from pendulum_env import PendulumEnv

    
class REINFORCEAgent:
    def __init__(self, policy:Policy, best:Policy, discount_factor:int=0.99, lr:float=1e-3, memory_size:int=10000, stddev:int=20):
        """ 
        Initialise l'agent REINFORCE.

        Parameters:
        - discount_factor (float): Facteur d'actualisation pour les récompenses futures.
        - lr (float): Taux d'apprentissage.
        - memory_size (int): Taille maximale de la mémoire.
        """
        self.discount_factor = discount_factor
        self.stddev = stddev
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.best = best
        self.policy_network = policy
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

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
        action_distribution = torch.distributions.Normal(action, torch.tensor(self.stddev))  # Écart-type = 10
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
    
    def update(self, best:Policy, batch_size:int=25):
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
                discounted_sum = reward + self.discount_factor * discounted_sum
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
        self.policy_network.train()
        self.optimizer.zero_grad()
        total_policy_loss.backward()
        self.optimizer.step()

plt.ion()

def plot_reward(total_rewards, show_result=False):
    plt.figure(2)
    durations_t = torch.tensor(total_rewards, dtype=torch.float)
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

def train(policy:Policy, env:PendulumEnv, num_episodes:int=1000, discount_factor:float=0.99, lr:float=1e-3, max_iter:int=1000, num_sim_steps:int=1, save_path:str="trained_policy.pth", batch_size:int=25, stddev:int=20):
    """
    Entraîne le modèle Policy pour stabiliser un double pendule.
    
    Arguments :
    - policy : une instance de la classe Policy (le réseau de neurones).
    - env : une instance de l'environnement (comme PendulumEnv).
    - num_episodes : nombre total d'épisodes d'entraînement.
    - discount_factor : facteur d'actualisation pour les récompenses futures.
    - lr : taux d'apprentissage pour l'optimiseur.
    - max_iter : nombre maximum d'iterations par episode.
    - save_path : chemin vers le fichier de sauvegarde de la politique.
    
    Retourne :
    - total_rewards : une liste contenant les récompenses totales pour chaque épisode.
    """
    agent = REINFORCEAgent(policy=policy, best=policy, discount_factor=discount_factor, lr=lr, stddev=stddev)
    total_rewards = [] 

    for episode in range(num_episodes):
        # Réinitialisation de l'environnement
        state, _ = env.reset()  # État initial (vecteur de taille 6)
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        iter = 0
        best_reward = -np.inf
        while not done and iter < max_iter:
            sampled_action, log_prob = agent.act(state)

            # Appliquer l'action à l'environnement
            next_state, reward, done, _, _ = env.step(sampled_action.item(), num_sim_steps=num_sim_steps)
            
            # Enregistrer la récompense et la log-probabilité
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            
            # Passer à l'état suivant
            iter += 1
            state = next_state
        # Calcul de la récompense totale pour l'épisode
        total_episode_reward = sum(episode_rewards)
        total_rewards.append(total_episode_reward)
        if total_episode_reward > best_reward:
            torch.save(policy.state_dict(), "best_"+save_path)
            agent.best = policy

        agent.remember(state, sampled_action, episode_rewards, episode_log_probs)
        plot_reward(total_rewards=total_rewards)

        if len(agent.memory) >= batch_size:
            for i in range(int(len(agent.memory)//batch_size)):
                    agent.update(agent.best, batch_size)
            agent.stddev = agent.stddev*0.98
        
        # Afficher le résultat périodiquement
        print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_episode_reward}")
        if (episode + 1) % 10 == 0:
            torch.save(policy.state_dict(), f'{episode+1}_' + save_path)

    # Sauvegarde finale du modèle
    torch.save(policy.state_dict(), 'final_' + save_path)
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")  
    print('Complete')
    
    plot_reward(show_result=True)
    plt.savefig("plot_results\reinforce_training.png")
    plt.ioff()
    plt.show()
    return total_rewards

