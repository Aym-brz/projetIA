import torch
import torch.nn as nn
import torch.optim as optim
from tensordict.nn.distributions import NormalParamExtractor
from pendulum_env import PendulumEnv
from pendulum_env import max_speed
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from collections import deque

#plt.ion()
class Policy(nn.Module):
    def __init__(self, double_pendulum:bool=True):
        """
        Initializes the policy network. This network takes the state of the pendulum as input and outputs the action to be taken. The output is scaled to the action space by multiplying with 100.

        The network is composed of two hidden layers of size 64 with ReLU activation, and an output layer of size 1 with Tanh activation.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6 if double_pendulum else 4, 16),
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
        state = env.reset()  # État initial (vecteur de taille 6)
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        iter = 0
        best_reward = -np.inf
        while not done and iter < max_iter:
            sampled_action, log_prob = agent.act(state)

            # Appliquer l'action à l'environnement
            next_state, reward, done, _ = env.step(sampled_action.item(), num_sim_steps=num_sim_steps)
            
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

        if len(agent.memory) >= batch_size:
            for i in range(int(len(agent.memory)//batch_size)):
                    agent.update(agent.best, batch_size)
            agent.stddev = agent.stddev*0.98
        
        # Afficher le résultat périodiquement
        print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_episode_reward}")
        if (episode + 1) % 10 == 0:
            torch.save(policy.state_dict(), save_path)

    # Sauvegarde finale du modèle
    torch.save(policy.state_dict(), save_path)
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")
    
    return total_rewards

def evaluate_policy(policy: Policy, env: PendulumEnv, num_episodes: int = 10, max_iter: int = 2000, num_sim_step:int = 1):
    """
    Évalue la politique entraînée sur l'environnement du pendule.

    Arguments :
    - policy : une instance de la classe Policy (le réseau de neurones).
    - env : une instance de l'environnement (comme PendulumEnv).
    - num_episodes : nombre total d'épisodes d'évaluation.
    - max_iter : nombre maximum d'itérations par épisode.

    Retourne :
    - total_rewards : une liste contenant les récompenses totales pour chaque épisode.
    """
    policy.eval()  # Passer en mode évaluation (désactive dropout, batchnorm, etc.)
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        done = False
        iter = 0

        while not done and iter < max_iter:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy(state_tensor)
            next_state, reward, done, _ = env.step(action.item(), num_sim_steps=num_sim_step)
            episode_rewards.append(reward)
            state = next_state
            iter += 1

        total_episode_reward = sum(episode_rewards)
        total_rewards.append(total_episode_reward)
        print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_episode_reward}")

    return total_rewards

def main():
    rclpy.init()
    double_pendulum = False
    # Hyperparamètres
    num_episodes = 500
    discount_factor = 0.95
    learning_rate = 1e-3
    max_iter = 2000
    num_sim_step = 1
    stddev = 20
    save_path="trained_single_pendulum_policy.pth"
    batch_size = int(num_episodes/25)
    
    # Initialisation de l'environnement
    env = PendulumEnv(double_pendulum=double_pendulum)

    # Vérification des dimensions d'état et d'action
    state_dim = env.observation_space.shape[0]  # 6 pour le double pendule, 4 pour le simple
    action_dim = env.action_space.shape[0]      # 1 pour la vitesse du chariot

    print(f"Dimensions de l'état : {state_dim}, Dimensions de l'action : {action_dim}")

    # Initialisation de la politique
    policy = Policy(double_pendulum=double_pendulum)
    try:
        policy.load_state_dict(torch.load('best_'+save_path))
    except:
        pass



    # Entraînement de la politique
    total_rewards = train(policy, env, num_episodes=num_episodes, discount_factor=discount_factor, lr=learning_rate, max_iter=max_iter, num_sim_steps=num_sim_step, save_path=save_path, batch_size=batch_size, stddev=stddev)

    # Affichage des résultats
    plt.plot(total_rewards)
    plt.title("Évolution des récompenses totales")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.grid()
    plt.show()

    # Charger le modèle sauvegardé
    policy = Policy(double_pendulum=double_pendulum)  # Créer une nouvelle instance de Policy
    policy.load_state_dict(torch.load('best_'+save_path))  # Charger les poids

    # Évaluation de la politique entraînée
    evaluation_rewards = evaluate_policy(policy, env, num_episodes=10, max_iter=max_iter)

    # Affichage des résultats d'évaluation
    plt.plot(evaluation_rewards)
    plt.title("Évolution des récompenses totales en évaluation")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    main()
    #plt.ioff()
    plt.show()