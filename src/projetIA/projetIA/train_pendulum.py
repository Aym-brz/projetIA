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

plt.ion()
class Policy(nn.Module):
    def __init__(self, double_pendulum:bool=True):
        """
        Initializes the policy network. This network takes the state of the pendulum as input and outputs the action to be taken. The output is scaled to the action space by multiplying with 100.

        The network is composed of two hidden layers of size 64 with ReLU activation, and an output layer of size 1 with Tanh activation.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6 if double_pendulum else 4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32,1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.network(x) * max_speed  # scale to action space
    

class REINFORCEAgent:
    def __init__(self, policy:Policy,  gamma=0.99, lr=1e-3, memory_size=10000):
        """
        Initialise l'agent REINFORCE.

        Parameters:
        - gamma (float): Facteur d'actualisation pour les récompenses futures.
        - lr (float): Taux d'apprentissage.
        - memory_size (int): Taille maximale de la mémoire.
        """
        self.gamma = gamma
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
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
        action_distribution = torch.distributions.Normal(action, torch.tensor([10.0]))  # Écart-type = 10
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
    
    def update(self, batch_size=25):
        # Sélectionner un batch aléatoire de transitions de la mémoire
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), size=batch_size, replace=False)
        policy_loss = []

        for idx in batch:
            state, action, reward, log_prob = self.memory[idx]
            for _,_,reward, _ in reversed(self.memory[idx]):
                discounted_sum = reward + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns = torch.tensor(returns, dtype=torch.float32)

            policy_loss_onebatch = []
            for (state, action, reward, log_prob), discounted_sum in zip(self.memory, returns):
                policy_loss_onebatch.append(-log_prob * discounted_sum)

            policy_loss.append(torch.cat(policy_loss_onebatch).sum())

        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.memory.clear()


def train(policy:Policy, env:PendulumEnv, num_episodes:int=1000, gamma:float=0.99, lr:float=1e-3, max_iter:int=1000, num_sim_steps:int=1, save_path:str="trained_policy.pth", batch_size:int=25):
    """
    Entraîne le modèle Policy pour stabiliser un double pendule.
    
    Arguments :
    - policy : une instance de la classe Policy (le réseau de neurones).
    - env : une instance de l'environnement (comme PendulumEnv).
    - num_episodes : nombre total d'épisodes d'entraînement.
    - gamma : facteur d'actualisation pour les récompenses futures.
    - lr : taux d'apprentissage pour l'optimiseur.
    - max_iter : nombre maximum d'iterations par episode.
    - save_path : chemin vers le fichier de sauvegarde de la politique.
    
    Retourne :
    - total_rewards : une liste contenant les récompenses totales pour chaque épisode.
    """
    agent = REINFORCEAgent(policy, gamma=gamma, lr=lr)
    # Optimiseur pour entraîner la politique
    optimizer = agent.optimizer
    
    # Liste pour suivre les récompenses totales
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
            # # Convertir l'état en tenseur PyTorch
            # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # # Générer l'action à partir de la politique
            # action = policy(state_tensor)

            # # Ajouter de la variabilité (exploration) avec une distribution normale
            # action_distribution = torch.distributions.Normal(action, torch.tensor([10.0]))  # Écart-type = 10
            # sampled_action = action_distribution.sample()  # Obtenir une action
            # log_prob = action_distribution.log_prob(sampled_action)  # Log-probabilité de l'action
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
            best_reward = total_episode_reward

        
        # # Calcul du retour (return) et mise à jour des gradients
        # returns = []
        # discounted_sum = 0
        # for r in reversed(episode_rewards):
        #     discounted_sum = r + gamma * discounted_sum
        #     returns.insert(0, discounted_sum)  # Insérer en début de liste
        
        # # Normalisation des retours pour stabilité numérique
        # returns = torch.tensor(returns, dtype=torch.float32)

        # # Calcul de la perte
        # policy_loss = []
        # for log_prob, cumulated_return in zip(episode_log_probs, returns):
        #     policy_loss.append(log_prob * cumulated_return)  # Perte pour chaque étape
        # # Moyenne sur toutes les étapes
        # policy_loss = torch.cat(policy_loss).sum()
        
        agent.remember(state, sampled_action, episode_rewards, episode_log_probs)

        if len(agent.memory) >= batch_size:
            for i in range(len(agent.memory)//batch_size):
                    agent.update(batch_size)
        # # Optimisation
        # optimizer.zero_grad()
        # policy_loss.backward()
        # optimizer.step()
        
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
    gamma = 0.99
    learning_rate = 1e-3
    max_iter = 2000
    num_sim_step = 1
    save_path="trained_single_pendulum_policy.pth"
    batch_size = num_episodes/20
    
    # Initialisation de l'environnement
    env = PendulumEnv(double_pendulum=double_pendulum)

    # Vérification des dimensions d'état et d'action
    state_dim = env.observation_space.shape[0]  # 6 pour le double pendule, 4 pour le simple
    action_dim = env.action_space.shape[0]      # 1 pour la vitesse du chariot

    print(f"Dimensions de l'état : {state_dim}, Dimensions de l'action : {action_dim}")

    # Initialisation de la politique
    policy = Policy(double_pendulum=double_pendulum)
    try:
        policy.load_state_dict(torch.load('best'+save_path))
    except:
        pass



    # Entraînement de la politique
    total_rewards = train(policy, env, num_episodes=num_episodes, gamma=gamma, lr=learning_rate, max_iter=max_iter, num_sim_steps=num_sim_step, save_path=save_path, batch_size=batch_size)

    # Affichage des résultats
    plt.plot(total_rewards)
    plt.title("Évolution des récompenses totales")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.grid()
    plt.show()

    # Charger le modèle sauvegardé
    policy = Policy(double_pendulum=double_pendulum)  # Créer une nouvelle instance de Policy
    policy.load_state_dict(torch.load(save_path))  # Charger les poids

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
    plt.ioff()
    plt.show()