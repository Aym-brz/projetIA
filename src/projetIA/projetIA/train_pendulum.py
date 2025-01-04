import torch
import torch.nn as nn
import torch.optim as optim
from pendulum_env import PendulumEnv
from pendulum_env import max_speed
import numpy as np
import matplotlib.pyplot as plt
import rclpy

class Policy(nn.Module):
    def __init__(self):
        """
        Initializes the policy network. This network takes the state of the pendulum as input and outputs the action to be taken. The output is scaled to the action space by multiplying with 100.

        The network is composed of two hidden layers of size 64 with ReLU activation, and an output layer of size 1 with Tanh activation.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # output between -1 and 1, scaled by action space
        )
        
    def forward(self, x):
        return self.network(x) * max_speed  # scale to action space


def train(policy:Policy, env:PendulumEnv, num_episodes:int=1000, gamma:float=0.99, lr:float=1e-3, max_iter:int=1000, save_path:str="trained_policy.pth"):
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
    # Optimiseur pour entraîner la politique
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Liste pour suivre les récompenses totales
    total_rewards = []

    for episode in range(num_episodes):
        # Réinitialisation de l'environnement
        state = env.reset()  # État initial (vecteur de taille 6)
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        iter = 0
        while not done and iter < max_iter:
            # Convertir l'état en tenseur PyTorch
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Générer l'action à partir de la politique
            action = policy(state_tensor)
            
            if torch.isnan(action).any():
                print("action nan")
                print(f"action:{action}")
            if torch.isnan(state_tensor).any():
                print("state_tensor nan")
                print(f"state tensor:{state_tensor}")

            
            # Ajouter de la variabilité (exploration) avec une distribution normale
            action_distribution = torch.distributions.Normal(action, torch.tensor([10.0]))  # Écart-type = 10
            sampled_action = action_distribution.sample()  # Obtenir une action
            log_prob = action_distribution.log_prob(sampled_action)  # Log-probabilité de l'action
            
            # Appliquer l'action à l'environnement
            next_state, reward, done, _ = env.step(sampled_action.item(), num_sim_steps=10)
            
            if torch.isnan(torch.tensor(next_state)).any():
                print("torch state nab bext state")
            if torch.isnan(torch.tensor(reward)).any():
                print("torch state nan reward")
            
            # Enregistrer la récompense et la log-probabilité
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            
            # Passer à l'état suivant
            iter += 1
            state = next_state
            if done:
                print(f"Episode {episode + 1} done after {iter} iterations")
        
        # Calcul de la récompense totale pour l'épisode
        total_episode_reward = sum(episode_rewards)
        total_rewards.append(total_episode_reward)
        
        # Calcul du retour (return) et mise à jour des gradients
        returns = []
        discounted_sum = 0
        for r in reversed(episode_rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)  # Insérer en début de liste
        
        # Normalisation des retours pour stabilité numérique
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calcul de la perte
        policy_loss = []
        for log_prob, Gt in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * Gt)  # Perte pour chaque étape
        
        # Moyenne sur toutes les étapes
        policy_loss = torch.cat(policy_loss).sum()
        
        # Optimisation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Afficher le résultat périodiquement
        if (episode + 1) % 100 == 0:
            torch.save(policy.state_dict(), save_path)
            print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_episode_reward}")
    
    # Sauvegarde finale du modèle
    torch.save(policy.state_dict(), save_path)
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")
    
    return total_rewards


rclpy.init()
# Initialisation de l'environnement
env = PendulumEnv()

# Vérification des dimensions d'état et d'action
state_dim = env.observation_space.shape[0]  # Par exemple, 6 pour le double pendule
action_dim = env.action_space.shape[0]      # Par exemple, 1 pour une force sur le chariot

print(f"Dimensions de l'état : {state_dim}, Dimensions de l'action : {action_dim}")

# Initialisation de la politique
policy = Policy()

# Hyperparamètres
num_episodes = 1000
gamma = 0.99
learning_rate = 1e-3
max_iter = 1000
save_path="trained_policy.pth"


# Entraînement de la politique
total_rewards = train(policy, env, num_episodes=num_episodes, gamma=gamma, lr=learning_rate, max_iter=max_iter, save_path=save_path)

# Affichage des résultats
plt.plot(total_rewards)
plt.title("Évolution des récompenses totales")
plt.xlabel("Épisode")
plt.ylabel("Récompense totale")
plt.grid()
plt.show()



# Charger le modèle sauvegardé
policy = Policy()  # Créer une nouvelle instance de Policy
policy.load_state_dict(torch.load(save_path))  # Charger les poids
policy.eval()  # Passer en mode évaluation (désactive dropout, batchnorm, etc.)

# Exemple d'utilisation
state = env.reset()  # Réinitialiser l'environnement
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
action = policy(state_tensor)  # Prédire l'action pour l'état actuel
print(f"Action prédite : {action.item()}")
