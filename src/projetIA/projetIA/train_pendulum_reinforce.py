import torch
import torch.optim as optim
from collections import deque
import random
import matplotlib
import matplotlib.pyplot as plt 
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
    
    def remember(self, state, action, reward, discounted_sum, time_step):
        """
        Stores a transition in the replay memory.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.
        action : int
            The action taken at the current state.
        reward : float
            The reward received after taking the action.
        discounted_sum : float
            The discounted cumulative return for the episode.
        time_step : int
            The time step at which this transition occurred.
        """
        self.memory.append((state, action, reward, discounted_sum, time_step))


    def update(self, batch_size):
        """
        Met à jour le réseau de politique en utilisant un mini-batch d'expériences.

        Parameters:
        ----------
        batch_size : int
            Taille du mini-batch pour l'entraînement.
        """
        if len(self.memory) < batch_size:
            return

        # Prélever un mini-batch aléatoire
        minibatch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        returns = []

        # Préparer les données du mini-batch
        for state, action, reward, discounted_sum, time_step in minibatch:
            states.append(state)
            actions.append(action)
            returns.append(discounted_sum)

        # Convertir les données en tenseurs PyTorch
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Calculer les probabilités d'actions prédictes par le modèle
        action_probs = self.policy_network(states_tensor)
        action_log_probs = torch.log(action_probs)

        # Extraire les log-probabilités des actions prises

        # Calcul de la perte : -log(pi(a|s)) * G
        loss = -torch.mean(action_log_probs * returns_tensor)

        # Mise à jour des poids du réseau
        self.optimizer.zero_grad()
        loss.backward()
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
        state, _ = env.reset()
        episode_memory = []
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, _ , _= env.step(action)
            episode_memory.append((state, action, reward, log_prob))
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
        plot_reward(total_rewards=total_rewards)
        
        # Calculer les retours actualisés pour l'épisode
        discounted_sum = 0
        for i in reversed(range(len(episode_memory))):
            state, action, reward, log_prob = episode_memory[i]
            discounted_sum = reward + agent.discount_factor * discounted_sum
            agent.remember(state, action, reward, discounted_sum, i)

        # Mettre à jour le réseau avec un mini-batch
        if len(agent.memory) >= batch_size:
            agent.update(batch_size)

        print(f"Épisode {episode + 1}/{num_episodes}, Récompense : {episode_reward}")
        if (episode + 1) % 10 == 0:
            torch.save(policy.state_dict(), f'{episode+1}_' + save_path)

    # Sauvegarde finale du modèle
    torch.save(policy.state_dict(), 'final_' + save_path)
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")

    plot_reward(show_result=True)
    plt.savefig("plot_results\reinforce_training.png")
    plt.ioff()
    plt.show()
    
    return total_rewards