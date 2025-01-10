import torch
from pendulum_env import PendulumEnv
import matplotlib.pyplot as plt
import rclpy
from network import FeedForwardNetwork
from network import DQN

def evaluate_policy(policy: FeedForwardNetwork|DQN, env: PendulumEnv, num_episodes: int = 10, max_iter: int = 2000, num_sim_step:int = 1, plot=True):
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
            next_state, reward, done, _ , _ = env.step(action.item(), num_sim_steps=num_sim_step)
            episode_rewards.append(reward)
            state = next_state
            iter += 1

        total_episode_reward = sum(episode_rewards)
        total_rewards.append(total_episode_reward)
        print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_episode_reward}")

    if plot:
        plt.plot(total_rewards)
        plt.title("Évolution des récompenses totales en évaluation")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense totale")
        plt.grid()
        plt.show()
    return total_rewards

def main():
    rclpy.init()
    double_pendulum = False
    starting_up = False
    max_iter = 3000
    is_DQN = False
    

    # Initialisation de l'environnement
    env = PendulumEnv(double_pendulum=double_pendulum, starting_up=starting_up, DQN=is_DQN)

    save_path="final_trained_single_pendulum_policy.pth"
    # Charger le modèle sauvegardé
    if is_DQN:
        policy = DQN(env.observation_space.shape[0], env.action_space.shape[0])
    else:
        policy = FeedForwardNetwork(double_pendulum=double_pendulum)  # Créer une nouvelle instance de Policy
    
    policy.load_state_dict(torch.load(save_path))  # Charger les poids
    # Évaluation de la politique entraînée
    evaluation_rewards = evaluate_policy(policy, env, num_episodes=10, max_iter=max_iter, plot=True)
    
if __name__ == "__main__":
    main()
    plt.show()