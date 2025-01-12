import torch
from pendulum_env import PendulumEnv
import matplotlib.pyplot as plt
import rclpy
from network import FeedForwardNetwork
from network import DQN_NN
from eval_policy import evaluate_policy
import os
import glob
import re
from train_pendulum_DQN import train as train_DQN
from train_pendulum_reinforce import train as train_reinforce

def get_best_model_path(load_path):
    files = glob.glob(os.path.join(load_path, "best_*.pth"))
    if not files:
        raise FileNotFoundError("No saved policies found.")
    best_file = max(files, key=lambda f: float(re.findall(r"[-+]?\d*\.\d+|\d+", f)[-1]))
    return best_file

def main():
    double_pendulum = True
    starting_up = False
    training = True
    evaluating = True
    DQN = True
    
    load_path = f"saved_policies/double_pendulum/DQN/starting_down" # model to load to resume training
    save_path = f"saved_policies/double_pendulum/DQN/starting_down" # path to save the models
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    num_episodes = 5000 # number of training episodes
    
    # Hyperparamètres
    hyperparameters = {
        'MAX_EPISODE_LENGTH': 800,  # maximum length of an episode
        'NUM_SIM_STEPS': 1 if double_pendulum else 5,         # number of simulation steps for each action
    }
    
    if DQN:
        hyperparameters.update({
            'GAMMA': 0.995,             # discount factor
            'LR': 0.0003,               # learning rate
            'MEM_SIZE': 20000,          # memory size
            'BATCH_SIZE': 512,          # size of batches (in steps)
            'EPSILON_START': 0.9,       # initial value of epsilon
            'EPSILON_END': 0.05,        # final value of epsilon
            'EPSILON_DECAY': 80000,     # decay rate of epsilon (in simulation steps)
            'TAU': 0.005,               # soft update rate of target network
        })
    else:
        hyperparameters.update({
            'GAMMA': 0.95,               # discount factor
            'LR': 0.001,                 # learning rate
            'MEM_SIZE': 10000,           # memory size
            'BATCH_SIZE': 15,            # size of batches (in episodes)
            'STDDEV_START': 1.0,         # standard deviation for sampling actions
            'STDDEV_END': 0.2,          # final standard deviation
        })


    # Initialisation de l'environnement
    rclpy.init()
    env = PendulumEnv(double_pendulum=double_pendulum, starting_up=starting_up, DQN=DQN)

    # Vérification des dimensions d'état et d'action
    state_dim = env.observation_space.shape[0]  # 7 pour le double pendule, 5 pour le simple
    if DQN:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]      # 1 pour la vitesse du chariot

    print(f"Dimensions de l'état : {state_dim}, Dimensions de l'action : {action_dim}")
    
    if training:
        if DQN:    
            policy_net = DQN_NN(state_dim, action_dim)
            target_net = DQN_NN(state_dim, action_dim)
            try:
                best_policy_path = get_best_model_path(load_path)
                policy_net.load_state_dict(torch.load(best_policy_path))
                print(f"Loaded saved best policies from {best_policy_path}.")
                target_net.load_state_dict(torch.load(best_policy_path.replace("policy", "target")))
            except FileNotFoundError:
                print("No saved policies found. Starting training from scratch.")
            total_rewards = train_DQN(policy_net, target_net, env, num_episodes=num_episodes, save_path=save_path, hyperparameters=hyperparameters) 
        
        else:
            # Initialisation de la politique
            policy = FeedForwardNetwork(double_pendulum=double_pendulum)
            try:
                policy.load_state_dict(torch.load(load_path))
            except:
                print('No policy found, training from scratch') 
            total_rewards = train_reinforce(policy, env, num_episodes=num_episodes, save_path=save_path, hyperparameters=hyperparameters) 
        # Entraînement de la politique

        # Affichage des résultats
        plt.plot(total_rewards)
        plt.title("Évolution des récompenses totales")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense totale")
        plt.grid()
        plt.show()
    if evaluating:
        # Charger le modèle sauvegardé
        if DQN:
            policy = DQN_NN(state_dim, action_dim)  # Créer une nouvelle instance de Policy
        else: 
            policy = FeedForwardNetwork(double_pendulum=double_pendulum)
        try:
            best_policy_path = get_best_model_path(save_path)
        except FileNotFoundError:
            best_policy_path = get_best_model_path(load_path)
        print(f'testing policy {best_policy_path}')
        policy.load_state_dict(torch.load(best_policy_path))
        # Évaluation de la politique entraînée
        evaluation_rewards = evaluate_policy(policy, env, num_episodes=10, max_iter=800, plot=True)

if __name__ == "__main__":
    main()