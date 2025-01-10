import torch
from pendulum_env import PendulumEnv
import matplotlib.pyplot as plt
import rclpy
from network import Policy
from eval_policy import evaluate_policy

# choose the training method : 
DQN = False
first_method = True

if DQN:
    from train_pendulum_DQN import train
elif first_method:
    from train_pendulum import train
else:
    from train_pendulum_reinforce import train

def main():
    double_pendulum = False
    starting_up = False
    # Hyperparamètres
    num_episodes = 300
    discount_factor = 0.95
    learning_rate = 1e-3
    max_iter = 800
    num_sim_step = 5
    stddev = 0.3
    load_path = "best_trained_single_pendulum_policy.pth" # model to load to resume training
    save_path= "trained_single_pendulum_policy.pth" # path to save the model
    batch_size = int(num_episodes/25)

    # Initialisation de l'environnement
    rclpy.init()
    env = PendulumEnv(double_pendulum=double_pendulum, starting_up=starting_up)

    # Vérification des dimensions d'état et d'action
    state_dim = env.observation_space.shape[0]  # 6 pour le double pendule, 4 pour le simple
    action_dim = env.action_space.shape[0]      # 1 pour la vitesse du chariot

    print(f"Dimensions de l'état : {state_dim}, Dimensions de l'action : {action_dim}")

    # Initialisation de la politique
    policy = Policy(double_pendulum=double_pendulum)
    try:
        policy.load_state_dict(torch.load(load_path))
    except:
        print('No policy found, training from scratch') 

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
    evaluation_rewards = evaluate_policy(policy, env, num_episodes=10, max_iter=max_iter, plot=True)


    
if __name__ == "__main__":
    main()
    #plt.ioff()
    plt.show()