import torch
from pendulum_env import PendulumEnv
import pandas as pd
import numpy as np
import os
import rclpy
from network import FeedForwardNetwork
from network import DQN_NN
from eval_policy import evaluate_policy

def evaluate_all_policies(policies_dir, double_pendulum=False, num_episodes=2, max_iter=800):
    """Evaluate all policies in directory and return results"""
    results = []
    
    # Get all .pth files in directory
    policy_files = sorted([f for f in os.listdir(policies_dir) if (f.endswith('.pth') and f.startswith('policy'))])
    
    # Initialize environment
    env = PendulumEnv(double_pendulum=double_pendulum, starting_up=False, DQN=True)
    
    for policy_file in policy_files:
        # Extract episode number from filename
        episode_num = int(policy_file.split('_')[-1].split('.')[0])
        
        # Load policy
        policy = DQN_NN(env.observation_space.shape[0], env.action_space.n)
        policy.load_state_dict(torch.load(os.path.join(policies_dir, policy_file)))
        
        # Evaluate policy
        rewards = evaluate_policy(policy, env, num_episodes=num_episodes, 
                                max_iter=max_iter, num_sim_step=5, plot=False)
        
        # Store results
        results.append({
            'episode': episode_num,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'policy_file': policy_file
        })
        
        print(f"Evaluated {policy_file}: Mean reward = {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    return pd.DataFrame(results)

def delete_policies_below_threshold(df, policies_dir, reward_threshold):
    """Delete policy files with mean reward below the threshold and their corresponding target networks"""
    for _, row in df.iterrows():
        if row['mean_reward'] < reward_threshold:
            policy_file_path = os.path.join(policies_dir, row['policy_file'])
            target_file_path = os.path.join(policies_dir, 'target_' + row['policy_file'][7:])
            
            if os.path.exists(policy_file_path):
                os.remove(policy_file_path)
                print(f"Deleted {policy_file_path}")
            
            if os.path.exists(target_file_path):
                os.remove(target_file_path)
                print(f"Deleted {target_file_path}")
    # Example usage:
    # results_df = evaluate_all_policies(policies_dir)
    # delete_policies_below_threshold(results_df, policies_dir, reward_threshold=50)

def main():
    rclpy.init()
    # Directory containing policies
    policies_dir = "saved_policies/DQN/starting_down"
    
    # # Evaluate all policies
    # results = evaluate_all_policies(policies_dir)
    
    # # Save results
    # results.to_csv(f'{policies_dir}/policy_evaluation_results.csv', index=False)

    results = pd.read_csv(f'{policies_dir}/policy_evaluation_results.csv')
    delete_policies_below_threshold(results, policies_dir=policies_dir, reward_threshold=750)
if __name__ == "__main__":
    main()