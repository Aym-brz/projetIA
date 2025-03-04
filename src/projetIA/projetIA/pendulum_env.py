
"""
This module defines a custom OpenAI Gymnasium environment for controlling the pendulum in a Gazebo simulation.
The environment is compatible with both single and double pendulum setups and can be used with 
Deep Q-Network (DQN) or continuous action spaces.
Classes:
    PendulumEnv: A custom Gym environment for controlling a pendulum in a Gazebo simulation.
Functions:
    main(): Initializes the ROS2 node and demonstrates a sequence of control requests.
Usage:
    Run this module as a script to start the PendulumEnv environment and execute a sequence of control requests.
    Import this module to use the environment in another script.
"""
import gym
import numpy as np
import rclpy
from rclpy.node import Node
from world_control import GazeboControlClient
from speed_publisher import SpeedPublisher
from state_subscriber import StateSubscriber
import time

max_speed = 20.0

class PendulumEnv(gym.Env, Node):
    def __init__(self, double_pendulum: bool = True, starting_up: bool = False, DQN: bool = False):
        super().__init__('pendulum_env')
        # 
        self.double_pendulum = double_pendulum
        self.DQN = DQN
        if DQN:
            self.discretizition = 25
            self.action_space = gym.spaces.Discrete(self.discretizition)
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8 if double_pendulum else 5,))
        self.gazebo_control_client = GazeboControlClient()
        self.speed_publisher_node = SpeedPublisher()
        self.joint_state_sub = StateSubscriber(double_pendulum=double_pendulum, starting_up=starting_up)     
        self.done = False
        self.state = self.joint_state_sub.get_state()

    def map_action_to_speed(self, action):
        return np.linspace(-1, 1, self.discretizition)[action]
   
    def compute_reward(self):
        state = self.state
        if self.double_pendulum:
            # state contains [cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot, x, x_dot]
            reward = 1/4 * (1 - state[0]) + 1/4 *(1 + state[3]) - (abs(state[-2])/5)**2
        else:
            # state contains [cos(theta), sin(theta), theta_dot, x, x_dot]
            reward = 1/2 * (1 - state[0]) - (abs(state[-2])/5)**2
            
        return reward
        
    def step(self, action, num_sim_steps: int=5):
        """
        Execute one step in the environment with the given action.
        Args:
            action (numpy.ndarray): The action to be taken, which is a numpy array with a single float value between -1 and 1 if not DQN.
        Returns:
            tuple: A tuple containing:
                - state (numpy.ndarray): The new state of the environment after taking the action.
                - reward (float): The reward received after taking the action.
                - done (bool): A boolean indicating whether the episode has ended.
                - done (bool): Provided for compatibility with OpenAI Gym's API
                - info (dict): An empty dictionary, provided for compatibility with OpenAI Gym's API.
        """
        # Set the speed of the trolley
        if self.DQN:
            action = self.map_action_to_speed(action)
        self.speed_publisher_node.set_speed(action*max_speed)
        # Wait for new state 
        self.gazebo_control_client.make_simulation_steps(num_sim_steps)
        self.state = self.joint_state_sub.get_state()
        # reward for upright position, close to the center
        reward = self.compute_reward()       

        if abs(self.state[-2]) >= 5 : # done if trolley reaches limits
            reward -= 400
            self.done = True
            print("Trolley reached limits")
        
        if abs(self.state[2]) >= 20: # done if angular velocity is too high
            print("Angular velocity too high")
            self.done = True
        
        if self.double_pendulum and abs(self.state[5]) >= 30:
            print("Angular velocity too high")
            self.done = True
        return self.state, reward, self.done, self.done, {}
        
    def reset(self):
        """
        Reset the simulation to its initial state.
        Returns:
            - numpy.ndarray: The initial state of the environment after the reset.
            - info (dict): An empty dictionary, provided for compatibility with OpenAI Gym's API.
        """
        # Reset the simulation
        self.speed_publisher_node.set_speed(0)
        self.gazebo_control_client.send_control_request(pause=False, reset=True)
        state = self.joint_state_sub.get_state()
        self.done = False
        return state, {}

def main():
    rclpy.init()
    double_pendulum = False
    starting_up = False
    DQN = False

    env = PendulumEnv(double_pendulum=double_pendulum, starting_up=starting_up, DQN=DQN)
    for i in range(3):
        state = env.reset()
        time.sleep(5)
        for i in range(10):
            env.step(1, 5)
            time.sleep(1)


if __name__=="__main__":
    main()

