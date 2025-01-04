import gym
import numpy as np
import rclpy
from rclpy.service import Service
from rclpy.node import Node
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import math
from world_control import GazeboControlClient
from speed_publisher import SpeedPublisher
from state_subscriber import StateSubscriber
import time

max_speed = 20.0

class PendulumEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('pendulum_env')
        # 
        self.action_space = gym.spaces.Box(low=-max_speed, high=max_speed, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.gazebo_control_client = GazeboControlClient()
        self.speed_publisher_node = SpeedPublisher()
        self.joint_state_sub = StateSubscriber()     
        
        self.done = False
        
    def step(self, action, num_sim_steps: int=10):
        """
        Execute one step in the environment with the given action.
        Args:
            action (numpy.ndarray): The action to be taken, which is a numpy array with a single float value.
        Returns:
            tuple: A tuple containing:
                - state (numpy.ndarray): The new state of the environment after taking the action.
                - reward (float): The reward received after taking the action.
                - done (bool): A boolean indicating whether the episode has ended.
                - info (dict): An empty dictionary, provided for compatibility with OpenAI Gym's API.
        """
        # Set the speed of the trolley
        self.speed_publisher_node.set_speed(action)
        # Wait for new state 
        self.gazebo_control_client.make_simulation_steps(num_sim_steps)
        state = self.joint_state_sub.get_state()
        # reward for upright position, close to the center
        objective_state = np.array([180, 0, 0, 0, 0, 0])
        
        reward = -sum([ (abs(state[0]-180)%360)**2, # upper joint up
                        min(abs(state[2]%360), abs((state[2]-360))%360)**2,# lower joint straight
                        (state[4]*360/10)**2                 # center of the rail
                    ])
        
        self.done = abs(state[4]) >= 4.89  # done if trolley reaches limits
        if self.done:
            reward -= 10000000
        return state, reward, self.done, {}
        
    def reset(self):
        """
        Reset the simulation to its initial state.
        Returns:
            numpy.ndarray: The initial state of the environment after the reset.
        """
        # Reset the simulation
        self.gazebo_control_client.send_control_request(pause=False, reset=True)
        # Reset the state
        self.done = False
        state = self.joint_state_sub.get_state()
        return state

def main():
    rclpy.init()
    env = PendulumEnv()
    env.gazebo_control_client.send_control_request(pause=False, reset=True)
    for i in range(3):
        env.step(-5, 100)
        print(env.joint_state_sub.get_state())
        time.sleep(1)


if __name__=="__main__":
    main()

