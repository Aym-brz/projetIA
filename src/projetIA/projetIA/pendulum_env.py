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

class PendulumEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('pendulum_env')
        # 
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        
        self.speed_publisher_node = SpeedPublisher()
        self.joint_state_sub = StateSubscriber()
        self.state = np.zeros(6)
        self.done = False
        
        self.gazebo_control_client = GazeboControlClient()
        
    def step(self, action):
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
        self.speed_publisher_node.set_speed(action[0])
        
        # Wait for new state 
        rclpy.spin_once(self)
        
        state = self.joint_state_sub.get_state()
        
        # TODO Define the reward properly
        # reward for upright position, close to the center
        ojective_state = np.array([90, 0, 0, 0, 0, 0])
        reward = -sum(abs(ojective_state[[0,2,4]] - state[[0,2,4]]))
        done = abs(state[4]) >= 2.0  # done if trolley reaches limits
        
        return state, reward, done, {}
        
    def reset(self):
        """
        Reset the simulation to its initial state.
        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        # Reset the simulation
        self.gazebo_control_client.send_control_request(pause=False, reset=True)
        # Reset the state
        self.state = np.zeros(6)
        self.done = False
        
        return self.state

def main():
    env = PendulumEnv()


if __name__=="__main__":
    main()

