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

class PendulumEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('pendulum_env')
        # 
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        
        self.publisher = self.create_publisher(Float64, '/trolley_speed_cmd', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.state = np.zeros(6)
        self.done = False
        
    def joint_state_callback(self, msg):
        self.state[:] = [msg.position[0]%np.pi*180/np.pi, msg.velocity[0]*180/np.pi,  # upper joint position [-180° to +180°] and velocity [°/s]
                           msg.position[1]%np.pi*180/np.pi, msg.velocity[1]*180/np.pi,  # lower joint position [-180° to +180°] and velocity [°/s]
                           msg.position[2], msg.velocity[2]]  # trolley position [-2 to 2] and velocity |-inf to inf]
        
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
        
        msg = Float64()
        msg.data = float(action[0])
        self.publisher.publish(msg)
        
        # Wait for new state (implement proper synchronization)
        rclpy.spin_once(self)
        
        # TODO Define the reward properly
        reward = -abs(np.pi - self.state[0]) - abs(self.state[1])  # reward for upright position, close to the center and low velocity
        done = abs(self.state[4]) >= 2.0  # done if trolley reaches limits
        
        return self.state, reward, done, {}
        
    def reset(self):
        # TODO Implement reset logic

        # https://gazebosim.org/api/sim/8/reset_simulation.html
        self.state = np.zeros(6)
        return self.state

def main():
    env = PendulumEnv()


if __name__=="__main__":
    main()
    
