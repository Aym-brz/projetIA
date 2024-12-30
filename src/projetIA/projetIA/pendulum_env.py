import gym
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import math

class PendulumEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('pendulum_env')
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        
        self.publisher = self.create_publisher(Float64, '/trolley_speed_cmd', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.state = np.zeros(6)
        self.done = False
        
    def joint_state_callback(self, msg):
        self.state[:] = [msg.position[0]%(2*np.pi)*180/np.pi, msg.velocity[0]*180/np.pi,  # upper joints position and velocity [° and °/s]
                           msg.position[1]%(2*np.pi)*180/np.pi, msg.velocity[1]*180/np.pi,  # lower joint position and velocity [° and °/s]
                           msg.position[2], msg.velocity[2]]  # trolley position and velocity
        
    def step(self, action):
        msg = Float64()
        msg.data = float(action[0])
        self.publisher.publish(msg)
        
        # Wait for new state (implement proper synchronization)
        rclpy.spin_once(self)
        
        reward = -abs(math.pi - self.state[0]) - abs(math.pi - self.state[1])  # reward for upright position
        done = abs(self.state[4]) > 2.0  # done if trolley reaches limits
        
        return self.state, reward, done, {}
        
    def reset(self):
        # Implement reset logic (e.g., through a service call to Gazebo)
        self.state = np.zeros(6)
        return self.state