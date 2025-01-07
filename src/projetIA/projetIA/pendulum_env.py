import gym
import numpy as np
import rclpy
from rclpy.node import Node
from world_control import GazeboControlClient
from speed_publisher import SpeedPublisher
from state_subscriber import StateSubscriber
import time

rclpy.init()
max_speed = 15.0

class PendulumEnv(gym.Env, Node):
    def __init__(self, double_pendulum: bool = True):
        super().__init__('pendulum_env')
        self.action_space = gym.spaces.Box(low=-max_speed, high=max_speed, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6 if double_pendulum else 4,))

        self.double_pendulum = double_pendulum
        self.gazebo_control_client = GazeboControlClient()
        self.speed_publisher_node = SpeedPublisher()
        self.joint_state_sub = StateSubscriber(double_pendulum=double_pendulum)     
        self.done = False
        self.state = np.zeros(6 if double_pendulum else 4)
    
    def compute_reward(self, previous_state):
        state = self.state
        if self.double_pendulum:       
            instability = 1/100_000*sum([
                            # 2*180**2,                    # reward for staying alive
                           -(abs(state[0]-180)%360)**2, # upper joint up
                            #(abs(state[1])%360),
                            180**2-min(abs(state[2]%360), abs((state[2]-360))%360)**2,# lower joint straight
                            #(abs(state[3])%360),
                            (5/10*360)**2-(state[4]*360/10)**2,                 # center of the rail
                        ])
        else:
            instability = np.sqrt(sum([  
                            ((abs(state[0]-180)%360)/180)**2,     # angle deviation
                            #+ 100 if abs(state[0]-180)%360 < 3 else 0,
                            (abs(state[1])/1000)**2,            # angular velocity
                            (abs(state[2])/5)**2       # position on the rail
                        ])
            )
        stability = np.exp(-instability**2/(2*0.3**2))
        force_punishment = (abs(state[3]-previous_state[3])/(2*max_speed))**2

        reward = stability - force_punishment
        return reward
        
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
        self.speed_publisher_node.set_speed(action)
        # Wait for new state 
        self.gazebo_control_client.make_simulation_steps(num_steps=1)
        previous_state = np.copy(self.state)
        self.state = self.joint_state_sub.get_state()
        # reward for upright position, close to the center
        reward = self.compute_reward(previous_state=previous_state)        

        # self.done = abs(state[-2]) >= 5  # done if trolley reaches limits
        if abs(self.state[-2]) >= 5 :
            # reward -= 150
            self.done = True
        return self.state, reward, self.done, self.done, {}
        
    def reset(self):
        """
        Reset the simulation to its initial state.
        Returns:
            numpy.ndarray: The initial state of the environment after the reset.
        """
        # Reset the simulation
        self.speed_publisher_node.set_speed(0)
        self.gazebo_control_client.send_control_request(pause=False, reset=True)
        state = self.joint_state_sub.get_state()
        # Reset the state
        self.done = False
        return state, {}
    
    def render(self):
        return None

def main():
    env = PendulumEnv(double_pendulum=False)
    for i in range(3):
        state = env.reset()
        time.sleep(5)
        for i in range(100):
            env.step(-5)


if __name__=="__main__":
    main()

