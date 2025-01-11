"""
This module defines a ROS2 node for subscribing to the '/joint_states' topic and processing joint state messages.
The state of the joints and trolley can be accessed and reset.
Classes:
    StateSubscriber: A ROS2 node that subscribes to the '/joint_states' topic and processes joint state messages.
Functions:
    main(): Initializes the ROS2 node and prints the state of the joints and trolley at regular intervals.
Usage:
    Run this module as a script to start the StateSubscriber node and print the state of the joints and trolley.
    Import this module to use the node in another script.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class StateSubscriber(Node):
    """
    A ROS2 node that subscribes to the '/joint_states' topic and processes joint state messages.
    Attributes:
        joint_state_sub (Subscription): The subscription to the '/joint_states' topic.
        state (numpy.ndarray): A 1D array that holds the state of the joints and trolley.
            for double pendulum, state contains [cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot, x, x_dot]
            for single pendulum, state contains [cos(theta), sin(theta), theta_dot, x, x_dot]
    Methods:
        __init__(): Initializes the StateSubscriber node and sets up the subscription.
        joint_state_callback(msg): Callback function that processes incoming joint state messages.
        reset(): Resets the state to an array of zeros and returns the state.
    """
    
    def __init__(self, double_pendulum: bool = True, starting_up = False):
        super().__init__('state_subscriber')
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.double_pendulum = double_pendulum
        if self.double_pendulum:
            self.state = np.zeros(8)
        else:
            self.state = np.zeros(5)
        self.starting_up = starting_up

        
    def joint_state_callback(self, msg):
        if self.double_pendulum : # double pendulum case
            self.state[:] = [np.cos(msg.position[0]), np.sin(msg.position[0]), msg.velocity[0], 
                             np.cos(msg.position[1]), np.sin(msg.position[1]), msg.velocity[1], 
                             msg.position[2], msg.velocity[2]]
        else:   # single pendulum state
            self.state[:] = [np.cos(msg.position[0]), np.sin(msg.position[0]), msg.velocity[0], 
                             msg.position[1], msg.velocity[1]]
        
    def get_state(self):
        """Read the state of the joints and return it. 

        Returns:
            state (numpy.ndarray): A 1D array that holds the state of the joints and trolley.
              for double pendulum, state contains [cos(theta1), sin(theta1), theta1_dot, cos(theta2), sin(theta2), theta2_dot, x, x_dot]
              for single pendulum, state contains [cos(theta), sin(theta), theta_dot, x, x_dot]
        """
        rclpy.spin_once(self)
        return self.state   

def main():
    rclpy.init()
    state_subscriber = StateSubscriber(double_pendulum=False)
    while True:
        # Print the state every second
        time.sleep(1)
        print(state_subscriber.get_state())
    
if __name__ == '__main__':
    main()