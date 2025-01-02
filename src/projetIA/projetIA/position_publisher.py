#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controls the angles of the upper and lower joints of the double pendulum.
Not used, as the position is set thanks to a PID, and the position setpoint cannot be removed. 
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time
import random
import math

class PositionPublisher(Node):
    """
    A ROS2 node that publishes joint positions for the double pendulum.
    
    Attributes:
        upper_publisher_ (Publisher): Publisher for upper joint position
        lower_publisher_ (Publisher): Publisher for lower joint position
    """
    def __init__(self):
        super().__init__('position_publisher')
        self.upper_publisher_ = self.create_publisher(
            Float64, '/upper_joint_pos_cmd', 10)
        self.lower_publisher_ = self.create_publisher(
            Float64, '/lower_joint_pos_cmd', 10)
        self.__upper_pos = 0.0
        self.__lower_pos = 0.0

    def set_upper_position(self, position):
        """Set the position of the upper joint.
        
        Args:
            position (float): Position in radians
        """
        self.__upper_pos = position
        msg = Float64()
        msg.data = self.__upper_pos
        self.upper_publisher_.publish(msg)

    def set_lower_position(self, position):
        """Set the position of the lower joint.
        
        Args:
            position (float): Position in radians
        """
        self.__lower_pos = position
        msg = Float64()
        msg.data = self.__lower_pos
        self.lower_publisher_.publish(msg)

    def set_positions(self, upper_position, lower_position):
        """Set the positions of both the upper and lower joints.
        
        Args:
            upper_position (float): Position of the upper joint in radians
            lower_position (float): Position of the lower joint in radians
        """
        self.set_upper_position(upper_position)
        self.set_lower_position(lower_position)

    def set_random_positions(self):
        """Set random positions for both the upper and lower joints between -π and π."""
        upper_position = random.uniform(-math.pi, math.pi)
        lower_position = random.uniform(-math.pi, math.pi)
        self.set_positions(upper_position, lower_position)

def main():
    rclpy.init()
    position_publisher = PositionPublisher()
    while True:
        # Example position sequence
        position_publisher.set_random_positions()
        time.sleep(5)
        position_publisher.set_positions(0.0, 0.0)
        time.sleep(5)

if __name__== '__main__':
    main()