#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
To change the speed during the simulation, call the set_speed service:
ros2 service call /set_speed std_srvs/srv/SetFloat "{data: 1.5}"
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time

class SpeedPublisher(Node):
    """
    A ROS2 node that publishes the speed of the trolley on the /trolley_speed_cmd topic.
    Attributes:
        publisher_ (Publisher): The publisher to the /trolley_speed_cmd topic.
        speed (float): The speed of the trolley.
    Methods:
        __init__(): Initializes the SpeedPublisher node and sets up the publisher.
        set_speed(speed): Sets the speed of the trolley and publishes it on the /trolley_speed_cmd topic.
    """
    def __init__(self):
        super().__init__('speed_publisher')
        self.publisher_ = self.create_publisher(Float64, '/trolley_speed_cmd', 10)
        self.__speed = 0.0  # Initial speed value

    def set_speed(self, speed):
        """
        Set the speed of the trolley and publish it on the /trolley_speed_cmd topic.

        Args:
            speed (float): The speed value to set.
        """
        self.__speed = speed
        msg = Float64()
        msg.data = self.__speed
        self.publisher_.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    speed_publisher = SpeedPublisher()
    while True:
        # Example speed sequence
        speed_publisher.set_speed(1.0)
        time.sleep(5)
        speed_publisher.set_speed(-1.0)
        time.sleep(5)

if __name__ == '__main__':
    main()