"""
This module defines a ROS2 node for controlling the speed of the trolley in the Gazebo simulation
by publishing on the /trolley_speed_cmd topic.
The speed can be dynamically changed by calling the set_speed method. 

Classes:
    SpeedPublisher: A ROS2 node that publishes the speed of the trolley.
Functions:
    main(args=None): Initializes the ROS2 node and demonstrates a sequence of control requests.
Usage:
    Run this module as a script to start the SpeedPublisher node and set a sequence of speedss.
    Import this module to use the node in another script.
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
        self.__speed = float(speed)
        msg = Float64()
        msg.data = self.__speed
        self.publisher_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    speed_publisher = SpeedPublisher()
    while True:
        # Example speed sequence
        speed_publisher.set_speed(10.0)
        time.sleep(1)
        speed_publisher.set_speed(-10.0)
        time.sleep(1)

if __name__ == '__main__':
    main()