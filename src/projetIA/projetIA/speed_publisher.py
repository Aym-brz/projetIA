#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
To change the speed during the simulation, call the set_speed service:
ros2 service call /set_speed std_srvs/srv/SetFloat "{data: 1.5}"
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_msgs.msg import Float64

class SpeedPublisher(Node):
    def __init__(self):
        super().__init__('speed_publisher')
        self.publisher_ = self.create_publisher(Float64, '/trolley_speed_cmd', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.speed = 1.0  # Initial speed value
        # self.srv = self.create_service(SetFloat, 'set_speed', self.set_speed_callback)  # Create the service

    # def set_speed_callback(self, request, response):
    #     """
    #     Callback function to handle speed change requests.

    #     This function is called when a request is made to the set_speed service.
    #     """
    #     self.speed = request.data
    #     response.success = True
    #     response.message = f"Speed set to {self.speed}"
    #     return response

    def timer_callback(self):
        """
        Publish the current speed value on the /trolley_speed_cmd topic.

        This function is called at a rate of 10 Hz by the timer.
        """
        msg = Float64()
        msg.data = self.speed
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    speed_publisher = SpeedPublisher()
    rclpy.spin(speed_publisher)
    # # Destroy the node explicitly
    # speed_publisher.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()