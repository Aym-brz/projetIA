"""
This module defines a ROS2 node for controlling the Gazebo simulation world.
Classes:
    GazeboControlClient: A client node to interact with the Gazebo world control service.
Functions:
    main(args=None): Initializes the ROS2 node and demonstrates a sequence of control requests.
Usage:
    Run this module as a script to start the GazeboControlClient node and execute a sequence of control requests.
    Import this module to use the node in another script.
"""
import rclpy
from rclpy.node import Node
import time
from ros_gz_interfaces.srv import ControlWorld

class GazeboControlClient(Node):
    def __init__(self):
        super().__init__('world_control')
        self.client = self.create_client(ControlWorld, '/world/default/control')

        # Wait for the service to become available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the Gazebo control service...')

        self.get_logger().info('Gazebo control service is available.')

    def send_control_request(self, pause: bool=False, reset: bool=True):
        request = ControlWorld.Request()
        request.world_control.pause = pause
        request.world_control.reset.all = reset
        self.client.call(request, 0.5)
        
    def make_simulation_steps(self, num_steps: int=10):
        """Pause the simulation and execute multiple steps.

        Args:
            num_steps (int, optional): Number of steps to do. Defaults to 10.
        """
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = num_steps
        self.client.call(request, 0)
        

        

def main(args=None):
    rclpy.init(args=args)
    node = GazeboControlClient()
    # Example sequence: Start, Pause, Reset
    node.send_control_request(pause=False, reset=False)
    time.sleep(5)
    node.send_control_request(pause=True, reset=False)
    time.sleep(1)
    node.send_control_request(pause=True, reset=True)


if __name__ == '__main__':
    main()
