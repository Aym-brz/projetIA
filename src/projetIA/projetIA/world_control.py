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

    def send_control_request(self, pause: bool, reset: bool):
        request = ControlWorld.Request()
        request.world_control.pause = pause
        request.world_control.reset.all = reset

        self.get_logger().info(
            f'Sending control request: pause={pause}, reset={reset}'
        )
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Service response: {future.result()}')
        else:
            self.get_logger().error('Service call failed.')

def main(args=None):
    rclpy.init(args=args)
    node = GazeboControlClient()
    while True:
        # Example sequence: Start, Pause, Reset
        node.send_control_request(pause=False, reset=False)
        time.sleep(5)
        node.send_control_request(pause=True, reset=False)
        time.sleep(5)
        node.send_control_request(pause=True, reset=True)


if __name__ == '__main__':
    main()
