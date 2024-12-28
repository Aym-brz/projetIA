import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ForcePublisher(Node):
    def __init__(self):
        super().__init__('force_publisher')
        self.publisher_ = self.create_publisher(Float64, 'pendulum/force', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = Float64()
        msg.data = 0.0  # Replace with your force value
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    force_publisher = ForcePublisher()
    rclpy.spin(force_publisher)
    force_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()