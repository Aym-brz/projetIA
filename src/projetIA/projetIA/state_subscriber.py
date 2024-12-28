import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

class StateSubscriber(Node):
    def __init__(self):
        super().__init__('state_subscriber')

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.state = np.zeros(6)
        self.done = False
        
    def joint_state_callback(self, msg):
        self.state[4:6] = msg.position[0], msg.velocity[0]  # trolley position and velocity
        print(self.state)  # Print the state vector
        
    def tf_callback(self, msg):
        # Extract angles from transformation matrix
        self.state[0:4] = [msg.transforms[0].transform.rotation.z,  # pendulum1 angle
                          msg.transforms[1].transform.rotation.z,  # pendulum2 angle
                          msg.transforms[0].transform.rotation.w,  # pendulum1 angular velocity
                          msg.transforms[1].transform.rotation.w]  # pendulum2 angular velocity
        print(self.state)  # Print the state vector

        
    def reset(self):
        # Implement reset logic (e.g., through a service call to Gazebo)
        self.state = np.zeros(6)
        return self.state
    
    def __print__(self):
        print(self.state)

def main():
    rclpy.init()
    state_subscriber = StateSubscriber()
    rclpy.spin(state_subscriber)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()