import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

class StateSubscriber(Node):
    """
    A ROS2 node that subscribes to the '/joint_states' topic and processes joint state messages.
    Attributes:
        joint_state_sub (Subscription): The subscription to the '/joint_states' topic.
        state (numpy.ndarray): A 1D array of size 6 that holds the state of the joints and trolley.
        done (bool): A flag indicating whether the processing is done.
    Methods:
        __init__(): Initializes the StateSubscriber node and sets up the subscription.
        joint_state_callback(msg): Callback function that processes incoming joint state messages.
        reset(): Resets the state to an array of zeros and returns the state.
        __print__(): Prints the current state.
    """
    
    def __init__(self):
        super().__init__('state_subscriber')
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.state = np.zeros(6)
        self.done = False
        
    def joint_state_callback(self, msg):
        self.state[:] = [msg.position[0]%(2*np.pi)*180/np.pi, msg.velocity[0]*180/np.pi,  # upper joints position and velocity [° and °/s]
                           msg.position[1]%(2*np.pi)*180/np.pi, msg.velocity[1]*180/np.pi,  # lower joint position and velocity [° and °/s]
                           msg.position[2], msg.velocity[2]]  # trolley position and velocity
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