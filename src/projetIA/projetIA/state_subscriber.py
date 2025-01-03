import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import time

class StateSubscriber(Node):
    """
    A ROS2 node that subscribes to the '/joint_states' topic and processes joint state messages.
    Attributes:
        joint_state_sub (Subscription): The subscription to the '/joint_states' topic.
        state (numpy.ndarray): A 1D array of size 6 that holds the state of the joints and trolley.
            [upper joint position, upper joint velocity, lower joint position, lower joint velocity, trolley position, trolley velocity]
            [°, °/s, °, °/s, m, m/s]
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
        self.__state = np.zeros(6)
        self.print_state = False
        
    def joint_state_callback(self, msg):
        self.__state[:] = [msg.position[0]%(2*np.pi)*180/np.pi, msg.velocity[0]*180/np.pi,  # upper joints position and velocity [° and °/s]
                           msg.position[1]%(2*np.pi)*180/np.pi, msg.velocity[1]*180/np.pi,  # lower joint position and velocity [° and °/s]
                           msg.position[2], msg.velocity[2]]  # trolley position and velocity
       
    def get_state(self):
        """Read the state of the joints and return it. 

        Returns:
            state (numpy.ndarray): A 1D array of size 6 that holds the state of the joints and trolley.
                [upper joint position, upper joint velocity, lower joint position, lower joint velocity, trolley position, trolley velocity]
                [°, °/s, °, °/s, m, m/s]
        """
        rclpy.spin_once(self)
        if self.print_state:
            print(self)
        return self.__state   
             
    def __print__(self):
        print(self.__state)
        


def main():
    rclpy.init()
    state_subscriber = StateSubscriber()
    state_subscriber.print_state = True
    while True:
        # Print the state every second
        time.sleep(1)
        print(state_subscriber.get_state())
    
if __name__ == '__main__':
    main()