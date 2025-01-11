"""
Launch file for the double pendulum simulation using Gazebo and ROS 2s
This launch file sets up the necessary nodes and processes to run the double pendulum simulation in Gazebo,
including the Gazebo server, ROS-Gazebo bridge, and custom nodes for controlling the simulation and publishing speed.
Functions:
    generate_launch_description(): Generates and returns the launch description for the simulation.
Nodes:
    gazebo_controller_node: Node for controlling the Gazebo world.
    speed_publisher_node: Node for publishing speed information.
Processes:
    gazebo_server: Process to start the Gazebo server with the specified SDF model.
    ros_gz_bridge: Bridge between ROS 2 and Gazebo using the specified configuration file.
Usage:
    This launch file can be executed using the ROS 2 launch command.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from ros_gz_bridge.actions import RosGzBridge
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Récupération du chemin absolu de ce fichier
    package_dir = os.path.dirname(__file__)
    
    # Chemin relatif pour le modèle SDF
    model_path = os.path.join(package_dir, '../models/double_pendulum_rail.sdf')
    
    # Chemin relatif pour le fichier YAML de configuration
    bridge_config_path = os.path.join(package_dir, '../config/bridge_config.yaml')
    
    # Gazebo server process
    gazebo_server = ExecuteProcess(
        cmd=['gz', 'sim', model_path])
    
    # Bridge
    ros_gz_bridge = RosGzBridge(
        bridge_name='ros_gz_bridge',
        config_file=bridge_config_path)
    
    # Service bridge configuration is not available in the python API.
    # The service bridge can be started with the following command:
    if not os.fork():
        os.system('ros2 run ros_gz_bridge parameter_bridge /world/default/control@ros_gz_interfaces/srv/ControlWorld')
        return


    return LaunchDescription([
        gazebo_server,
        DeclareLaunchArgument('rqt', default_value='true',
                              description='Open RQt.'),
        ros_gz_bridge
    ])