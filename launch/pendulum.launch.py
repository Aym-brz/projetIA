from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from ros_gz_bridge.actions import RosGzBridge
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

    """     # ROS-Gazebo bridge utilisant le fichier YAML
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[bridge_config_path]
    )
    
    return LaunchDescription([
        gazebo_server,
        ros_gz_bridge,
    ]) """
    # Bridge
    ros_gz_bridge = RosGzBridge(
        bridge_name='ros_gz_bridge',
        config_file=bridge_config_path)

    return LaunchDescription([
        gazebo_server,
        DeclareLaunchArgument('rqt', default_value='true',
                              description='Open RQt.'),
        ros_gz_bridge,
    ])