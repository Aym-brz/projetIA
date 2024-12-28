Installation des elements nécessaires
Documentation pour installer gazebo et ros : utiliser ros2 jazzy et gazebo harmonic en suivant ceci : https://gazebosim.org/docs/all/ros_installation/

Installation de ROS jazzy 
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html 

installation de gazebo harmonic : 
https://gazebosim.org/docs/harmonic/install_windows/ 

Création d'un workspace (https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html) et d'un package (https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html) ros 

à partir des fichiers de démo: https://github.com/gazebosim/ros_gz/tree/ros2/ros_gz_sim_demos

création du fichier pour décrire le pendule et son environment
création du fichier bridge_config.yaml, pour faire le lien entre gazebo et ROS
création du fichier pendulum.launch.py, qui lance la simulation avec ROS et gazebo reliés.
=> le double pendule est simulé et publie les positions et vitesse sur les topics ROS

création du fichier speed_publisher.py, mise à jour de bridge_config.yaml et pendulum.launch.py et setup.py pour définir le fichier speed_publisher comme porte entrée. 
=> le double pendule bouge à la vitesse définie initialement dans speed_publisher

crétion du fichier state_subscriber.py, mise à jour de setup.py pour définir le fichier speed_publisher comme porte entrée. 
=> reste à extraire les bonnes données

fait par github copilot et pas encore testé : pendulum_env et train_pendulum

# Double Pendulum on Rail Simulation

This project aims to simulate and train a double pendulum on a rail to balance itself in an inverted position using reinforcement learning. The simulation is implemented in Gazebo and Pytorch, with ROS 2 serving as the middleware interface.

## Features

- Simulates a double pendulum on a rail in Gazebo.
- Uses ROS 2 for interfacing sensor data and controlling the pendulum.
- Incorporates Pytorch to define and implement reinforcement learning logic.

## Prerequisites

Before running the project, ensure you have the following installed, following these instructions: 
    [Installation instructions](https://gazebosim.org/docs/all/ros_installation/)

1. **ROS 2** (version jazzy)  
   [Installation instructions](https://docs.ros.org/en/jazzy/Installation/)  
2. **Gazebo** (version harmonic)  
   [Installation instructions](https://gazebosim.org/docs/harmonic/)  

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Aym-brz/projetIA
   cd projetIA
   ```

2. Build the workspace:
   ```bash
   colcon build
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

## Project Structure

```plaintext
projectroot
├── src/
│   ├── config/                     # Contains configuration files
│   │   └── bridge_config.yaml      
│   ├── models/                     # Contains SDF models
│   │   └── double_pendulum_rail.sdf  # Description of the robot (geometry, joints, aspect, environment)
│   │   └── default_world.sdf       # Default empty environment
│   ├── launch/                     # Contains launch files for the simulation
│   │   └── pendulum.launch.py      
│   ├── projetIA/                   # Python library for the project
│   │   └── speed_publisher.py      # ROS node to publish the speed of the trolley
│   │   └── state_subscriber.py     # ROS node to read the speeds and positions
│   ├── scripts/                    # Contains training scripts
│   │   └── train.py                # Training policy
│   ├── README.md                   # Documentation
│   ├── setup.py                    # Setup script for the ROS 2 package
│   └── package.xml                 # ROS 2 package metadata
└── README.md
```

## Usage

1. Launch the Gazebo simulation with ROS 2 bridge
   ```bash
   ros2 launch projetIA pendulum.launch.py
   ```
2. From another terminal, data can be retreived :
   - Thanks to a Node subscribed to the right topics
   ```bash
   ros2 run projetIA state_subscriber
   ```
   - Manually 
      - Angles and rotation speeds of joints can be found on the ROS topic /tf:
      ```bash
      ros2 topic echo /tf
      ```
      - Position and velocity of parts can be found on the ROS topic /joint_states:
      ```bash
      ros2 topic echo /joint_states
      ```
3. the velocity of the trolley can be set by publishing a float to the topic /trolley_speed_cmd:
   ```bash
   ros2 topic pub /trolley_speed_cmd std_msgs/msg/Float64 "data: 4.0"
   ```

   A publisher Node for the speed can be created by :
   ```bash
   ros2 run projetIA speed_publisher
   ```


## Training Methodology
The pendulum starts in a random initial position. The reinforcement learning algorithm encourages the pendulum to reach and maintain an inverted balance through reward-based feedback. No supervised learning is used; instead, the reward function incentivizes minimizing angular deviations and controlling velocities.

## Reward Function

The reward is calculated as:
- **Positive Terms**:
  - Maintaining angles near the upright position for both pendulum links.
  - Minimizing velocities (both angular and linear).
- **No Penalty for Failures**: The pendulum resets in random positions after each training episode.

## TODO
- Fix the state_subscriber so that the data extracted is correct
- Add possibility to change the speed of the trolley
- Receive the angle and speed of the joints in pytorch
- Send trolley speed to the model from pytorch
- Create the policy (reward function)
- Train the model
