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


https://gazebosim.org/api/sim/8/jointcontrollers.htmls
création du fichier speed_publisher.py, mise à jour de bridge_config.yaml et pendulum.launch.py et setup.py pour définir le fichier speed_publisher comme porte entrée. 
=> le double pendule bouge à la vitesse définie initialement dans speed_publisher

crétion du fichier state_subscriber.py, mise à jour de setup.py pour définir le fichier speed_publisher comme porte entrée. 
=> reste à extraire les bonnes données



Fix the bridge for the world control service 
   
   can be started through ROS with 

   ```bash
   ros2 run ros_gz_bridge parameter_bridge /world/default/control@ros_gz_interfaces/srv/ControlWorld
   ```
   
   or equivalently 
   ```bash
   ros2 run ros_gz_bridge parameter_bridge /world/default/control@ros_gz_interfaces/srv/ControlWorld@gz.msgs.WorldControl@gz.msgs.Boolean
   ```
   with the shape : 
   ```bash
   parameter_bridge <service@ROS2_srv_type[@Ign_req_type@Ign_rep_type]> 
   ```

fait par github copilot et pas encore testé : pendulum_env et train_pendulum

# Double Pendulum on Rail Simulation

This project aims to simulate and train a double pendulum on a rail to balance itself in an inverted position using reinforcement learning. The simulation is implemented in Gazebo and Pytorch, with ROS 2 serving as the middleware interface.

## Features

- Simulates a double pendulum on a rail in Gazebo.
- Uses ROS 2 for interfacing sensor data and controlling the pendulum.
- Incorporates Pytorch to define and implement reinforcement learning logic.

## Prerequisites

Requires Python 3.12

Before running the project, ensure you have python 3 installed, as well as:
1. **ROS 2** (version jazzy)  
2. **Gazebo** (version harmonic)  
3. **ros-gazebo bridge** 

The following instructions work for Ubuntu 24.04 LTS (Noble)

### Install ROS 2 Jazzy 
[Installation instructions](https://docs.ros.org/en/jazzy/Installation/)

Make sure you have a locale which supports UTF-8. If you are in a minimal environment (such as a docker container), the locale may be something minimal like POSIX.
```bash
locale  # check for UTF-8
```
You will need to add the ROS 2 apt repository to your system.
First ensure that the Ubuntu Universe repository is enabled.
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```
Now add the ROS 2 GPG key with apt.
```bash
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```
Then add the repository to your sources list.
```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
Update your apt repository caches after setting up the repositories.
```bash
sudo apt update
sudo apt upgrade
```
Desktop Install (Recommended): ROS, RViz, demos, tutorials.
```bash
sudo apt install ros-jazzy-desktop
```

Add ro2 to the path: open this file with a text editor
```bash
nano ~/.bashrc
```
and add this line at the end : 
```bash
source /opt/ros/jazzy/setup.bash
```
Save and close the file, and source it:
```bash
source ~/.bashrc
```

### Install Gazebo Harmonic 
[Installation instructions](https://gazebosim.org/docs/harmonic/install_ubuntu/) 

```bash
sudo apt-get install curl lsb-release gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```
All Gazebo should be ready to use and the gz sim app ready to be executed.

### Install the bridge between Gazebo and ROS 
[Installation instructions](https://gazebosim.org/docs/all/ros_installation/)

The following command will install the correct version of Gazebo and ros_gz for your ROS installation on a Linux system. 
```bash
sudo apt-get install ros-jazzy-ros-gz
```

sudo apt install python3-pip
pip install rospy catkin-tools



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
│   │   └── pendulum_env.py         # Training environment
│   │   └── train_pendulum.py       # Training script
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
2. From another terminal, the speed and position of the joints can be retreived :
   - Thanks to a Node subscribed to the right topics
   ```bash
   ros2 run projetIA state_subscriber
   ```
   - Manually 
      - position and velocity of joints can be found on the ROS topic /joint_states:
      ```bash
      ros2 topic echo /joint_states
      ```
      - Position and rotation of parts can be found on the ROS topic /tf:
      ```bash
      ros2 topic echo /tf
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
  - Maintaining position near the center for the trolley.
  - Minimizing velocities (both angular and linear).
- **No Penalty for Failures**: The pendulum resets in random positions after each training episode.

## TODO
- Finish the environment (pendulum_env.py)
   - Create the reward function

- Write the script for the training
   - Implement the RL algoritm
   - Train the model