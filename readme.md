Documentation pour installer gazebo et ros : utiliser ros2 jazzy et gazebo harmonic  


Installation de ROS jazzy 
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html 

installation de gazebo harmonic : 
https://gazebosim.org/docs/harmonic/install_windows/ 

 
Création d’un fichier pour le robot (géométrie du robot, articulations, aspect):
double_pendulum_rail.sdf

Création d'un fichier pour le monde par défaut (taille, luminosité,...):
default_worlf.sdf
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
project_root/
├── config/
│   └── bridge_config.yaml      # Configuration for the ros_gz bridge
├── models/
│   └── double_pendulum_rail.sdf  # SDF model for the pendulum system
├── launch/
│   └── pendulum.launch.py      # Launch file for the simulation
├── README.md                   # Documentation
├── CMAkesLists.txt             # Setup script for the ROS 2 package
└── package.xml                 # ROS 2 package metadata
```

## Usage

1. Launch the Gazebo simulation with ROS 2 bridge
   ```bash
   ros2 launch projetIA pendulum.launch.py
   ```

## Training Methodology
The pendulum starts in a random initial position. The reinforcement learning algorithm encourages the pendulum to reach and maintain an inverted balance through reward-based feedback. No supervised learning is used; instead, the reward function incentivizes minimizing angular deviations and controlling velocities.

### Reward Function

The reward is calculated as:
- **Positive Terms**:
  - Maintaining angles near the upright position for both pendulum links.
  - Minimizing velocities (both angular and linear).
- **No Penalty for Failures**: The pendulum resets in random positions after each training episode.

