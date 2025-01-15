# Double Pendulum on Rail Simulation

This project aims to simulate and train a single/double pendulum on a rail to balance itself in an inverted position using reinforcement learning. The simulation is implemented in Gazebo and Pytorch, with ROS 2 serving as the middleware interface.

## Features

- Simulates a double pendulum on a rail in Gazebo.
- Uses ROS 2 for interfacing sensor data and controlling the pendulum.
- Reinforcement learning setup using Gymnasium-compatible environments.
- Dynamic control, data publishing, and state monitoring nodes.

## Prerequisites
This project has been developped under Ubuntu 24.04 LTS. We didn't manage to make Gazebo work in WSL.

Requires Python 3.12

Before running the project, ensure you the following installed.
1. **ROS 2** (version jazzy)  
2. **Gazebo** (version harmonic)  
3. **ros-gz bridge** 

The following instructions work for Ubuntu 24.04 LTS (Noble)

### Install ROS 2 Jazzy 
The following intructions are extracted from the official [installation instructions](https://docs.ros.org/en/jazzy/Installation/)

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
The following intructions are extracted from the official [installation instructions](https://gazebosim.org/docs/harmonic/install_ubuntu/) 

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


### Install the python packages

Create a new virtual environment and add all the required packages to it.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



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
├── src/projetIA
│   ├── config/                        # Contains configuration files
│   │   └── bridge_config.yaml            # Bridge configuration between Gazebo and ROS topics
│   ├── models/                        # Contains SDF models
│   │   └── default_world.sdf             # Empty environment
│   │   └── double_pendulum_rail.sdf      # Description of the double pendulum
│   │   └── simple_pendulum_rail.sdf      # Description of the single pendulum
│   │   └── simple_pendulum_up_rail.sdf   # Single pendulum initialized from up
│   ├── launch/                        # Contains launch files for simulations and the ros-gz bridge
│   │   └── pendulum.launch.py            # launch the double pendulum 
│   │   └── simple_pendulum_up.launch.py  # launch the simple pendulum upwards
│   │   └── simple_pendulum.launch.py     # launch the simple pendulum 
│   │   └── test_pendulum.launch.py       # launch the double pendulum and test the different nodes.
│   ├── projetIA/                      # Python library for the project
│   │   └── eval_policy.py                # Evaluate the a policy obtain
│   │   └── main.py                       # Launch the training and the evaluation
│   │   └── network.py                    # Structure of the neural networks
│   │   └── state_subscriber.py           # ROS node to read the speeds and positions
│   │   └── speed_publisher.py            # ROS node to publish the speed of the trolley
│   │   └── pendulum_env.py               # Creating Gymnasium environment, starts all the ROS nodes required
│   │   └── train_pendulum_reinforce.py   # Training script for DQN
│   │   └── train_pendulum_reinforce.py   # Training script for REINFORCE algorithm
│   │   └── train_pendulum.py             # Training script (not working, implementation of reinforce)
│   │   └── world_control.py              # ROS node to start, pause and reset the simulation
│   ├── setup.py                       # Setup script for the ROS 2 package
│   └── package.xml                    # ROS 2 package metadata
├── README.md                          # Documentation
└── requirements.txt                   # Python requirements

```

## Usage

1. Launch the Gazebo simulation with ROS 2 bridge:
   ```bash
   ros2 launch projetIA pendulum.launch.py
   ```
   This will launch the simulation, as well as the Gazebo - ROS bridge.

2. The simulation can be interacted with manually, to check that everything work correctly.
   - The speed and position of the joints can be retrieved on the ROS `/joint_states` topic:
     - Using a node subscribed to the right topics:
       ```bash
       ros2 run projetIA state_subscriber
       ```
     - Manually:
       ```bash
       ros2 topic echo /joint_states
       ```
   - The velocity of the trolley can be set by publishing a float to the topic `/trolley_speed_cmd`:
     - Using a node publishing to the right topics:
       ```bash
       ros2 run projetIA speed_publisher
       ```
     - Manually:
       ```bash
       ros2 topic pub /trolley_speed_cmd std_msgs/msg/Float64 "data: 4.0"
       ```
   - The simulation can be started, paused, and reset by publishing on the ROS topic `/world/default/control`:
     - Using a node:
       ```bash
       ros2 run projetIA world_control
       ```

3. Training or evaluating a policy can be launched by running the file `src/projetIA/projetIA/main.py` (set the different parameters in this file).

4. It is also possible to evaluate a policy by running the file `src/projetIA/projetIA/eval_policy.py`

https://github.com/user-attachments/assets/d1cfd3cd-00e4-4f16-bdd5-1c643b5e6a2d

Stability test, applying a force through the Gazebo UI (recording the screen significantly impacted the simulation performance, forcing us to record indirectly):

https://github.com/user-attachments/assets/d1b557f0-6838-49be-af98-7bc6c8f91b20

To replicate these videos, follow these steps:

1. Launch a Gazebo simulation with a simple pendulum starting downwards:
  ```bash
  ros2 launch projetIA simple_pendulum.launch.py
  ```

2. Run the `eval_policy` script with the following settings in the main function:
  ```python
  double_pendulum = False
  starting_up = False
  max_iter = 10000
  is_DQN = True
  save_path = "saved_policies/single_pendulum/DQN/starting_down/policy_DQN_2930.pth"
  ```

3. Apply a force to the pendulum using the "Apply Force Torque" section in Gazebo:
  - Click on the three dots in the top right corner of the Gazebo window to display this section.
  - Click on the pendulum and select "upper_link" from the link list.
  - Set the force magnitude in the Y direction.
  - Click the "Apply Force" button will apply the force for a short duration.


## Training Methodology
The pendulum starts on the stable low position. The reinforcement learning algorithm encourages the pendulum to reach and maintain an inverted balance through reward-based feedback. No supervised learning is used; instead, the reward function incentivizes minimizing angular deviations.
## Reward Function
We tried the following reward functions:

- First version:
  - **Positive Terms**:
    - Maintaining angles near the upright position for both pendulum links.
    - Maintaining position near the center for the trolley.
  - **No Penalty for Failures**

- Second version:
  - **Stability Terms**:
    - Instability compute as Maintaining angles near the upright position and Maintaining position near the center for the trolley.
    - Stability as the exponential of the negative instability: the stability will increase the reward function if it is near the goal, and decrease the reward funtion if it is away from the goal.
  - **Force punishment**: Derivative of the speed, this will dismunish the reward function if there is too much variation in the speed.
  - **Penalty for Failures**: Simulation reset after failure + penalty if the trolley reach the border.

- Final version:
  - **Positive Terms**: Angles close to upright position for both pendulum links.  
  - **Negative Terms**: Position far the center for the trolley.
  - **Penalty for Failures**: Simulation reset after failure + penalty if the trolley reach the border.


## Improvement
- Run with GPU for longer training
- DDPG implementation

## Comment
- Better to have a graphic card to run the code. With CPU - intel i7 - 8650U, DQN training last for 10h


## Sources and inspirations
- ros-gz: https://github.com/gazebosim/ros_gz/tree/ros2/ros_gz_sim_demos
- RL definition: https://www.ibm.com/think/topics/reinforcement-learning
- RL: https://www.sciencedirect.com/science/article/abs/pii/S0952197623017025
- Gym environment documentation: https://www.gymlibrary.dev/api/core/
- PPO: https://github.com/ericyangyu/PPO-for-Beginners 
- DQN: 
   - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0280071, 
   - https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py
- Reinforce (and DDQP + DQN) : https://github.com/fredrikmagnus/RL-for-Inverted-Pendulum
- DDPG : 
   - https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb
   - https://github.com/openai/spinningup/blob/master/docs/algorithms/ddpg.rst

