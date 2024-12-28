Prise en main de roz-gz

# dossier de démos ROS
/opt/ros/jazzy/share/ros_gz_sim_demos/models/


# lancer un monde appelé default avec le double pendule
QT_QPA_PLATFORM=xcb ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=/opt/ros/jazzy/share/ros_gz_sim_demos/models/double_pendulum_model.sdf


# lancer un monde vide, appelée empty
QT_QPA_PLATFORM=xcb ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=empty.sdf

# ajouter le véhicule au monde "default"
ros2 launch ros_gz_sim gz_spawn_model.launch.py world:=default file:=$(ros2 pkg prefix --share ros_gz_sim_demos)/models/vehicle/model.sdf name:=my_vehicle x:=5.0 y:=5.0 z:=0.5

# create the default world, and add the pendulum to it
QT_QPA_PLATFORM=xcb ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=/home/aymeric/Documents/projetIA/default_world.sdf
ros2 launch ros_gz_sim gz_spawn_model.launch.py world:=default file:=/home/aymeric/Documents/projetIA/double_pendulum_rail.sdf name:=my_pendulum

