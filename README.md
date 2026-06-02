# Autonomous Mapping Robot

This repository contains the source code and configuration files for an autonomous mapping robot project.

To use it, you need to know little knowledge about ROS to code and run it

## Requirements
Ubuntu 24.04
ROS 2 Jazzy
Gazebo
RViz2

sudo apt install ros-jazzy-sensor-msgs
sudo apt install ros-jazzy-urdf-tutorial
sudo apt install ros-jazzy-tf2-tools
sudo apt install ros-jazzy-xacro
sudo apt-get install ros-jazzy-rqt-image-view
sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers
sudo apt install ros-jazzy-rclcpp-lifecycle

## Features

- ROS 2 based robot system
- Autonomous mapping
- Path planning
- Lidar-based perception
- Robot simulation and visualization

Gazebo(simulation): This is all source code for you can simulate the vehicle in your computer (no need real hardware)
you just need to 


## How to run

colcon build
source install/setup.bash

Simulation:
ros2 launch my_robot_description my_robot.launch.xml 

Real:
ros2 launch real_car_description my_robot.launch.xml 
