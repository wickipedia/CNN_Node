cmake_minimum_required(VERSION 2.8.3)
project(cnn_node)

find_package(catkin REQUIRED COMPONENTS
  rospy 
  rosbag
  std_msgs
)

catkin_package()
catkin_python_setup()
