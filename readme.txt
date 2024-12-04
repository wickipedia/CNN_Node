docker run --net=host -e VEHICLE_NAME=queenmary2 -e ROS_MASTER_URI=http://192.168.1.72:11311/ -e ROS_IP=http://192.168.1.206/ -e kPd=-1.5 -e kPp=-4.3 -e kId=0 -e gain=1 -e kIp=1.5 -e kDd=0 -e kDp=0 -it --rm duckietown/cnn_node:v1-amd64

