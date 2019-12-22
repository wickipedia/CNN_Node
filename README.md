# cnn_node

ROS package for lane following on the Duckiebot. 

## Funcionality
A node named _cnn_node_ is initialized. The node subscribes to _/camera_node/image/compressed_. When the node a new image message receives the image is turned into grayscale and cropped. The cropped grayscale image is the input to the convolutional neural network (CNN). The ouput of the CNN is the relative pose, distance to middle lane and relative angle, of the Duckiebot. The relative pose is used to computed the control signal (velocity and angular velocity) of the Duckiebot using a PID controller. Eventually, the control signal is published in the topic _cnn_node/car_cmd_.

In the following is a detailed explanation of the CNN and PID controller used in the _cnn_node_.

### CNN
Two different CNN Models are used to compute the distance to the middle lane and the relative heading. The trained models can be found in [packages/cnn_node/models](packages/cnn_node/models) as a Pytorch statedict model. The Pytorch classes with the models are in [pckages/cnn_node/include/dt_cnn/model.py](packages/cnn_node/include/dt_cnn/model.py)



### Controller
The controller used for this project made use of the already implemented PI-controller of [dt-core](https://github.com/duckietown/dt-core/tree/daffy/packages/lane_control) node. It is a tuned, cleaned-up and asymetric PID-controller. It weighs deviations to the outside of the tile more than towards the middle lane.
The controllers can be found in [packages/cnn_node/include/controller/controller.py](packages/cnn_node/include/controller/controller.py)
