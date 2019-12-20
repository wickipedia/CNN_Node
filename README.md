# cnn_node

ROS package for lane following on the Duckiebot. 

## Funcionality
A node named _cnn_node_ is initialized. The node subscribes to _/camera_node/image/compressed_. When the node a new image message receives the image is turned into grayscale and cropped. The cropped grayscale image is the input to the convolutional neural network (CNN). The ouput of the CNN is the relative pose, distance to middle lane and relative angle, of the Duckiebot. The relative pose is used to computed the control signal (velocity and angular velocity) of the Duckiebot using a PID controller. Eventually, the control signal is published in the topic _cnn_node/car_cmd_.

In the following is a detailed explanation of the CNN and PID controller used in the _cnn_node_.

### CNN
Two different CNNs are used to compute the distance to the middle lane and the relative heading. The trained models can be found in [packages/cnn_node/models](packages/cnn_node/models) as a Pytorch statedict model. The Pytorch classes with the models are in [pckages/cnn_node/include/dt_cnn/model.py](packages/cnn_node/include/dt_cnn/model.py)



### Controller
The controlles can be found in [packages/cnn_node/include/controller/controller.py](packages/cnn_node/include/controller/controller.py)


## Usage
Before the cnn_node can be executed make sure the custom docker images dt-core and dt-car-interface, and the default docker image dt-duckiebot-interface run on the duckiebot. The docker images dt-core and dt-car-interface can be found in (https://github.com/duckietown-ethz/proj-lfi-ml)[https://github.com/duckietown-ethz/proj-lfi-ml].

To build dt-core and dt-car-interface clone the repo (https://github.com/duckietown-ethz/proj-lfi-ml)[https://github.com/duckietown-ethz/proj-lfi-ml] and change into the folder containing the docker image and build it with:

```
dts devel build -f --arch arm32v7 -H VEHICLE_NAME.local
```
Before you run the custom docker images dt-core and dt-car-interface make sure no other dt-core and dt-car-interface run on the Duckiebot.

To run the docker images execute
```
docker -H VEHICLE_NAME.local run --name dt-core-cnn -v /data:/data --privileged --network=host -dit --restart unless-stopped duckietown/dt-core:daffy-arm32v7
```
```
docker -H VEHICLE_NAME
.local run --name dt-core-cnn -v /data:/data --privileged --network=host -dit --restart unless-stopped duckietown/dt-car-interface:daffy-arm32v7
```

