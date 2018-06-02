# Dockerized-YOLO-on-Rpi-Cluster
## Developer: Yun-Chen Lo
## Description
> Reference Implementation of paper **"Distributed Analytics in Fog Computing Platforms Using TensorFlow and Kubernetes"**

## Steps
### Test YOLO in Raspberry Pi's Docker
> 1. Install customized Raspbian OS to Rpi 3 & Rpi 2 using PiBackery [Link](http://www.pibakery.org/)
> 2. Connect laptop or PC to the same wifi that Rpi has connected to
> 3. ssh to your Rpi
> ```
> 	$ ssh user@hostname
> ```
> 4. Install Docker on Rpi [Link](https://blog.hypriot.com/post/run-docker-rpi3-with-wifi/)
> 5. Build the Docker Container yourself or pull from my pre-built one on Rpi
> ```
> $ sudo docker run -it yclo/raspbian-tensorflow-opencv3:first //directly run container
> ```
> **or**
> ```
> $ sudo docker run -it yclo/raspbian-tensorflow-opencv3:first bash //enter container environment
> ```
> you may type in following command in container environmet, it should work!
> ```
> $ time python3 demo.py
> ```
> you should see cat_out.jpg appear in the container folder, get it out from container and get the following photo

#### Before(cat.jpg)
 <img src="https://i.imgur.com/cJnye8f.jpg" width="200">

#### After(cat_out.jpg)
 <img src="https://i.imgur.com/t9Y7hxY.jpg" width="200">

### Break YOLO into parts

> 1. Form a new file demo_new.py, which contains session and model graph definition
> 2. Write demo_dist_2_node.py file, which will be runned on multiple machine
> 3. If you want to test locally, open 3 terminals, and enter tensorflow-yolo folder!
> 
> ```
> [terminal 1]$ python local_server.py 0 // for first worker to listen for running request
> [terminal 2]$ python local_server.py 1 // for second worker to listen for running request
> [terminal 3]$ python demo_dist_2_node.py // run a session and ask two workers to do their Job 
> ```

### Install Kubernetes(a docker management project)


### Write Kubernetes Orchestration File


### Important Links
> 1. [Distributed Tensorflow](https://learningtensorflow.com/lesson11/)
> 2. [YOLO Source](https://github.com/nilboy/tensorflow-yolo)
> 3. [Distributed Tensorflow](https://github.com/nesl/Distributed_TensorFlow)