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
> 4. Build the Docker Container yourself or pull from my pre-built one on Rpi
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

### Break YOLO into part

> 1.


### Install Kubernetes(a docker management project)
