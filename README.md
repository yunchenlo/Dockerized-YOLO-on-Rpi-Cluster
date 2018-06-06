# Dockerized-YOLO-on-Rpi-Cluster
## Developer: Yun-Chen Lo

## Description
> Reference Implementation of paper **"Distributed Analytics in Fog Computing Platforms Using TensorFlow and Kubernetes"**

![alt text](https://i.imgur.com/qMFnnzB.png)

## Dependencies
> Tensorflow 1.1.1
> 
> Kubeadm 1.10.2
> 
> kubelet 1.10.2
> 
> docker 1.13.1

## Folder Description
> **Application/yolo/**
> > single/ //write yolo for one machine
> > 
> > distributed/ //break yolo into three part, which could be handled by different devices
> 
> > docker-image/ //contains final distibuted yolo version and required Dockerfile
> 
> **Deployment/**
> >
> > helmChart/yolo/ //contains params to be passed during execution and worker.yaml, master.yaml
> > 1_node.sh
> > 2_node.sh
> > 3_node.sh
> 
> **Rpi_ENV/**
> 
> > describe my steps to setup a cluster with one pc as master and several Rpis as workers
> 
> **Experiment/**
> 
> > contains several files that I have experimented before

## References
1. [tensorflow YOLO implementation](https://github.com/nilboy/tensorflow-yolo )
2. [How to partition YOLO](https://github.com/WakeupTsai/tensorflow-applications)
3. [How to write Helm diagram](https://github.com/WakeupTsai/FogComputingPlatform-Auto-Deploy)

## Special Thanks to [Hua-Jun Hong](https://scholar.google.com/citations?user=NRAgnj4AAAAJ&hl=en)'s Help to this Project