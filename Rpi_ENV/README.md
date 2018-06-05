Install Kubernetes on one server(x86) and serveral workers(Rpis)
====
### Install Docker & Kubernetes on Server
##### Use official tutorial
[Kubernetes](https://kubernetes.io/docs/tasks/tools/install-kubeadm/)
### Install Docker on Rpis
```
$ wget -O docker.deb https://apt.dockerproject.org/repo/pool/main/d/docker-engine/docker-engine_1.13.1-0~raspbian-jessie_armhf.deb
$ sudo dpkg -i docker.deb
```
#### Turn off swap to prevent install error during Kubernetes
```
$ sudo dphys-swapfile swapoff &&  sudo dphys-swapfile uninstall &&  sudo update-rc.d dphys-swapfile remove
```

#### Add CGrounp Support
```
$ sudo vim /boot/cmdline.txt
// Add cgroup_enable=cpuset cgroup_enable=memory to the end of file
$ sudo reboot // to load new setting
```

### Install Kubernetes on Rpi
```
$ sudo su -
$ curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
$ echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" > /etc/apt/sources.list.d/kubernetes.list
$ apt-get update && apt-get install -y kubeadm=1.10.2-00
$ sudo apt-get install kubelet=1.10.2-00
//1.10.3 currently is not available
```

### Special Setup I did
> Because The master still cannot see my nodes, I suffer 1 day and found the below solution

```
$ sudo vim /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
// Comment # Environment="KUBELET_NETWORK_ARGS=--network-plugin=cni --cni-conf-dir=/etc/cni/net.d --cni-bin-dir=/opt/cni/bin"
$ sudo systemctl restart kubelet
$ systemctl daemon-reload
```
> Create arm proxy for raspberry pi
```
$ kubectl edit daemonset kube-proxy --namespace=kube-system
```
> and follow the steps to create a new proxy yaml file
[Link](https://gist.github.com/squidpickles/dda268d9a444c600418da5e1641239af)

### Set up Helm for Rpi
```
$ kubectl create serviceaccount --namespace kube-system tiller
$ kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
$ kubectl patch deploy --namespace kube-system tiller-deploy -p '{"spec":{"template":{"spec":{"serviceAccount":"tiller"}}}}'
$ kubectl set image deploy/tiller-deploy tiller=luxas/tiller:v2.6.1 --namespace kube-system
```

### References
1. [Close swap space](https://www.hanselman.com/blog/HowToBuildAKubernetesClusterWithARMRaspberryPiThenRunNETCoreOnOpenFaas.aspx)
2. [Install Docker on Rpi](https://forums.docker.com/t/how-can-i-install-a-specific-version-of-the-docker-engine/1993/4)