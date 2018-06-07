#!/bin/bash

kubectl label node yclo-pi-1 device=0 --overwrite
kubectl label node yclo-pi-2 device=1 --overwrite
kubectl label node yclo-pi-3 device=2 --overwrite
