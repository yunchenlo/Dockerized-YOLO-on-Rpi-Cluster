import time
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import numpy as np

from flask import Flask
from flask import request

from time import sleep

import threading

macc1="localhost:2222"
macc2="localhost:2223"
macc3="localhost:2224"

cluster = tf.train.ClusterSpec({"DSGraph":[macc1,macc2,macc3]})