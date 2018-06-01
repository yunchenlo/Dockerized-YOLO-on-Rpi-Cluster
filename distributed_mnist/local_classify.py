import tensorflow as tf 
import cv2
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave


x = tf.placeholder(tf.float32, [None, 784], name="input")
W = tf.Variable(tf.zeros([784, 10]), name = "w1")
b = tf.Variable(tf.zeros([10]), name = "b1")
prob=tf.matmul(x, W) + b
y = tf.nn.softmax(prob)

saver = tf.train.Saver()

print("model constructed!")

with tf.Session() as sess:
	# load ckpt
	saver.restore(sess, "saved_model/model.ckpt")
	print("Model restored.")

	#preprocess image
	im1 = imread('7.jpg', mode='L')
	im1 = imresize(im1, (28, 28))
	im1 = im1.reshape((1, 784))
	im1[0] = (im1[0]*1.0 )/max(im1[0]*1.0)* 255.0
	im1 = (255.0-im1)/255.0

	result=sess.run(y,feed_dict={x: im1})
	print(result)