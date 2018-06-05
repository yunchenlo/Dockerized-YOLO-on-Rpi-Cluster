import tensorflow as tf 
import cv2
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import threading
# cluster network hostname & port
macc1="localhost:2222"
macc2="localhost:2223"



cluster = tf.train.ClusterSpec({"local":[macc1,macc2]})

mac1="grpc://localhost:2222"
mac2="grpc://localhost:2223"
inputsess=[mac1, mac2]

def main():
	thread=[]
	worker(1)
	'''
	for i in range(len(inputsess)):
		t=threading.Thread(target=worker,args=(i,))
		thread.append(t)
		t.start()
		t.join()
	'''

def worker(i):
	global inputsess
	# worker 1
	with tf.device("/job:local/task:0"):
		x = tf.placeholder(tf.float32, [None, 784], name="input")
		W = tf.Variable(tf.zeros([784, 10]), name = "w1")

	# worker 2
	with tf.device("/job:local/task:1"):
		b = tf.Variable(tf.zeros([10]), name = "b1")
		prob=tf.matmul(x, W) + b
		y = tf.nn.softmax(prob)

	saver = tf.train.Saver()

	print("model constructed!")

	with tf.Session(inputsess[i]) as sess:
		# load ckpt
		saver.restore(sess, "saved_model/model.ckpt")
		print("Model restored.")

		#preprocess image
		im1 = imread('7.jpg', mode='L')
		im1 = imresize(im1, (28, 28))
		im1 = im1.reshape((1, 784))
		im1[0] = (im1[0]*1.0 )/max(im1[0]*1.0)* 255.0
		im1 = (255.0-im1)/255.0
		for i in range(1):
			result=sess.run(y,feed_dict={x: im1})
			print(result)


if __name__=='__main__':
    main()