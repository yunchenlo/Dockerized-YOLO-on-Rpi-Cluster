import sys

sys.path.append('./')

#from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

def process_predicts(predicts):
	p_classes = predicts[0, :, :, 0:20]
	C = predicts[0, :, :, 20:22]
	coordinate = predicts[0, :, :, 22:]

	p_classes = np.reshape(p_classes, (7, 7, 1, 20))
	C = np.reshape(C, (7, 7, 2, 1))

	P = C * p_classes

	index = np.argmax(P)

	index = np.unravel_index(index, P.shape)

	class_num = index[3]

	coordinate = np.reshape(coordinate, (7, 7, 2, 4))

	max_coordinate = coordinate[index[0], index[1], index[2], :]

	biascenter = max_coordinate[0]
	ycenter = max_coordinate[1]
	w = max_coordinate[2]
	h = max_coordinate[3]

	biascenter = (index[1] + biascenter) * (448/7.0)
	ycenter = (index[0] + ycenter) * (448/7.0)

	w = w * 448
	h = h * 448

	biasmin = biascenter - w/2.0
	ymin = ycenter - h/2.0

	biasmax = biasmin + w
	ymax = ymin + h

	return biasmin, ymin, biasmax, ymax, class_num

''' Constant Value '''
alpha = 0.1
cell_size = 7
num_classes = 20
boxes_per_cell = 2
image_size = 448

''' Input Image '''
image = tf.placeholder(tf.float32, (1, 448, 448, 3))

''' CONV 1 '''
with tf.variable_scope("conv1") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 3, 16]), name = "weights")
	biases = tf.Variable(tf.zeros([16]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv1 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' MAxPOOL 1 '''
maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

''' CONV 2 '''
with tf.variable_scope("conv2") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 16, 32]), name = "weights")
	biases = tf.Variable(tf.zeros([32]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(maxpool1, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv2 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' MAxPOOL 2 '''
maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

''' CONV 3 '''
with tf.variable_scope("conv3") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 32, 64]), name = "weights")
	biases = tf.Variable(tf.zeros([64]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(maxpool2, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv3 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' MAxPOOL 3 '''
maxpool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

''' CONV 4 '''
with tf.variable_scope("conv4") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 64, 128]), name = "weights")
	biases = tf.Variable(tf.zeros([128]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(maxpool3, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv4 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' MAxPOOL 4 '''
maxpool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

''' CONV 5 '''
with tf.variable_scope("conv5") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 128, 256]), name = "weights")
	biases = tf.Variable(tf.zeros([256]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(maxpool4, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv5 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' MAxPOOL 5 '''
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

''' CONV 6 '''
with tf.variable_scope("conv6") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 256, 512]), name = "weights")
	biases = tf.Variable(tf.zeros([512]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(maxpool5, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv6 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' MAxPOOL 6 '''
maxpool6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                  padding='SAME')

''' CONV 7 '''
with tf.variable_scope("conv7") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 512, 1024]), name = "weights")
	biases = tf.Variable(tf.zeros([1024]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(maxpool6, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv7 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' CONV 8 '''
with tf.variable_scope("conv8") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 1024, 1024]), name = "weights")
	biases = tf.Variable(tf.zeros([1024]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv8 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' CONV 9 '''
with tf.variable_scope("conv9") as scope:
	kernel = tf.Variable(tf.zeros([3, 3, 1024, 1024]), name = "weights")
	biases = tf.Variable(tf.zeros([1024]), name = "biases")
	# conv layer
	conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, biases)
# leaky relu
bias = tf.cast(bias, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
conv9 = 1.0 * mask * bias + alpha * (1 - mask) * bias

temp_conv = tf.transpose(conv9, (0, 3, 1, 2))

''' LOCAL 1 '''
with tf.variable_scope("local1") as scope:
	weights = tf.Variable(tf.zeros([50176, 256]), name = "weights")
	biases = tf.Variable(tf.zeros([256]), name = "biases")
	reshape = tf.reshape(temp_conv, [tf.shape(temp_conv)[0], -1])
	# local layer
	local = tf.matmul(reshape, weights) + biases
# leaky relu
bias = tf.cast(local, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
local1 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' LOCAL 2 '''
with tf.variable_scope("local2") as scope:
	weights = tf.Variable(tf.zeros([256, 4096]), name = "weights")
	biases = tf.Variable(tf.zeros([4096]), name = "biases")
	reshape = tf.reshape(local1, [tf.shape(local1)[0], -1])
	# local layer
	local = tf.matmul(reshape, weights) + biases
# leaky relu
bias = tf.cast(local, dtype=tf.float32)
bool_mask = (bias > 0)
mask = tf.cast(bool_mask, dtype=tf.float32)
local2 = 1.0 * mask * bias + alpha * (1 - mask) * bias

''' LOCAL 3 '''
with tf.variable_scope("local3") as scope:
	weights = tf.Variable(tf.zeros([4096, 1470]), name = "weights")
	biases = tf.Variable(tf.zeros([1470]), name = "biases")
	reshape = tf.reshape(local2, [tf.shape(local1)[0], -1])
	# local layer
	local = tf.matmul(reshape, weights) + biases
# no leaky relu
local3 = tf.identity(local, name = "local3")

n1 = cell_size * cell_size * num_classes

n2 = n1 + cell_size * cell_size * boxes_per_cell

class_probs = tf.reshape(local3[:, 0:n1], (-1, cell_size, cell_size, num_classes))
scales = tf.reshape(local3[:, n1:n2], (-1, cell_size, cell_size, boxes_per_cell))
boxes = tf.reshape(local3[:, n2:], (-1, cell_size, cell_size, boxes_per_cell * 4))

local3 = tf.concat([class_probs, scales, boxes], 3)

predicts = local3

saver = tf.train.Saver()

print("model constructed!")

sess = tf.Session()
np_img = cv2.imread('cat.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

np_predict = sess.run(predicts, feed_dict={image: np_img})

xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out.jpg', resized_img)
sess.close()