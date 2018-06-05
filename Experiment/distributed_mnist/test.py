from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import time
import tensorflow as tf

start=time.time()

x = tf.placeholder(tf.float32, [None, 784], name="input")

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"]})

with tf.device("/job:local/task:0"):
    W = tf.Variable(tf.zeros([784, 10]), name = "w1")
    b = tf.Variable(tf.zeros([10]), name = "b1")

time1=time.time()
#print ("Time1: "+str(time1-start)+" seconds")

with tf.device("/job:local/task:1"):
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10], name = "output")

time2=time.time()
#print ("Time2: "+str(time2-time1)+" seconds")
with tf.device("/job:local/task:2"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

time3=time.time()
#print ("Time3: "+str(time3-time2)+" seconds")

# swd--save the Checkpoint file
saver = tf.train.Saver()

#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session("grpc://localhost:2224") as sess:
    # swd save ckpt
    saver.restore(sess, "saved_model/model.ckpt")
    print("\nModel restored.\n")

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
time4=time.time()
#print ("Time4: "+str(time4-time3)+" seconds")

end=time.time()
print ("Computing time: "+str(end-start)+" seconds")