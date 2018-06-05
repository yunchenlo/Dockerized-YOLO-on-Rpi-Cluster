import sys
task_number=int(sys.argv[1])
import tensorflow as tf

macc1="localhost:2222"
macc2="localhost:2223"

cluster = tf.train.ClusterSpec({"local":[macc1,macc2]})
server = tf.train.Server(cluster, job_name="local", task_index=task_number)

print("Server starting #{}".format(task_number))

server.start()
server.join()

