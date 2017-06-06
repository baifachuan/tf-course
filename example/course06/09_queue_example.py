from __future__ import print_function
import tensorflow as tf
import numpy as np

N_SAMPLES = 1000
NUM_THREADS = 4
# Generating some simple data
# todo create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
data =
# todo create 1000 random labels of 0 and 1
target =
# todo create FIFOQueue
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

enqueue_op = queue.enqueue_many([data, target])
data_sample, label_sample = queue.dequeue()

# create ops that do something with data_sample and label_sample

# todo create NUM_THREADS to do enqueue
qr =
with tf.Session() as sess:
    # create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for step in range(100):  # do to 100 iterations
        if coord.should_stop():
            break
        data_batch, label_batch = sess.run([data_sample, label_sample])
    coord.request_stop()
    coord.join(enqueue_threads)
