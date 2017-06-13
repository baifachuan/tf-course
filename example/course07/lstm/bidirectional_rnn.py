
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/mnist", one_hot=True)

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 1000
batch_size = 128
display_step = 10

#todo Network Parameters
n_input =  # MNIST data input (img shape: 28*28)
n_steps =  # timesteps
n_hidden =  # hidden layer num of features
n_classes =  # MNIST total classes (0-9 digits)

#todo tf Graph input
x =
y =

#todo Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out':
}
biases = {
    'out':
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    #todo Permuting batch_size and n_steps
    x =
    #todo Reshape to (n_steps*batch_size, n_input)
    x =
    #todo Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x =

    # Define lstm cells with tensorflow
    #todo Forward direction cell
    lstm_fw_cell =
    #todo Backward direction cell
    lstm_bw_cell =

    #todo Get lstm cell output
    try:
        outputs, _, _ =
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    #todo  Linear activation, using rnn inner loop last output
    return

pred = BiRNN(x, weights, biases)

#todo Define loss and optimizer
cost =
optimizer =

#todo Evaluate model
correct_pred =
accuracy =

#todo Initializing the variables
init =

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
