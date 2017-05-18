import tensorflow as tf

from layers import *


def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu

    conv1 = conv(input, "conv1", [3, 3, 1], [2, 2], padding='SAME')

    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu

    conv2 = conv(conv1, "conv2", [3, 3, 8], [2, 2], padding='SAME')

    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu

    conv3 = conv(conv2, "conv3", [3, 3, 8], [2, 2], padding='SAME')

    # FC: output_dim: 100, no non-linearity
    return fc(conv3, "fc", 100, non_linear_fn=None)


def decoder(input):

    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu

    fc_de = fc(input, "decode_fc", 128)

    # Reshape to [batch_size, 4, 4, 8]
    fc_de = tf.reshape(fc_de, [-1, 4, 4, 8])

    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu

    deconv1 = deconv(fc_de, "deconv1", [3, 3, 8], [2, 2])

    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu

    deconv2 = deconv(deconv1, "deconv2", [8, 8, 1], [2, 2], padding='VALID')

    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    deconv3 = deconv(deconv2, "deconv3", [7, 7, 1], [1, 1], padding='VALID',
                     non_linear_fn=tf.sigmoid)

    return deconv3


def autoencoder(input_shape):
    # Define place holder with input shape
    X = tf.placeholder(tf.float32, input_shape)

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(X)

        # Pass encoding into decoder to obtain reconstructed image
        decoding = decoder(encoding)

        # Return input image (placeholder) and reconstructed image
    return X, decoding
