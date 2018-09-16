import tensorflow as tf
import numpy as np
import os

# Suppress tensorflow AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# DeepSense framework, type(features) == dict, type(features['x']) == tensor in R^(d * 2f * tau)
def ds(features, labels, mode):
    # Each sensor gives a tensor input, each input goes into ds_individual
    X = []
    for key in features.keys():
        for i in range(features[key].shape):
            m = features['vision'][:,:,i]
            a = ds_cnn_individual(m)
            X.append(np.matrix.flatten(a))
    # Flatten and concatenate all outputs, go into the same subnet architecture with different kernel sizes
    X = np.array(X)
    print("Final shape: ", X.shape)
    # Recurrent layers
    return X


# DeepSense individual mode CNN, m is a mode of d dimension under DFT in interval of 1 sec, type(m) = numpy matrix, shape = (d, 2f)
def ds_cnn_individual(m):
    d = m.shape[0]

    # Input M = (d, 2f), individual feature at a time
    input_layer = tf.reshape(m.shape)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[d, 16],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=64,
        kernel_size=16,
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=4)
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=64,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu
    )
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2)
    return pool3
