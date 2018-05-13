import tensorflow as tf
import numpy as np
import math

def autoencoder(input_shape, n_filters, filter_sizes):

    # Placeholder which will then be fed input to the network:
    input = tf.placeholder(tf.float32, input_shape, name='input')

    # Convert input to square tensor:
    x_dim = np.sqrt(input.get_shape().as_list()[1])
    if x_dim != int(x_dim): raise ValueError('Unsupported input dimensions')
    x_dim = int(x_dim)
    x_tensor = tf.reshape(input, [-1, x_dim, x_dim, n_filters[0]])
    current_input = x_tensor

    # Build the encoder by iterating over the n_filters and adding a convolution layer for each item:
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())

        # Random-initialized tensor representing the layer:
        W = tf.Variable(
            tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
                -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))

        # Activation function:
        b = tf.Variable(tf.zeros([n_output]))

        encoder.append(W)
        output = leak(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # Store the latent representation
    latent = current_input

    # Now build the decoder. Reverse the layers first:
    encoder.reverse()
    shapes.reverse()

    # Build the decoder layer-by-layer using the encoder weights:
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = leak(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(input)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # The result is the reconstruction of the inout:
    output = current_input

    # Cost function that measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(output - x_tensor))

    return {'input': input, 'latent': latent, 'output': output, 'cost': cost}

import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

def run_on_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    mnist_len = len(mnist.train.images[0])
    mnist_dim = int(np.sqrt(mnist_len))

    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(input_shape=[None, mnist_dim * mnist_dim], n_filters=[1, 10, 10, 10], filter_sizes=[3, 3, 3, 3])

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    tf_session = tf.Session()
    tf_session.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
        train = None
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            tf_session.run(optimizer, feed_dict={ae['input']: train})
        print(epoch_i, tf_session.run(ae['cost'], feed_dict={ae['input']: train}))

    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    reconstructed = tf_session.run(ae['output'], feed_dict={ae['input']: test_xs_norm})

    fig, axs = plt.subplots(2, n_examples)
    for example_i in range(n_examples):
        axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (mnist_dim, mnist_dim)))
        axs[1][example_i].imshow(np.reshape(np.reshape(reconstructed[example_i, ...], (mnist_len,)) + mean_img, (mnist_dim, mnist_dim)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

def leak(x):
    p = 0.2
    with tf.variable_scope("leak"):
        f1 = 0.5 * (1 + p)
        f2 = 0.5 * (1 - p)
        return f1 * x + f2 * abs(x)

if __name__ == '__main__':
    run_on_mnist()
