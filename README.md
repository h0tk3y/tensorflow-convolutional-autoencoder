# tensorflow-convolutional-autoencoder
A simple TensorFlow application for a convolutional autoencoder tested on MNIST.

This implementation follows the tutorial at [pkmital/tensorflow_tutorials](https://github.com/pkmital/tensorflow_tutorials).

In the demo setup `run_on_mnist`, it interprets the MNIST data as 28x28 inputs and then uses three convolutional layers, each with
10 filters (for the number of output digits), reducing the data item size doubly each time.

The output of the last layer is therefore the latent low-dimensional representation of the input, as per encoding learned by the 
autoencoder.

To reconstruct the data (and check the quality), the latent representation is decoded with the layers built with the same weight tensors
which are used for encoding, and the output (of the same dimensions as the input) is compared pixel-by-pixel with the input — this is the
score function.

#### Example run

Here's an example run log:

```
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
2018-05-14 00:22:38.354169: I C:\tf_jenkins\workspace\tf-nightly-windows\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
0 470.65076
1 399.151
2 349.7995
3 315.3262
4 331.46826
5 296.24704
6 308.81763
7 278.4167
8 292.5509
9 270.76398
(10, 28, 28, 1)
```

The score chart:

![image](https://user-images.githubusercontent.com/1888526/39972512-485b004c-571a-11e8-9dda-4fe3b119f8b1.png)

And the visualized sample, original items at the top, decoded ones at the bottom:

![image](https://user-images.githubusercontent.com/1888526/39972517-64eecacc-571a-11e8-85e7-7c939b040d84.png)

To compare, if we reduce the number of filters to just 5, the decoded items become much noisier:

![image](https://user-images.githubusercontent.com/1888526/39972618-fea8f09c-571b-11e8-8866-1e731dd1c588.png)
