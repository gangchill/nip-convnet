# nip-convnet

## execution of current version
Dependencies:
* tensorflow (tested with 1.1.0 )
* python 2.7 (tested with 2.7.12)
* matplotlib (tested with 1.5.1 )
* pandas (tested with 0.20.2)
* Pillow (tested with 4.1.1)
* scipy (tested with 0.19.0)
* scikit-learn (tested with 0.18.1)

To train and test a simple single-layer autoencoder on the MNIST dataset, simply call 'python train_and_test_simple_mnist_autoencoder.py'

## project description
We want to train a neural network to classify images. Before we do that, an Autoencoder is trained for the network to pertain information of its input. The weights obtained from training the autoencoder are used for initializing a neural network for image classification. It has been shown that this pre-training of the network allows for obtaining higher generalization performance than when starting from a random weight initialization. This project will be about using a convolutional architecture for the Autoencoder that is well suited for visual data in obtaining said improved weight initialization. Initially we will reproduce the experiment of following paper:

Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011). Stacked convolutional auto-encoders for hierarchical feature extraction. Artificial Neural Networks and Machine Learning–ICANN 2011, 52-59. 

Other Relevant Papers:
* Bengio et al. Representation Learning: A Review and New Perspectives

* Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. Advances in neural information processing systems, 19, 153.

* Makhzani, A., & Frey, B. (2014, September). A winner-take-all method for training sparse convolutional autoencoders. In NIPS Deep Learning Workshop.

* D. Hamester, P. Barros and S. Wermter, Face expression recognition with a 2-channel Convolutional Neural Network, 2015 International Joint Conference on Neural Networks (IJCNN), Killarney, 2015, pp. 1-8.

Datasets:
* http://www.pitt.edu/~emotion/ck-spread.htm

Tutorials: 
* https://www.tensorflow.org/tutorials/
