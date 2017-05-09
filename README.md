# nip-convnet

We want to train a neural network to classify images. Before we do that, an Autoencoder is trained for the network to pertain information of its input. The weights obtained from training the autoencoder are used for initializing a neural network for image classification. It has been shown that this pre-training of the network allows for obtaining higher generalization performance than when starting from a random weight initialization. This project will be about using a convolutional architecture for the Autoencoder that is well suited for visual data in obtaining said improved weight initialization. Initially we will reproduce the experiment of following paper:

Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011). Stacked convolutional auto-encoders for hierarchical feature extraction. Artificial Neural Networks and Machine Learning–ICANN 2011, 52-59. 

Other Relevant Papers:
* Introductory paper on pre-training using Auto-encoders: Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. Advances in neural information processing systems, 19, 153.

* Makhzani, A., & Frey, B. (2014, September). A winner-take-all method for training sparse convolutional autoencoders. In NIPS Deep Learning Workshop.

* D. Hamester, P. Barros and S. Wermter, Face expression recognition with a 2-channel Convolutional Neural Network, 2015 International Joint Conference on Neural Networks (IJCNN), Killarney, 2015, pp. 1-8.

Datesets:
* http://www.pitt.edu/~emotion/ck-spread.htm

Tutorials: 
* https://www.tensorflow.org/tutorials/
