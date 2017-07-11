# CNN results for restricted MNIST (1k train images)

CNN architecture as described in paper:

The network has 6 hidden layers: 1) convolutional layer with 100 5x5 filters per
input channel; 2) max-pooling layer of 2x2; 3) convolutional layer with 150 5x5
filters per map; 4) max-pooling layer of 2x2; 5) convolutional layer of 200 maps
of size 3x3; 6) a fully-connected layer of 300 hidden neurons. The output layer
has a softmax activation function with one neuron per class.

## First run results 
Test set accuracy: 
- 0.9227 (random init)
- 0.9253 (pre-trainig)

Training stats:
![training results](results/1k_mnist_training_results.png)

First layer filter comparison:
- pre-trained:
![pre-trained filter](results/flf_pre_trained.png)

- random-init:
![random-init filter](results/flf_random_init.png)
