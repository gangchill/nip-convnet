#!/bin/bash  

# RUN all mnist pre-trained vs random-init experiments 

# bash experiments/paper_reference_experiments/paper_reference_cnn/mnist/mnist_1k/evaluate_pre_training.sh
bash experiments/paper_reference_experiments/paper_reference_cnn/mnist/mnist_10k/evaluate_pre_training.sh
bash experiments/paper_reference_experiments/paper_reference_cnn/mnist/mnist_full/evaluate_pre_training.sh

