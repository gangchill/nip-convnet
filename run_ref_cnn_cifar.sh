#!/bin/bash  

# RUN all cifar pre-trained vs random-init experiments 


bash experiments/paper_reference_experiments/paper_reference_cnn/cifar/cifar_1k/train_paper_reference_cnn.sh

bash experiments/paper_reference_experiments/paper_reference_cnn/cifar/cifar_10k/train_paper_reference_cnn.sh

bash experiments/paper_reference_experiments/paper_reference_cnn/cifar/cifar_full/train_paper_reference_cnn.sh

