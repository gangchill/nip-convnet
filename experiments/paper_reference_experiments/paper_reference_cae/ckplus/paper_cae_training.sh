#!/bin/bash
echo '## ########## ##'
echo '# CAE TRAINING #'    
echo '## ########## ##'

# same script as ../cae_trainin.sh but this one keeps all variables in a seperate folder next to the script to keep track of different experiments and have everything in one place
# current folder:
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_ROOT_DIR=$PARENT_DIR

echo "training root is $TRAINING_ROOT_DIR"

# example script to train a cae on the MNIST dataset. 
# SUPPOSES IT GETS CALLED FROM THE ROOT DIRECTORY!!! (nip-convnet)
# This is supposed to be a template/demo, make a copy of it if you want to change anything

# LOG FOLDER, logs are stored in logs/$LOG_FOLDER, weights in weights/$LOG_FOLDER
LOG_FOLDER="mnist_paper_net_cae_2"

# General settings, all additional settings can be adjusted in the config file
DATASET="CKPLUS"									# currently available datasets: "MNIST", "MNIST_SMALL", "CIFAR10", "CKPLUS"
CONFIG_FILE_PATH="$TRAINING_ROOT_DIR/paper_cae.ini"	# config file, some examples are stored in nip-convnet/configs
WEIGHTS_PATH="None"								# replace "None" with a path to a tensorflow checkpoint to initialize the weights with this checkpoint
REGULARIZATION_FACTOR="0."						# L1 regularization factor on the encoding representation

RUN_NAME="paper_cae_test_0.1" # change to custom run name if desired, "None" uses the generated run name defined in train_and_test_cae.py
python train_and_test_cae.py $DATASET $CONFIG_FILE_PATH $WEIGHTS_PATH $LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $REGULARIZATION_FACTOR $TRAINING_ROOT_DIR
