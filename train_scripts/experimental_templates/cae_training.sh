#!/bin/bash
echo '## ########## ##'
echo '# CAE TRAINING #'    
echo '## ########## ##'

# example script to train a cae on the MNIST dataset. 
# SUPPOSES IT GETS CALLED FROM THE ROOT DIRECTORY!!! (nip-convnet)
# This is supposed to be a template/demo, make a copy of it if you want to change anything

# LOG FOLDER, logs are stored in logs/$LOG_FOLDER, weights in weights/$LOG_FOLDER
LOG_FOLDER="CAE_MNIST_demo"

# General settings, all additional settings can be adjusted in the config file
DATASET="MNIST"									# currently available datasets: "MNIST", "MNIST_SMALL", "CIFAR10", "CKPLUS"
CONFIG_FILE_PATH="configs/CAE/cae_2l_tanh.ini"	# config file, some examples are stored in nip-convnet/configs
WEIGHTS_PATH="None"								# replace "None" with a path to a tensorflow checkpoint to initialize the weights with this checkpoint
REGULARIZATION_FACTOR="0."						# L1 regularization factor on the encoding representation

RUN_NAME="your_run_name" # change to custom run name if desired, "None" uses the generated run name defined in train_and_test_cae.py
python train_and_test_cae.py $DATASET $CONFIG_FILE_PATH $WEIGHTS_PATH $LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $REGULARIZATION_FACTOR

# copy the script itself to the log folder (useful to have all run information in one place to keep track of the experiments)
# assumes that the training script is called from the root directory (nip-convnet)
# ATTENTION: currently only works if RUN_NAME != "NONE"
cp train_scripts/experimental_templates/cae_training.sh logs/$LOG_FOLDER/$RUN_NAME/cae_training.sh