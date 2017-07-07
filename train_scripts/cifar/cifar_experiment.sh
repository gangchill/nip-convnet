#!/bin/bash          
echo '## #################################################################################### ##'
echo '# EXAMPLE SCRIPT THAT TRAINS A CAE AND COMPARES CNN ACCURACIES WITH/WITHOUT PRE-TRAINING #'    
echo '## #################################################################################### ##'

GLOBAL_LOG_FILE="train_scripts/train_logs.txt"

DATASET="CIFAR10"

CAE_LOG_FOLDER="01_CAE_script"
CNN_LOG_FOLDER="01_CNN_script"

CAE_CONFIG_FILE_PATH="train_scripts/cifar/simple_cae_config.ini"
CNN_CONFIG_FILE_PATH="train_scripts/cifar/simple_cnn_config.ini"

# -----------------------------------------------------------------------------------------------------------
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXX TRAIN CAE XXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

WEIGHTS_PATH="None"
CAE_RUN_NAME="CAE"

python train_and_test_cae.py $DATASET $CAE_CONFIG_FILE_PATH $WEIGHTS_PATH $CAE_LOG_FOLDER $CAE_RUN_NAME >> $GLOBAL_LOG_FILE

# -----------------------------------------------------------------------------------------------------------
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN REFERENCE CNN XXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

INIT_MODE="default"
WEIGHTS_PATH="None"
RUN_NAME="CNN_rand_init"
TEST_SET_BOOL=false

python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $RUN_NAME $TEST_SET_BOOL >> $GLOBAL_LOG_FILE

# -----------------------------------------------------------------------------------------------------------
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN PRE-TRAINED CNN XXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

INIT_MODE="pre_trained_encoding"
WEIGHTS_PATH="weights/$CAE_LOG_FOLDER/$CAE_RUN_NAME"
LOG_FOLDER="01_CNN_script"
RUN_NAME="CNN_rand_init"
TEST_SET_BOOL=false

python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $RUN_NAME $TEST_SET_BOOL >> $GLOBAL_LOG_FILE