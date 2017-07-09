#!/bin/bash          
echo '## #################################################################################### ##'
echo '# EXAMPLE SCRIPT THAT TRAINS A CAE AND COMPARES CNN ACCURACIES WITH/WITHOUT PRE-TRAINING #'    
echo '## #################################################################################### ##'

# current folder:
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_ROOT_DIR=$PARENT_DIR

echo "Working in $PARENT_DIR"


DATASET="CKPLUS"
TEST_SET_BOOL=false

CAE_LOG_FOLDER="01_CKPLUS_CAE_script"
CNN_LOG_FOLDER="01_CKPLUS_CNN_script"

CAE_CONFIG_FILE_PATH="train_scripts/ckplus/simple_cae_config.ini"
CNN_CONFIG_FILE_PATH="train_scripts/ckplus/simple_cnn_config.ini"

# -----------------------------------------------------------------------------------------------------------
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXX TRAIN CAE XXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

WEIGHTS_PATH="None"
CAE_RUN_NAME="CAE"
REGULARIZATION_FACTOR="0."

python train_and_test_cae.py $DATASET $CAE_CONFIG_FILE_PATH $WEIGHTS_PATH $CAE_LOG_FOLDER $CAE_RUN_NAME $REGULARIZATION_FACTOR $TRAINING_ROOT_DIR

# -----------------------------------------------------------------------------------------------------------
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN REFERENCE CNN XXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

INIT_MODE="default"
WEIGHTS_PATH="None"
RUN_NAME="CNN_rand_init"

python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $PARENT_DIR

# -----------------------------------------------------------------------------------------------------------
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN PRE-TRAINED CNN XXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

INIT_MODE="pre_trained_encoding"
WEIGHTS_PATH="weights/$CAE_LOG_FOLDER/$CAE_RUN_NAME"
RUN_NAME="CNN_pretrained_init"

python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $PARENT_DIR
