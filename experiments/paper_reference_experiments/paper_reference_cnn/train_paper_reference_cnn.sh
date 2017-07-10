#!/bin/bash          
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN REFERENCE CNN XXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# current folder:
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_ROOT_DIR=$PARENT_DIR

echo "Working in $PARENT_DIR"

DATASET="MNIST_SMALL"
CNN_LOG_FOLDER="1k_MNIST_paper_reference_model"
CNN_CONFIG_FILE_PATH="$PARENT_DIR/paper_reference_cnn.ini"

# if set to true, a test run using the test set is performed after the training
TEST_SET_BOOL=false


echo "Begin training the reference network"

INIT_MODE="default"
WEIGHTS_PATH="None"
CNN_RUN_NAME="reference_cnn_sslr_ib"


python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $CNN_RUN_NAME $TEST_SET_BOOL $PARENT_DIR