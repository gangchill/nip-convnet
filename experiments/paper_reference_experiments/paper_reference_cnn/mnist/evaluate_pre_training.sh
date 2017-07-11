#!/bin/bash          
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN REFERENCE CNN XXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# current folder:
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_ROOT_DIR=$PARENT_DIR

echo "Working in $PARENT_DIR"

DATASET="MNIST_SMALL"
CNN_LOG_FOLDER="1k_mnist_final"
CNN_CONFIG_FILE_PATH="$PARENT_DIR/paper_reference_cnn.ini"
# CNN_CONFIG_FILE_PATH="experiments/experimental_templates/cnn_in_folder_demo/relu_cnn.ini"

# if set to true, a test run using the test set is performed after the training
TEST_SET_BOOL=True

echo "Begin training the reference network"

INIT_MODE="resume"
WEIGHTS_PATH="None"
CNN_RUN_NAME="reference-random_init"


python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $CNN_RUN_NAME $TEST_SET_BOOL $PARENT_DIR 

echo "Begin training the pre-trained network"

INIT_MODE="pre_trained_encoding"
WEIGHTS_PATH="experiments/paper_reference_experiments/paper_reference_cae/weights/mnist_paper_net_cae_2/paper_cae_test_0.05"
CNN_RUN_NAME="pre-trained"


python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $CNN_RUN_NAME $TEST_SET_BOOL $PARENT_DIR

