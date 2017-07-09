#!/bin/bash          
echo '## ################################### ##'
echo '# DENSE LAYER OPTIMIZATION SANITY CHECK #'
echo '## ################################### ##'

# current folder:
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_ROOT_DIR=$PARENT_DIR

echo "Working in $PARENT_DIR"

# global: Dataset, log folder and conig file
DATASET="MNIST_SMALL"
LOG_FOLDER="cnn_training_template"
CONFIG_FILE_PATH="$PARENT_DIR/relu_cnn.ini"
TEST_SET_BOOL=False

echo "Training the network"

INIT_MODE="resume"
RUN_NAME="short_cnn_run"
PT_WEIGHTS_PATH="None"

python train_and_test_cnn.py $DATASET $CONFIG_FILE_PATH $INIT_MODE $PT_WEIGHTS_PATH $LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $PARENT_DIR