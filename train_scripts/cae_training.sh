#!/bin/bash          
echo '## ########## ##'
echo '# CAE TRAINING #'    
echo '## ########## ##'

DATASET="MNIST"
CONFIG_FILE_PATH="configs/CAE/cae_2l_sigmoid.ini"
PT_WEIGHTS_PATH="None"
LOG_FOLDER="14_CAE_regularization"
REGULARIZATION_FACTOR="0.000001"

RUN_NAME="sigmoid_bigger_lr$REGULARIZATION_FACTOR"

python train_and_test_cae.py $DATASET $CONFIG_FILE_PATH $PT_WEIGHTS_PATH $LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $REGULARIZATION_FACTOR