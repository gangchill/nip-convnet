#!/bin/bash          
echo '## ################################### ##'
echo '# DENSE LAYER OPTIMIZATION SANITY CHECK #'
echo '## ################################### ##'

echo '\nScript to compare the performance of pre-trained filters. Only the CNNs dense layers are optimized and compared to a reference dense layer optimization with random filters'

# global: Dataset, log folder and conig file
DATASET="MNIST_SMALL"
LOG_FOLDER="84_filter_sanity_check_lower_lr_decay"
CONFIG_FILE_PATH="configs/CNN/cnn_tanh_dense_layer_opt.ini"
TEST_SET_BOOL=False


echo '\n----------------'
echo 'Pre-trained Net:'
echo '----------------'

INIT_MODE="pre_trained_encoding"
RUN_NAME="(DO)_pt_encoding"
PT_WEIGHTS_PATH="weights/16_CAE_re/scaled_tanh_0.5-0.00000001"

python train_and_test_cnn.py $DATASET $CONFIG_FILE_PATH $INIT_MODE $PT_WEIGHTS_PATH $LOG_FOLDER $RUN_NAME $TEST_SET_BOOL


echo '\n--------------'
echo 'Reference Net:'
echo '--------------'

INIT_MODE="default"
WEIGHTS_PATH="None" 
REFERENCE_RUN_NAME="(DO)_reference_random_init"

python train_and_test_cnn.py $DATASET $CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $LOG_FOLDER $REFERENCE_RUN_NAME $TEST_SET_BOOL

