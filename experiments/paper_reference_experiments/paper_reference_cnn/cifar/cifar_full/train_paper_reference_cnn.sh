#!/bin/bash          
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXX TRAIN REFERENCE CNN XXXXXXXXXXXXXXXX'
echo 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# current folder:
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_ROOT_DIR=$PARENT_DIR

echo "Working in $PARENT_DIR"

# the script train RE_RUN_NUMBER reference nets and PRE_RUN_NUMBER pre-trained networks for statistics
REF_RUN_NUM=10
PRE_RUN_NUM=10

DATASET="CIFAR"
CNN_LOG_FOLDER="006_cifar_full_rr"
CNN_CONFIG_FILE_PATH="$PARENT_DIR/paper_reference_cnn.ini"
# CNN_CONFIG_FILE_PATH="experiments/experimental_templates/cnn_in_folder_demo/relu_cnn.ini"

# if set to true, a test run using the test set is performed after the training
TEST_SET_BOOL=True

echo "Begin training the reference network"

INIT_MODE="default"
WEIGHTS_PATH="None"

RUN_PREFIX="reference-random_init"

for (( i=1; i<=${REF_RUN_NUM}; i++ ))
do

	echo "Begin training reference network $i"
	RUN_NAME="${RUN_PREFIX}_${i}"  
	
	echo "Run name is: $RUN_NAME"	
	python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $PARENT_DIR 


done


echo "Begin training the pre-trained network"

INIT_MODE="pre_trained_encoding"
WEIGHTS_PATH="experiments/paper_reference_experiments/paper_reference_cae_cifar/weights/CIFAR_reference_cae/cifar_test_lr5.0"
RUN_PREFIX="pre-trained"

for (( i=1; i<=${PRE_RUN_NUM}; i++ ))
do
	echo "Begin training pre-trained network  $i"
	RUN_NAME="${RUN_PREFIX}_${i}"  

	python train_and_test_cnn.py $DATASET $CNN_CONFIG_FILE_PATH $INIT_MODE $WEIGHTS_PATH $CNN_LOG_FOLDER $RUN_NAME $TEST_SET_BOOL $PARENT_DIR

done
