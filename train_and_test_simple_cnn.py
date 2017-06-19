# ----------------------------------------------------
# train and test a simple convolutional neural network
# ----------------------------------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# import the  SCNN class from simple_.py
from models.cnn.simple_cnn import SCNN
from scripts.train_cnn import train_cnn

########
# MAIN #
########

def main():

	DATASET = "MNIST"

	if DATASET == "MNIST":
		# load mnist
		from tensorflow.examples.tutorials.mnist import input_data
		dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_size = (28, 28)
		num_classes = 10

	elif DATASET == "CKPLUS":
		import scripts.load_ckplus as load_ckplus
		dataset = load_ckplus.read_data_sets(one_hot=True)
		input_size = load_ckplus.INPUT_SIZE
		num_classes = load_ckplus.NUM_CLASSES

	elif DATASET == "CIFAR10":
		import scripts.cifar_10_input as cifar_10_input



	# input variables: x (images), y_ (labels), keep_prob (dropout rate)
	x  = tf.placeholder(tf.float32, [None, input_size[0]*input_size[1]], name='input_digits')

	y_ = tf.placeholder(tf.float32, [None, num_classes], name='target_labels')

	keep_prob = tf.placeholder(tf.float32)

	# reshape the input to NHWD format
	x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 1])


	## #### ##
	# CONFIG # 
	## #### ##

	use_config_file 	= False
	config_file_path 	= 'configs/config_example.jeneendung'

	# ------------------------------------------------------

	# ARCHITECTURE
	# feature extraction parameters
	filter_dims 	= [(5,5), (5,5)]
	hidden_channels = [16, 32] 
	pooling_type  = 'strided_conv' # dont change, std::bac_alloc otherwise (TODO: understand why)
	strides = None # other strides should not work yet
	activation_function = 'sigmoid'

	# fc-layer parameters:
	dense_depths = []

	# TRAINING
	# training parameters:
	batch_size 		= 100
	max_iterations	= 5001
	chk_iterations 	= 500
	dropout_k_p		= 0.5

	# only optimize dense layers and leave convolutions as they are
	fine_tuning_only = False

	if use_config_file:
		# load config file to class 
		# config class = ... 

		# batchsize = configclas.. 

	else:
		# move manual config stuff here
		pass

	# -------------------------------------------------------

	# construct names for logging
	log_folder_name = 'cnn_reference_try'
	log_path = os.path.join('logs', log_folder_name, run_prefix)

	architecture_str 	= 'a'  + '_'.join(map(lambda x: str(x[0]) + str(x[1]), filter_dims)) + '-' + '_'.join(map(str, hidden_channels)) + '-' + activation_function
	training_str 		= 'tr' + str(batch_size) + '_' + str(max_iterations) + '_' + str(dropout_k_p)
	run_prefix 			= 'mnist_cnn_' + architecture_str + training_str


	# RUN

	cnn = SCNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function)

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	# add logwriter for tensorboard
	writer = tf.summary.FileWriter(log_path, sess.graph)

	train_cnn(sess, cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations, writer, fine_tuning_only)


	writer.close()
	sess.close()



if __name__ == '__main__':
	main()