import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# import the autoencoder and cnn class
from models.cnn.cnn import CNN
from models.cae.convolutional_autoencoder import CAE

# import the training procedures
from scripts.train_cae import train_ae
from scripts.train_cnn import train_cnn

from train_and_test_cae import get_weight_file_name

def main():

	cae_dir 		= os.path.join('models', 'cae')
	cae_weights_dir	= os.path.join(cae_dir, 'weights')

	DATASET = "CKPLUS"

	if DATASET == "MNIST":
		# load mnist
		from tensorflow.examples.tutorials.mnist import input_data
		dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_size = (28, 28)
		num_classes = 10
		one_hot_labels = True
		nhwd_shape = False

	elif DATASET == "CKPLUS":
		import scripts.load_ckplus as load_ckplus
		dataset = load_ckplus.read_data_sets(one_hot=True)
		input_size = (49,64)
		num_classes = load_ckplus.NUM_CLASSES
		one_hot_labels = True
		nhwd_shape = False

	elif DATASET=="CIFAR10":
		dataset 		= "cifar_10" 	# signals the train_cnn function that it needs to load the data via cifar_10_input.py
		one_hot_labels 	= False			# changes the error functions because this cifar-10 version doesn't use a one-hot encoding
		input_size 		= (24, 24, 3)
		num_classes 	= 1
		nhwd_shape 		= True

		maybe_download_and_extract()

	if nhwd_shape == False:

		# input variables: x (images), y_ (labels), keep_prob (dropout rate)
		x  = tf.placeholder(tf.float32, [None, input_size[0]*input_size[1]], name='input_digits')
		# reshape the input to NHWD format
		x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 1])

	else:

		x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]], name='input_images')
		x_image = x

	if one_hot_labels:
		y_ = tf.placeholder(tf.float32, [None, num_classes], name='target_labels')
	else:
		y_ = tf.placeholder(tf.int64,   [None], name='target_labels')


	keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')



	## ##################### ##
	# ARCHITECTURE PARAMETERS #
	## ##################### ##

	# feature extraction (both CAE and CNN)
	filter_dims 	= [(5,5)]
	hidden_channels = [16]
	pooling_type  = 'strided_conv' # dont change, std::bac_alloc otherwise (TODO: understand why)
	strides = None # other strides should not work yet
	activation_function = 'lrelu'
	relu_leak = 0.2 # only for leaky relus

	error_function = 'cross-entropy' # default is cross-entropy

	# CAE only:
	tie_conv_weights = True

	# CNN only:
	dense_depths = []

	## ################# ##
	# TRAINING PARAMETERS #
	## ################# ##

	weight_init_mean 	= 0.
	weight_init_stddev 	= 1.
	initial_bias_value  = 0.0000000

	# currently, the same parameters are used for the training of the cae and the cnn
	batch_size 		= 100
	max_iterations	= 2001
	chk_iterations 	= 300
	dropout_k_p		= 0.5
	step_size 		= 0.000001

	step_size_cnn   = 0.01

	weight_file_name = get_weight_file_name(filter_dims, hidden_channels, pooling_type, activation_function, tie_conv_weights, batch_size, step_size, weight_init_mean, weight_init_stddev, initial_bias_value)

	log_folder_name = 'weight_pass_demo'
	run_name 	= '{}'.format(weight_file_name)
	# run_name = ''


	# folder to store the training weights in:
	model_save_parent_dir = 'weights'
	save_path = os.path.join(model_save_parent_dir, log_folder_name, run_name)
	check_dirs = [model_save_parent_dir, os.path.join(model_save_parent_dir, log_folder_name), os.path.join(model_save_parent_dir, log_folder_name), os.path.join(model_save_parent_dir, log_folder_name, run_name)]

	for directory in check_dirs:
		if not os.path.exists(directory):
			os.makedirs(directory)

	# only optimize dense layers of the cnn
	fine_tuning_only = False

	# log names
	log_folder_name = 'weight_passing_demo'
	architecture_str 	= 'a'  + '_'.join(map(lambda x: str(x[0]) + str(x[1]), filter_dims)) + '-' + '_'.join(map(str, hidden_channels)) + '-' + activation_function
	training_str 		= 'tr' + str(batch_size) + '_' + str(max_iterations) + '_' + str(dropout_k_p)
	run_prefix 			= 'mnist_wp_' + architecture_str + training_str
	log_path = os.path.join('logs', log_folder_name, run_prefix)

	encoding_weights_path = log_path + 'wp'

	weight_file_name_cae = architecture_str + training_str + '_cae'
	weight_file_name_cnn = architecture_str + training_str + '_cnn'


	## ######### ##
	# INIT MODELS # 
	## ######### ##

	autoencoder = CAE(x_image, filter_dims, hidden_channels, step_size, weight_init_stddev, weight_init_mean, initial_bias_value, strides, pooling_type, activation_function, tie_conv_weights, store_model_walkthrough = True, relu_leak = relu_leak)
	cnn = CNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function, one_hot_labels=one_hot_labels, scope_name='pre_trained_CNN', step_size = step_size_cnn)

	# second cnn with the same structure that will be trained independently from the autoencoder
	comparison_cnn = CNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function, one_hot_labels=one_hot_labels, scope_name='reference_CNN')

	## ###### ##
	# TRAINING #
	## ###### ##

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# add logwriter for tensorboard
	writer = tf.summary.FileWriter(log_path, sess.graph)

	# TODO
	init_iteration = 0
	best_accuracy_so_far = 0.

	# train the autoencoder
	train_ae(sess, writer, x, autoencoder, dataset, cae_dir, cae_weights_dir, weight_file_name, error_function, batch_size, init_iteration, max_iterations, chk_iterations, save_prefix = save_path)
	print('...finished training the cae')

	# save autoencoder weights to file
	autoencoder.store_encoding_weights(sess, encoding_weights_path)
	print('...saved cae encoding weights to file')

	# load the same weights into the cnn
	cnn.load_encoding_weights(sess, encoding_weights_path)
	print('...loaded the cae weights into the cnn')

	# train the cnn
	train_cnn(sess, cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_path, best_accuracy_so_far)
	print('...finished training the cnn')

	train_cnn(sess, comparison_cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_path, best_accuracy_so_far)
	print('...finished training comparison cnn')

	# train the comparison cnn



	writer.close()
	sess.close()


if __name__ == '__main__':
	main()