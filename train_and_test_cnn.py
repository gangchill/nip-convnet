# ----------------------------------------------------
# train and test a convolutional neural network
# ----------------------------------------------------

from tensorflow.python.framework import dtypes
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import tarfile
from six.moves import urllib

import matplotlib.pyplot as plt

# import the  convolutional neural network class
from models.cnn.cnn import CNN

from scripts.train_cnn 				import train_cnn
from scripts.from_github.cifar10 	import maybe_download_and_extract

import configs.config as cfg

from tensorflow.python.framework import dtypes

########
# MAIN #
########

def main():

	## #################### ##
	# INITIALIZATION OPTIONS #
	## #################### ##
	log_folder_name = '1k_MNIST'
	custom_run_name = 'pre-trained'
	DATASET = "MNIST_SMALL"

	initialization_mode = 'resume'
	use_config_file 	= False

	# initialization_mode:
	# 'resume'						: 	resume training from latest checkpoint in weights/log_folder_name/run_name if possible, otherwise default
	# 'from_folder'					: 	load last checkpoint from folder given in 
	# 'pre_trained_encoding'		:	load encoding weights from an auto-encoder
	# 'default'						: 	init weights at random
	
	# paths
	model_weights_directory = 'weights/55_CNN_CIFAR/a55_55-64_64-relutr128__0.5/best' 		# from_folder
	pre_trained_conv_weights_directory = 'weights/67_CAE_MNIST/a55_55-64_64-relu_max_poolingtr128__True/best'# pre_trained_encoding
	
	config_file_path 	= 'configs/simple_cnn_config.ini'							# use_config_file

	if DATASET == "MNIST":
		# load mnist
		from tensorflow.examples.tutorials.mnist import input_data
		dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_size = (28, 28)
		num_classes = 10
		one_hot_labels = True
		nhwd_shape = False

	elif DATASET == "MNIST_SMALL":
		N = 1000

		# load mnist
		from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
		from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
		from tensorflow.examples.tutorials.mnist import input_data

		complete_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True,)

		small_training_dataset = DataSet(complete_dataset.train._images[:N], complete_dataset.train._labels[:N], dtype=dtypes.uint8, reshape=False)

		dataset = Datasets(train= small_training_dataset, validation = complete_dataset.validation, test=complete_dataset.test)

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


	keep_prob = tf.placeholder(tf.float32)

	

	## #### ##
	# CONFIG # 
	## #### ##

	config_loader = cfg.ConfigLoader()

	if not use_config_file:

		# ARCHITECTURE
		# feature extraction parameters
		filter_dims 	= [(5,5), (5,5)]
		hidden_channels = [64, 64] 
		pooling_type  = 'strided_conv' # dont change, std::bac_alloc otherwise (TODO: understand why)
		strides = None # other strides should not work yet
		activation_function = 'relu'
		# fc-layer parameters:
		dense_depths = [384, 192]

		# TRAINING
		# training parameters:
		batch_size 		= 128
		max_iterations	= 10001
		chk_iterations 	= 1000
		dropout_k_p		= 0.5

		step_size 		= 0.1
		decay_steps		= 10000
		decay_rate		= 0.1

		# only optimize dense layers and leave convolutions as they are
		fine_tuning_only = False

		# store to config dict:
		config_dict = {}
		config_dict['filter_dims'] 			= filter_dims
		config_dict['hidden_channels'] 		= hidden_channels
		config_dict['pooling_type']  		= pooling_type
		config_dict['strides'] 				= strides
		config_dict['activation_function'] 	= activation_function
		config_dict['dense_depths'] 		= dense_depths
		config_dict['batch_size'] 			= batch_size
		config_dict['max_iterations'] 		= max_iterations
		config_dict['chk_iterations'] 		= chk_iterations
		config_dict['dropout_k_p'] 			= dropout_k_p 
		config_dict['fine_tuning_only'] 	= int(fine_tuning_only)
		config_dict['step_size'] 			= step_size
		config_dict['decay_steps']			= decay_steps
		config_dict['decay_rate']			= decay_rate

		config_loader.configuration_dict = config_dict

	else:
		# load config from file 
		print('Loading config from file {}'.format(config_file_path))
		config_loader.load_config_file(config_file_path, 'CNN')
		config_dict = config_loader.configuration_dict

		if config_dict is None:
			print('Loading not succesful')
			sys.exit()

		# init all config variables variables from the file
		filter_dims 			= config_dict['filter_dims']
		hidden_channels 		= config_dict['hidden_channels'] 
		pooling_type  			= config_dict['pooling_type'] 
		strides 				= config_dict['strides']
		activation_function 	= config_dict['activation_function']
		dense_depths 			= config_dict['dense_depths']
		batch_size 				= int(config_dict['batch_size'])
		max_iterations			= int(config_dict['max_iterations'])
		chk_iterations 			= int(config_dict['chk_iterations'])
		dropout_k_p				= float(config_dict['dropout_k_p']) 
		fine_tuning_only 		= bool(int(config_dict['fine_tuning_only']))
		step_size				= float(config_dict['step_size'])
		decay_steps				= int(config_dict['decay_steps'])
		decay_rate				= float(config_dict['decay_rate'])

		print('Config succesfully loaded')

	# -------------------------------------------------------

	# construct names for logging

	architecture_str 	= 'a'  + '_'.join(map(lambda x: str(x[0]) + str(x[1]), filter_dims)) + '-' + '_'.join(map(str, hidden_channels)) + '-' + activation_function
	training_str 		= 'tr' + str(batch_size) + '_' + '_' + str(dropout_k_p)
	
	if custom_run_name is None:
		run_name = architecture_str + training_str
	else:
		run_name = custom_run_name

	log_path = os.path.join('logs', log_folder_name, run_name)

	# folder to store the training weights in:
	model_save_parent_dir = 'weights'
	save_path = os.path.join(model_save_parent_dir, log_folder_name, run_name)
	check_dirs = [model_save_parent_dir, os.path.join(model_save_parent_dir, log_folder_name), os.path.join(model_save_parent_dir, log_folder_name), os.path.join(model_save_parent_dir, log_folder_name, run_name), os.path.join(model_save_parent_dir, log_folder_name, run_name, 'best')]
	
	for directory in check_dirs:
		if not os.path.exists(directory):
			os.makedirs(directory)


	## ###### ##
	# TRAINING #
	## ###### ##

	init_iteration = 0

	cnn = CNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function, one_hot_labels=one_hot_labels, step_size = step_size, decay_steps = decay_steps, decay_rate = decay_rate)

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	# add logwriter for tensorboard
	writer = tf.summary.FileWriter(log_path, sess.graph)

	# store config file in the folder
	config_loader.store_config_file(os.path.join(log_path, 'config.ini'), 'CNN')

	initialization_finished = False

	if initialization_mode == 'resume' or initialization_mode == 'from_folder':
		# initialize training with weights from a previous training 

		cwd = os.getcwd()

		if initialization_mode == 'from_folder':
			chkpnt_file_path = os.path.join(cwd, model_weights_directory)
		else:
			chkpnt_file_path = os.path.join(cwd, save_path)

		saver = tf.train.Saver(cnn.all_variables_dict)
		latest_checkpoint = tf.train.latest_checkpoint(chkpnt_file_path)

		print(latest_checkpoint)

		if latest_checkpoint is not None:

			print('Found checkpoint')

			init_iteration = int(latest_checkpoint.split('-')[-1]) + 1

			best_accuracy_so_far = float(latest_checkpoint.split('-')[-2])

			print('iteration is: {}'.format(init_iteration))
			print('accuracy is: {}'.format(best_accuracy_so_far))

			if initialization_mode == 'from_folder':
				print('retrieved weights from checkpoint, begin with new iteration 0')
				init_iteration = 0

			saver.restore(sess, latest_checkpoint)

			train_cnn(sess, cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, init_iteration,  max_iterations, chk_iterations, writer, fine_tuning_only, save_path, best_accuracy_so_far)

			initialization_finished = True

		else:
			print('No checkpoint was found, beginning with iteration 0')


	elif initialization_mode == 'pre_trained_encoding':

		if pre_trained_conv_weights_directory is not None:

			print('Trying to load conv weights from file')

			cwd = os.getcwd()
			chkpnt_file_path = os.path.join(cwd, pre_trained_conv_weights_directory)

			print('Looking for checkpoint in {}'.format(chkpnt_file_path))

			saver = tf.train.Saver()
			latest_checkpoint = tf.train.latest_checkpoint(chkpnt_file_path)

			print('Latest checkpoint is: {}'.format(latest_checkpoint))

			if latest_checkpoint is not None:

				cnn.load_encoding_weights(sess, latest_checkpoint)

				print('Initialized the CNN with encoding weights found in {}'.format(latest_checkpoint))


	if not initialization_finished:
		# always train a new autoencoder 
		train_cnn(sess, cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_path)


	# TODO Sabbir: store the current config in a config file in the logs/log_folder_name/run_name folder 

	writer.close()
	sess.close()



if __name__ == '__main__':
	main()
