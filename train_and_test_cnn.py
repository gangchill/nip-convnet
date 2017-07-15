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

	## ############## ##
	# ARGUMENT PARSING # 
	## ############## ##
	arguments = sys.argv

	if 8 <= len(arguments) <= 9:
		print('-----------------------------------------------------------------------------')
		print('{} started with {} arguments, they are interpreted as:'.format(arguments[0], len(arguments)))

		# 1: Datset
		DATASET = arguments[1]
		print('Dataset         : {}'.format(DATASET))

		# 2: config file path
		use_config_file 	= True
		config_file_path 	= arguments[2] 
		print('Config path 	: {}'.format(config_file_path))

		# 3: initialization options
		initialization_mode = str(arguments[3])

		# 4: path for pre-trained weights
		init_weights_path = str(arguments[4])

		if initialization_mode == 'pre_trained_encoding':
			pre_trained_conv_weights_directory = init_weights_path
			print('Init mode 	: pre-trained')
		elif initialization_mode == 'from_folder':
			model_weights_directory = init_weights_path
			print('Init mode 	: from_folder')
		elif initialization_mode == 'resume':
			print('Init mode 	: resume')
		else:
			print('Init mode 	: random initialization (default)')

		# 5: Log folder
		log_folder_name = arguments[5]
		print('Log folder  	: {}'.format(log_folder_name))

		# 6: run name
		custom_run_name = arguments[6]
		if str(custom_run_name) == 'None':
			custom_run_name = None
			print('run name 	: default')
		else:
			print('run name 	: {}'.format(custom_run_name))

		# 7: add a test set evaluation after the training process
		# 	 flag was formerly used to enable test set evaluation during training time but this should not e done
		test_evaluation = arguments[7]

		evaluate_using_test_set = False
		evaluation_set_size 	= 1024	
		evaluation_batch_size 	= 512 

		if str(test_evaluation) == 'true' or str(test_evaluation) == 'True':
			final_test_evaluation = True
		else:
			final_test_evaluation = False

		# 8: weights and log folder prefix (optional, default is the nip-convnet root)
		if len(arguments) == 9:
			root_dir_path = arguments[8]

			if str(root_dir_path) == 'None':
				root_dir_path = None
			else:
				if not os.path.exists(root_dir_path):
					print('Given root directory {} is not valid, please enter an existing folder.'.format(root_dir_path))
					root_dir_path = None

				else:
					print('Custom root directory {} given, logs and weights will be saved in there.'.format(root_dir_path))
		else:
			root_dir_path = None

		print('-----------------------------------------------------------------------------')

	
	elif len(arguments) == 1:
		print('Using default settings from file')
		## #################### ##
		# INITIALIZATION OPTIONS #
		## #################### ##
		log_folder_name = '001_lr_decay'
		custom_run_name = 'test' # 'sigmoid_pre-trained'
		DATASET = "MNIST_SMALL"

		root_dir_path = None

		# choose whether to use the real test set or the { 	validation set, (MNIST | CK+)
		# 													training   set, (CIFAR10)     }
		# the test set should only be used for the final evaluation of a models performance
		evaluate_using_test_set = False
		final_test_evaluation 	= True
		evaluation_set_size 	= 1024 			# set negative value for the whole set or restrain set size 
		evaluation_batch_size 	= 512    		# batch size used for testing

		use_config_file 	= False

		initialization_mode = 'resume'
		# initialization_mode:
		# 'resume'						: 	resume training from latest checkpoint in weights/log_folder_name/run_name if possible, otherwise default
		# 'from_folder'					: 	load last checkpoint from folder given in 
		# 'pre_trained_encoding'		:	load encoding weights from an auto-encoder
		# 'default'						: 	init weights at random
		# ------------------------------
		# from_folder:
		model_weights_directory = 'weights/1k_MNIST_CNN/sigmoid/best' 		
		# pre_trained_encoding
		pre_trained_conv_weights_directory = 'weights/07_CAE_MNIST_SIGMOID_debug/a55_55-64_64-sigmoid_max_poolingtr128__True'
		# use_config_file
		config_file_path 	= 'configs/simple_cnn_config.ini'	

	else:
		print('Wrong number of arguments!')
		print('Usage: {} dataset config_file_path init_mode pre-trained_weights_path log_folder run_name test_set_bool'.format(arguments[0]))
		print('dataset 					: (MNIST | MNIST_SMALL | CIFAR10 | CKPLUS)')
		print('config_file_path 		: relative path to config file to use')
		print('init_mode 				: (resume | from_folder | pre_trained_encoding | default')
		print('pre-trained_weights_path : (None : resume training | relative path to pre-trained conv weights')
		print('log_folder 				: log folder name (used in logs/ and weights/ subdirectories)')
		print('run_name 				: (None : auto generated run name | custom run name')
		print('test_set_bool 			: (true (Attention: str, bash-style lower case): use test set for validation | else: use training set to monitor progress')


	## #################### ##
	# DATASET INITIALIZATION #
	## #################### ##
	if DATASET == "MNIST":
		# load mnist
		from tensorflow.examples.tutorials.mnist import input_data
		dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_size = (28, 28)
		num_classes = 10
		one_hot_labels = True
		nhwd_shape = False


	elif DATASET == "MNIST_10k":
		
		N = 10000

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


	elif DATASET == "MNIST_1k":
		
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
		input_size = (68,65)
		num_classes = load_ckplus.NUM_CLASSES
		one_hot_labels = True
		nhwd_shape = False

	elif DATASET=="CIFAR10":
		dataset 		= "cifar_10" 	# signals the train_cnn function that it needs to load the data via cifar_10_input.py
		one_hot_labels 	= False			# changes the error functions because this cifar-10 version doesn't use a one-hot encoding
		input_size 		= (24, 24, 3) 
		num_classes 	= 10
		nhwd_shape 		= True

		maybe_download_and_extract()

	elif DATASET[:6]=="CIFAR_":
		# CIFAR_5k
		limit = int(DATASET.split('_')[1].split('k')[0])

		if limit > 0 and limit < 51:
			print("Using " + str(limit) + "k CIFAR training images")

			from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
			from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
			import scripts.load_cifar as load_cifar

			complete_dataset = load_cifar.read_data_sets(one_hot=True)

			small_training_dataset = DataSet(complete_dataset.train._images[:limit*1000], complete_dataset.train._labels[:limit*1000], dtype=dtypes.uint8, reshape=False)

			dataset = Datasets(train=small_training_dataset, validation=complete_dataset.validation, test=complete_dataset.test)

			one_hot_labels = True
			input_size = (32, 32, 3)
			num_classes = 10
			nhwd_shape = True
		else:
			raise Exception("CIFAR limit must be between 1k and 50k, is: " + str(limit))

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
		hidden_channels = [32, 32] 
		pooling_type  = 'strided_conv' # dont change, std::bac_alloc otherwise (TODO: understand why)
		strides = None # other strides should not work yet
		activation_function = 'sigmoid'
		# fc-layer parameters:
		dense_depths = [384, 192]

		# TRAINING
		# training parameters:
		batch_size 		= 128
		max_iterations	= 31
		chk_iterations 	= 1
		dropout_k_p		= 0.5

		step_size 		= 0.1
		decay_steps		= 2
		decay_rate		= 0.1

		weight_decay_regularizer = 0.

		weight_init_stddev 	= 0.2
		weight_init_mean 	= 0.
		initial_bias_value 	= 0.

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
		config_dict['weight_init_stddev'] 	= weight_init_stddev
		config_dict['weight_init_mean']		= weight_init_mean
		config_dict['initial_bias_value']	= initial_bias_value
		config_dict['weight_decay_regularizer'] = weight_decay_regularizer

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
		weight_init_stddev 		= float(config_dict['weight_init_stddev'])
		weight_init_mean 		= float(config_dict['weight_init_mean'])
		initial_bias_value 		= float(config_dict['initial_bias_value'])
		weight_decay_regularizer= float(config_dict['weight_decay_regularizer'])

		print('Config succesfully loaded')

	# -------------------------------------------------------

	# construct names for logging

	architecture_str 	= '(a)'  + '_'.join(map(lambda x: str(x[0]) + str(x[1]), filter_dims)) + '-' + '_'.join(map(str, hidden_channels)) + '-' + activation_function
	training_str 		= '(tr)' + str(batch_size) + '_' + '_' + str(dropout_k_p)
	
	if custom_run_name is None:
		run_name = architecture_str + '|' + training_str
	else:
		run_name = custom_run_name



	# LOG AND WEIGHTS FOLDER:

	if root_dir_path is not None:
		log_folder_parent_dir = os.path.join(root_dir_path, 'logs')
	else:
		log_folder_parent_dir = 'logs' 
	log_path = os.path.join(log_folder_parent_dir, log_folder_name, run_name)

	# folder to store the training weights in:
	if root_dir_path is not None:
		model_save_parent_dir = os.path.join(root_dir_path, 'weights')
	else:
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

	cnn = CNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function, one_hot_labels=one_hot_labels, step_size = step_size, decay_steps = decay_steps, decay_rate = decay_rate, weight_init_stddev = weight_init_stddev, weight_init_mean = weight_init_mean, initial_bias_value = initial_bias_value, weight_decay_regularizer=weight_decay_regularizer)

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

			train_cnn(sess, cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, init_iteration,  max_iterations, chk_iterations, writer, fine_tuning_only, save_path, best_accuracy_so_far, num_test_images=evaluation_set_size, test_batch_size=evaluation_batch_size, evaluate_using_test_set=evaluate_using_test_set, final_test_evaluation=final_test_evaluation)

			initialization_finished = True

		else:
			if initialization_mode == 'resume':
				print('No checkpoint was found, beginning with iteration 0')
			else:
				print('No checkpoint found, check the given folder!')
				sys.exit(1)


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

		else:
			print('no pre-trained weights file found ,check the given folder!')
			sys.exit(1)


	if not initialization_finished:
		# always train a new autoencoder 
		train_cnn(sess, cnn, dataset, x, y_, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_path, num_test_images=evaluation_set_size, test_batch_size=evaluation_batch_size, evaluate_using_test_set=evaluate_using_test_set, final_test_evaluation=final_test_evaluation)


	# TODO Sabbir: store the current config in a config file in the logs/log_folder_name/run_name folder 

	writer.close()
	sess.close()



if __name__ == '__main__':
	main()
