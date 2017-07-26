# --------------------------------------------------------------------------------------
# train and test a convolutional autoencoder with one hidden layer for the MNIST dataset
# --------------------------------------------------------------------------------------

import tensorflow as tf 
from tensorflow.python.framework import dtypes

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import os, sys
from functools import reduce


# import the simple autoencoder class from SAE.py
from models.cae.convolutional_autoencoder import CAE
from scripts.train_cae import train_ae
import configs.config as cfg
from scripts.from_github.cifar10 	import maybe_download_and_extract

########
# MAIN #
########

def main():

	## ############## ##
	# ARGUMENT PARSING #
	## ############## ##

	arguments = sys.argv
	
	print('len(arguments) = {}'.format(len(arguments)))
	print(arguments)

	if 7 <= len(arguments) <= 8:
		print('-----------------------------------------------------------------------------')
		print('{} started with {} arguments, they are interpreted as:'.format(arguments[0], len(arguments)))

		# 1: Datset
		DATASET = arguments[1]
		print('Dataset         : {}'.format(DATASET))

		# 2: config file path
		use_config_file 	= True
		config_file_path 	= arguments[2] 
		print('Config path 	: {}'.format(config_file_path))

		# 3: weight initialization
		init_weigts_path = str(arguments[3])
		if init_weigts_path == 'None':
			initialization_mode = 'resume'
			print('Init mode 	: resume')
		else:
			initialization_mode = 'from_folder'
			model_weights_dir = init_weigts_path
			print('Init mode 	: {}'.format(initialization_mode))

		# 4: Log folder
		log_folder_name = arguments[4]
		print('Log folder  	: {}'.format(log_folder_name))

		# 5: run name
		custom_run_name = arguments[5]
		if str(custom_run_name) == 'None':
			custom_run_name = None
			print('run name 	: default')
		else:
			print('run name 	: {}'.format(custom_run_name))

		# 6: regularization factor
		regularization_factor = float(arguments[6])

		print('-----------------------------------------------------------------------------')

		visualize_model_walkthrough = True

		# 7: weights and log folder prefix (default is the nip-convnet root)
		if len(arguments) == 8:
			root_dir_path = arguments[7]

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

	
	elif len(arguments) == 1:
		print('Using default settings from file')

		DATASET = "MNIST"

		# load architecture / training configurations from file
		use_config_file 	= False
		config_file_path 	= 'configs/cae_2l_sigmoid.ini'

		# important: these values need to be restored from file directly here, which means if a config file is specified, it needs to be loaded already here

		# restore weights from the last iteration (if the same training setup was used before)
		# restore_last_checkpoint = True
		initialization_mode = 'resume' # resume - default - from_file
		model_weights_dir = 'test'

		# store model walkthrough (no CIFAR support yet)
		visualize_model_walkthrough = True

		log_folder_name = '07_CAE_MNIST_SIGMOID_debug'
		custom_run_name = None

		regularization_factor = 0.001

		root_dir_path = None

	else:
		print('Wrong number of arguments!')
		print('Usage: {} dataset config_file_path pre-trained_weights_path log_folder run_name regularization_factor'.format(arguments[0]))
		print('dataset 					: (MNIST | MNIST_SMALL | CIFAR10 | CKPLUS)')
		print('config_file_path 		: relative path to config file to use')
		print('init_weights_path 	 	: (None : resume training | path to old checkpoint to init from')
		print('log_folder 				: log folder name (used in logs/ and weights/ subdirectories)')
		print('run_name 				: (None : auto generated run name | custom run name')
		print('regularization_factor	: (<= 0: do nothing | > 0: factor for L1 regularization of the hidden representation') 
		print('-----------------------------------------------------------------------------')

		sys.exit(1)

	## ########### ##
	# INPUT HANDING #
	## ########### ##

	if DATASET == "MNIST":
		# load mnist
		from tensorflow.examples.tutorials.mnist import input_data
		dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_size = (28, 28)
		num_classes = 10
		nhwd_shape = False

	elif DATASET == "MNIST_SMALL":
		N = 1000

		# load mnist
		from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
		from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
		from tensorflow.examples.tutorials.mnist import input_data

		complete_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)

		small_training_dataset = DataSet(complete_dataset.train._images[:N], complete_dataset.train._labels[:N], reshape=False)

		dataset = Datasets(train=small_training_dataset, validation = complete_dataset.validation, test=complete_dataset.test)

		input_size = (28, 28)
		num_classes = 10
		one_hot_labels = True
		nhwd_shape = False

	elif DATASET == "CKPLUS":
		import scripts.load_ckplus as load_ckplus
		dataset = load_ckplus.read_data_sets(split=False, one_hot=True, frames=100)
		input_size = (68, 65)
		num_classes = load_ckplus.NUM_CLASSES
		nhwd_shape = False

	elif DATASET=="CIFAR10":
		dataset 		= "cifar_10" 	# signals the train_ae function that it needs to load the data via cifar10_input.py
		input_size 		= (24, 24, 3)
		num_classes 	= 1
		nhwd_shape 		= True

		maybe_download_and_extract()

	elif DATASET[:6]=="CIFAR_":
		# CIFAR_nk
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

	else:
		print('ERROR: Dataset not available')

	if nhwd_shape == False:

		# input variables: x (images), y_ (labels), keep_prob (dropout rate)
		x  = tf.placeholder(tf.float32, [None, input_size[0]*input_size[1]], name='input_digits')
		# reshape the input to NHWD format
		x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 1])

	else: 

		x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]], name='input_images')
		x_image = x


	# directory containing the autoencoder file
	cae_dir 		= os.path.join('models', 'cae')


	## #### ##
	# CONFIG # 
	## #### ##

	# TODO Sabbir: begin what needs to be in the config file -----------------------------

	config_loader = cfg.ConfigLoader()

	if not use_config_file:

		# -------------------------------------------------------
		# AUTOENCODER SPECIFICATIONS
		filter_dims 	= [(5,5), (5,5)]
		hidden_channels = [64, 64]
		pooling_type 	= 'max_pooling'
		strides = None # other strides should not work yet
		activation_function = 'sigmoid'
		relu_leak = 0.2 # only for leaky relus

		error_function 	= 'mse' 					# default is cross-entropy
		optimizer_type 	= 'gradient_descent' 		# default is gradient descent

		output_reconstruction_activation = 'sigmoid'

		weight_init_mean 	= 0.001
		weight_init_stddev 	= 0.05
		initial_bias_value  = 0.001

		batch_size 		= 128
		max_iterations 	= 100001
		chk_iterations  = 500
		step_size 		= 0.1

		tie_conv_weights = True

		# store to config dict:
		config_dict = {}
		config_dict['filter_dims'] = filter_dims
		config_dict['hidden_channels'] = hidden_channels
		config_dict['pooling_type'] = pooling_type
		config_dict['strides'] = strides
		config_dict['activation_function'] = activation_function
		config_dict['relu_leak'] = relu_leak
		config_dict['error_function'] = error_function
		config_dict['optimizer_type'] = optimizer_type
		config_dict['output_reconstruction_activation'] = output_reconstruction_activation
		config_dict['weight_init_mean'] = weight_init_mean
		config_dict['weight_init_stddev'] = weight_init_stddev
		config_dict['initial_bias_value'] = initial_bias_value
		config_dict['batch_size'] = batch_size
		config_dict['max_iterations'] = max_iterations
		config_dict['chk_iterations'] = chk_iterations
		config_dict['step_size'] = step_size
		config_dict['tie_conv_weights'] = int(tie_conv_weights)

		config_loader.configuration_dict = config_dict

	else:
		# load config from file
		print('Loading config from file {}'.format(config_file_path))
		config_loader.load_config_file(config_file_path, 'CAE')
		config_dict = config_loader.configuration_dict

		if config_dict is None:
			print('Loading not succesful')
			sys.exit()

		# init all config variables variables from the file
		filter_dims = config_dict['filter_dims']
		hidden_channels = config_dict['hidden_channels']
		pooling_type = config_dict['pooling_type']
		strides = config_dict['strides']
		activation_function = config_dict['activation_function']
		relu_leak = float(config_dict['relu_leak'])
		error_function = config_dict['error_function']
		optimizer_type = config_dict['optimizer_type']
		output_reconstruction_activation = config_dict['output_reconstruction_activation']
		weight_init_mean = float(config_dict['weight_init_mean'])
		weight_init_stddev = float(config_dict['weight_init_stddev'])
		initial_bias_value = float(config_dict['initial_bias_value'])
		batch_size = int(config_dict['batch_size'])
		max_iterations = int(config_dict['max_iterations'])
		chk_iterations = int(config_dict['chk_iterations'])
		step_size = float(config_dict['step_size'])
		tie_conv_weights = bool(int(config_dict['tie_conv_weights']))

		print('Config succesfully loaded')

	# TODO Sabbir: end what needs to be in the config file -----------------------------

	weight_file_name = get_weight_file_name(filter_dims, hidden_channels, pooling_type, activation_function, tie_conv_weights, batch_size, step_size, weight_init_mean, weight_init_stddev, initial_bias_value)

	# log_folder_name = '02_CIFAR_2enc'
	# run_name = 'old_commit_style'
	# run_name 	= '{}'.format(weight_file_name)
	# run_name = '({}x{})|_{}_{}_{}|{}_{}'.format(activation_function, output_reconstruction_activation, weight_init_mean, weight_init_stddev, initial_bias_value)
	# run_name = '5x5_d2_smaller_init_{}_{}'.format(activation_function, output_reconstruction_activation)
	# run_name = '{}_{}({}|{})'.format(pooling_type, output_reconstruction_activation, '-'.join(map(str, hidden_channels)), step_size)
	# run_name = '{}_{}_{}_{}_{}({})'.format(DATASET, error_function, activation_function, output_reconstruction_activation,pooling_type, weight_init_mean)
	# run_name = 'relu_small_learning_rate_101_{}'.format(weight_file_name)
	# run_name = 'that_run_tho'
	# -------------------------------------------------------

	# construct names for logging

	architecture_str 	= 'a'  + '_'.join(map(lambda x: str(x[0]) + str(x[1]), filter_dims)) + '-' + '_'.join(map(str, hidden_channels)) + '-' + activation_function + '_' + pooling_type
	training_str 		= 'tr' + str(batch_size) + '_' + '_' + str(tie_conv_weights)

	if custom_run_name is None:
		run_name = architecture_str + training_str
	else:
		run_name = custom_run_name

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


	# CHECK FOLDER STRUCTURE
	save_path = os.path.join(model_save_parent_dir, log_folder_name, run_name)
	check_dirs = [	\
					# weights directories
					model_save_parent_dir, \
					os.path.join(model_save_parent_dir, log_folder_name), \
					os.path.join(model_save_parent_dir, log_folder_name), \
					os.path.join(model_save_parent_dir, log_folder_name, run_name), \
					os.path.join(model_save_parent_dir, log_folder_name, run_name, 'best'), \
					# log directories
					log_folder_parent_dir, \
					os.path.join(log_folder_parent_dir, log_folder_name), \
					log_path \
					]
	
	for directory in check_dirs:
		if not os.path.exists(directory):
			os.makedirs(directory)


	## ###### ##
	# TRAINING #
	## ###### ##


	# construct autoencoder (5x5 filters, 3 feature maps)
	autoencoder = CAE(x_image, filter_dims, hidden_channels, step_size, weight_init_stddev, weight_init_mean, initial_bias_value, strides, pooling_type, activation_function, tie_conv_weights, store_model_walkthrough = visualize_model_walkthrough, relu_leak = relu_leak, optimizer_type = optimizer_type, output_reconstruction_activation=output_reconstruction_activation, regularization_factor=regularization_factor)

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	print("Begin autencoder training")
	
	writer = tf.summary.FileWriter(log_path, sess.graph)

	# store config file in the folder
	config_loader.store_config_file(os.path.join(log_path, 'config.ini'), 'CAE')

	init_iteration = 0

	if initialization_mode == 'resume' or initialization_mode == 'from_folder':
		# initialize training with weights from a previous training 

		cwd = os.getcwd()

		if initialization_mode == 'resume':
			chkpnt_file_path = save_path
		else: 
			chkpnt_file_path = model_weights_dir
		

		print('Looking for checkpoint')

		saver = tf.train.Saver(autoencoder.all_variables_dict)
		latest_checkpoint = tf.train.latest_checkpoint(chkpnt_file_path)

		if latest_checkpoint is not None:

			print('Found checkpoint')

			init_iteration 					= int(latest_checkpoint.split('-')[-1]) + 1
			smallest_reconstruction_error  	= float(latest_checkpoint.split('-')[-2])

			print('iteration is: {}'.format(init_iteration))
			print('smallest reconstruction error so far was {}'.format(smallest_reconstruction_error))

			if initialization_mode == 'from_folder':
				print('retrieved weights from checkpoint, begin with new iteration 0')
				init_iteration = 0

			saver.restore(sess, latest_checkpoint)

			train_ae(sess, writer, x, autoencoder, dataset, cae_dir, weight_file_name, error_function, batch_size, init_iteration, max_iterations, chk_iterations, save_prefix = save_path, minimal_reconstruction_error = smallest_reconstruction_error)

		else:
			print('No checkpoint was found, beginning with iteration 0')
			train_ae(sess, writer, x, autoencoder, dataset, cae_dir, weight_file_name, error_function, batch_size,init_iteration,  max_iterations, chk_iterations, save_prefix = save_path)


	else:
		# always train a new autoencoder 
		train_ae(sess, writer, x, autoencoder, dataset, cae_dir, weight_file_name, error_function, batch_size,init_iteration,  max_iterations, chk_iterations, save_prefix = save_path)

	# print('Test the training:')

	# visualize_cae_filters(sess, autoencoder)
	if visualize_model_walkthrough:
		visualize_ae_representation(sess, x_image, autoencoder, dataset, log_folder_name, run_name, input_size, 5)


	# add logwriter for tensorboard
	writer.close()

	sess.close()

def get_weight_file_name(filter_dims, hidden_channels, pooling_type, activation_function, tie_conv_weights, batch_size, step_size, weight_init_mean, weight_init_stddev, initial_bias_value):
	# define unique file name for architecture + training combination

	# architecture:
	filter_dims_identifier 		= reduce(lambda x,y: '{}|{}'.format(x,y), map(lambda xy: '{},{}'.format(xy[0],xy[1]), filter_dims))
	hidden_channels_identifier 	= reduce(lambda x,y: '{}|{}'.format(x,y), hidden_channels)
	
	mp_identifier = pooling_type

	if tie_conv_weights:
		tying_str = '_TW'
	else:
		tying_str = ''

	architecture_identifier = '({}-{}{}-{}{})'.format(filter_dims_identifier, hidden_channels_identifier, mp_identifier, activation_function, tying_str)

	# training:
	training_identifier = '({},{},{}, {}, {})'.format(batch_size, step_size, weight_init_mean, weight_init_stddev, initial_bias_value)

	return '{}-{}'.format(architecture_identifier, training_identifier)


def visualize_cae_filters(sess, autoencoder): 

	folders = ['filters']
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	print('save the filters to file:')

	with sess.as_default():
		cae_filters = autoencoder.conv_weights.eval()

	cae_filters = cae_filters[0]

	num_filters = cae_filters.shape[3]

	fig = plt.figure(figsize=(num_filters * 10, 10))

	fntsz=30

	plt.suptitle('Filter visualizations convolutional autoencoder', fontsize=fntsz)

	for i in range(num_filters):
		plt.subplot(1, num_filters, i+1)
		plt.imshow(cae_filters[:,:,0,i], interpolation='none', cmap='gray_r')
		plt.axis('off')

	plt.savefig(os.path.join('filters', 'filter_example.png'))
	plt.close(fig)


def visualize_ae_representation(sess, input_placeholder, autoencoder, mnist, folder_name, run_name, input_size, num_images = 100, use_training_set = False, common_scaling = False, plot_first_layer_filters = False, max_maps_per_layer = 10, show_colorbar = True):

	# initialize folder structure if not yet done
	print('...checking folder structure')
	folders = ['digit_reconstructions', os.path.join('digit_reconstructions', folder_name), os.path.join('digit_reconstructions', folder_name, run_name)]
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	cae_filters = []
	walkthrough = []

	if use_training_set:
		dataset = mnist.train.images
	else:
		dataset = mnist.test.images

	encoding, reconst, error, walkthrough = sess.run([autoencoder.encoding, autoencoder.reconstruction, autoencoder.error, autoencoder.model_walkthrough], feed_dict={input_placeholder: dataset[0:num_images].reshape(num_images, input_size[0], input_size[1], 1)})

	print('jener error: ', error)


	with sess.as_default():
		for cw in autoencoder.conv_weights:
			cae_filters.append(cw.eval())


	print(len(cae_filters), len(walkthrough))

	# workaround to make old code work
	cae_filters = cae_filters[0]


	num_filters = cae_filters.shape[3]


	if autoencoder.pooling_type == 'max_pooling' or autoencoder.pooling_type == 'strided_conv':
		code_dimx = 7
	else:
		code_dimx = 28

	code_dimy = code_dimx

	print('cae_filters.shape = {}'.format(cae_filters[0].shape))
	print('encoding.shape    = {}'.format(encoding.shape))
	print('reconst.shape     = {}'.format(reconst.shape))
	print('walkthrough shapes = {}'.format(map(np.shape, walkthrough)))

	print('save {} example images to file'.format(num_images))

	for i in range(num_images):

		print('...treating image {}'.format(i))

		print('representation walkthrough')

		fig = plt.figure(figsize=(40 , 40))

		max_size 			= np.max(np.array(autoencoder.hidden_channels))
		hidden_layer_count 	= len(walkthrough)
		
		rows = hidden_layer_count + 2
		cols = max_size

		# plot input
		plt.subplot(rows, 1, 1)
		plt.imshow(dataset[i].reshape(input_size[0], input_size[1]), cmap='gray', interpolation='None')
		plt.axis('off')
		if show_colorbar:
			plt.colorbar(orientation="horizontal",fraction=0.07)

		# plot reconstruction
		plt.subplot(rows,1 , rows)
		plt.imshow(reconst[i].reshape(input_size[0], input_size[1]), cmap='gray', interpolation='None')
		plt.axis('off')
		if show_colorbar:
			plt.colorbar(orientation="horizontal",fraction=0.07)
		
		stretcher = 0
		print 'hlc: ', hidden_layer_count

		for c in range(hidden_layer_count):
			hc_size = walkthrough[c].shape[3]

			if c <= hidden_layer_count / 2:
				stretcher += 1

			else:
				stretcher -= 1
			
			print stretcher

			if max_maps_per_layer > 0:
				hc_size = min(hc_size, max_maps_per_layer) # * stretcher
			
			

			for r in range(hc_size):

				# plot feature map of filter r in the c-th hidden layer
				plt.subplot(rows,hc_size, (c + 1) * hc_size + r + 1)
				plt.imshow(walkthrough[c][i,:,:,r], cmap='gray', interpolation='none')
				plt.axis('off')
				if show_colorbar:
					plt.colorbar(orientation="horizontal",fraction=0.07)


		# plt.tight_layout()
		plt.savefig(os.path.join('digit_reconstructions', folder_name, run_name, '{}_{}_feature_maps.png'.format(run_name,i)))
		plt.close(fig)
		'''
		print('filter + representation')
		fig = plt.figure(figsize=(10 * num_filters , 40))

		if plot_first_layer_filters:

			plt.subplot(4,1,1)
			# plt.title('input image', fontsize=fontsize)
			plt.imshow(dataset[i].reshape(input_size[0], input_size[1]), cmap='gray', interpolation='None')
			plt.axis('off')

			print('minimum_filter_value: ', np.min(cae_filters[:,:,0,:]))

			max_abs_filters 	= np.max(np.absolute(cae_filters[:,:,0,:]))
			max_abs_encodings 	= np.max(np.absolute(encoding[i,:,:,:]))

			norm_filters 	= mpl.colors.Normalize(vmin=-max_abs_filters,vmax=max_abs_filters)
			norm_encodings 	= mpl.colors.Normalize(vmin=-max_abs_encodings,vmax=max_abs_encodings)


			for f in range(num_filters):

				plt.subplot(4,num_filters, num_filters + f + 1)

				if common_scaling:
					plt.imshow(cae_filters[:,:,0,f], cmap='gray', interpolation='None', norm=norm_filters)
				else:
					plt.imshow(cae_filters[:,:,0,f], cmap='gray', interpolation='None')

				plt.axis('off')

				plt.subplot(4,num_filters, 2 * num_filters + f + 1)

				if common_scaling:
					plt.imshow(encoding[i,:,:,f], cmap='gray', interpolation='None', norm=norm_encodings)
				else:
					plt.imshow(encoding[i,:,:,f], cmap='gray', interpolation='None')

				plt.axis('off')

			plt.subplot(4,1,4)
			# plt.title('reconstruction', fontsize=fontsize)
			plt.imshow(reconst[i].reshape(input_size[0], input_size[1]), cmap='gray', interpolation='None')
			plt.axis('off')

			plt.tight_layout()

			plt.savefig(os.path.join('digit_reconstructions', 'cae_example{}.png'.format(i)))

			plt.close(fig)
		'''

if __name__ == '__main__':
	main()
