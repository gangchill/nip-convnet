# --------------------------------------------------------------------------------------
# train and test a convolutional autoencoder with one hidden layer for the MNIST dataset
# --------------------------------------------------------------------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from functools import reduce


# import the simple autoencoder class from SAE.py
from models.cae.convolutional_autoencoder import CAE
from scripts.train_cae import train_ae

########
# MAIN #
########

def main():

	# directory containing the autoencoder file
	cae_dir 		= os.path.join('models', 'cae')
	cae_weights_dir	= os.path.join(cae_dir, 'weights')

	# restore weights from the last iteration (if the same training setup was used before)
	restore_last_checkpoint = True

	# store model walkthrough (no CIFAR support yet)
	visualize_model_walkthrough = True

	## ########### ##
	# INPUT HANDING #
	## ########### ##


	DATASET = "CIFAR10"

	if DATASET == "MNIST":
		# load mnist
		from tensorflow.examples.tutorials.mnist import input_data
		dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_size = (28, 28)
		num_classes = 10
		nhwd_shape = False

	elif DATASET == "CKPLUS":
		import scripts.load_ckplus as load_ckplus
		dataset = load_ckplus.read_data_sets(one_hot=True)
		input_size = (49, 64)
		num_classes = load_ckplus.NUM_CLASSES
		nhwd_shape = False

	elif DATASET=="CIFAR10":
		dataset 		= "cifar_10" 	# signals the train_ae function that it needs to load the data via cifar10_input.py
		input_size 		= (24, 24, 3)
		num_classes 	= 1
		nhwd_shape 		= True

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



	## #### ##
	# CONFIG # 
	## #### ##

	# TODO Sabbir: begin what needs to be in the config file -----------------------------

	# AUTOENCODER SPECIFICATIONS
	filter_dims 	= [(5,5), (5,5)]
	hidden_channels = [16, 16]
	pooling_type 	= 'strided_conv'
	strides = None # other strides should not work yet
	activation_function = 'relu'
	relu_leak = 0.2 # only for leaky relus

	error_function 	= 'mse' 					# default is cross-entropy
	optimizer_type 	= 'gradient_descent' 		# default is gradient descent

	output_reconstruction_activation = 'scaled_tanh'

	weight_init_mean 	= 0.001
	weight_init_stddev 	= 0.05
	initial_bias_value  = 0.001

	batch_size 		= 128
	max_iterations 	= 10001
	chk_iterations  = 100
	step_size 		= 0.1

	tie_conv_weights = True

	# TODO Sabbir: end what needs to be in the config file -----------------------------


	weight_file_name = get_weight_file_name(filter_dims, hidden_channels, pooling_type, activation_function, tie_conv_weights, batch_size, step_size, weight_init_mean, weight_init_stddev, initial_bias_value)


	log_folder_name = '02_CIFAR_2enc'
	run_name = 'old_commit_style'
	# run_name 	= '{}'.format(weight_file_name)
	# run_name = '({}x{})|_{}_{}_{}|{}_{}'.format(activation_function, output_reconstruction_activation, weight_init_mean, weight_init_stddev, initial_bias_value)
	# run_name = '5x5_d2_smaller_init_{}_{}'.format(activation_function, output_reconstruction_activation)
	# run_name = '{}_{}({}|{})'.format(pooling_type, output_reconstruction_activation, '-'.join(map(str, hidden_channels)), step_size)
	# run_name = '{}_{}_{}_{}_{}({})'.format(DATASET, error_function, activation_function, output_reconstruction_activation,pooling_type, weight_init_mean)
	# run_name = 'relu_small_learning_rate_101_{}'.format(weight_file_name)
	# run_name = 'that_run_tho'


	# folder to store the training weights in:
	model_save_parent_dir = 'weights'
	save_path = os.path.join(model_save_parent_dir, log_folder_name, run_name)
	check_dirs = [model_save_parent_dir, os.path.join(model_save_parent_dir, log_folder_name), os.path.join(model_save_parent_dir, log_folder_name), os.path.join(model_save_parent_dir, log_folder_name, run_name)]
	
	for directory in check_dirs:
		if not os.path.exists(directory):
			os.makedirs(directory)


	## ###### ##
	# TRAINING #
	## ###### ##


	# construct autoencoder (5x5 filters, 3 feature maps)
	autoencoder = CAE(x_image, filter_dims, hidden_channels, step_size, weight_init_stddev, weight_init_mean, initial_bias_value, strides, pooling_type, activation_function, tie_conv_weights, store_model_walkthrough = True, relu_leak = relu_leak, optimizer_type = optimizer_type, output_reconstruction_activation=output_reconstruction_activation)

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	print("Begin autencoder training")
	
	writer = tf.summary.FileWriter("logs/{}/{}".format(log_folder_name, run_name), sess.graph)

	init_iteration = 0

	if restore_last_checkpoint:
		# initialize training with weights from a previous training 

		cwd = os.getcwd()
		chkpnt_file_path = os.path.join(cwd, save_path)

		saver = tf.train.Saver(autoencoder.all_variables_dict)
		latest_checkpoint = tf.train.latest_checkpoint(chkpnt_file_path)

		if latest_checkpoint is not None:

			print('Found checkpoint')

			init_iteration 					= int(latest_checkpoint.split('-')[-1]) + 1
			smallest_reconstruction_error  	= float(latest_checkpoint.split('-')[-2])

			print('iteration is: {}'.format(init_iteration))
			print('smallest reconstruction error so far was {}'.format(smallest_reconstruction_error))

			saver.restore(sess, latest_checkpoint)

			train_ae(sess, writer, x, autoencoder, dataset, cae_dir, cae_weights_dir, weight_file_name, error_function, batch_size, init_iteration, max_iterations, chk_iterations, save_prefix = save_path, minimal_reconstruction_error = smallest_reconstruction_error)

		else:
			print('No checkpoint was found, beginning with iteration 0')
			train_ae(sess, writer, x, autoencoder, dataset, cae_dir, cae_weights_dir, weight_file_name, error_function, batch_size,init_iteration,  max_iterations, chk_iterations, save_prefix = save_path)


	else:
		# always train a new autoencoder 
		train_ae(sess, writer, x, autoencoder, dataset, cae_dir, cae_weights_dir, weight_file_name, error_function, batch_size,init_iteration,  max_iterations, chk_iterations, save_prefix = save_path)

	# print('Test the training:')

	# visualize_cae_filters(sess, autoencoder)
	if visualize_model_walkthrough and not DATASET == 'CIFAR10':
		visualize_ae_representation(sess, x_image, autoencoder, dataset, log_folder_name, run_name, input_size, 2)


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


def visualize_ae_representation(sess, input_placeholder, autoencoder, mnist, folder_name, run_name, input_size, num_images = 100, use_training_set = False, common_scaling = False, plot_first_layer_filters = False):

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
	# TODO: change the visualization to be able to show all filters + feature maps
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

		fig = plt.figure(figsize=(10 * num_filters , 40))

		max_size 			= np.max(np.array(autoencoder.hidden_channels))
		hidden_layer_count 	= len(walkthrough)
		
		rows = hidden_layer_count + 2
		cols = max_size

		# plot input
		plt.subplot(rows, 1, 1)
		plt.imshow(dataset[i].reshape(input_size[0], input_size[1]), cmap='gray', interpolation='None')
		plt.axis('off')
		plt.colorbar(orientation="horizontal",fraction=0.07)

		# plot reconstruction
		plt.subplot(rows,1 , rows)
		plt.imshow(reconst[i].reshape(input_size[0], input_size[1]), cmap='gray', interpolation='None')
		plt.axis('off')
		plt.colorbar(orientation="horizontal",fraction=0.07)

		for c in range(hidden_layer_count):
			hc_size = walkthrough[c].shape[3]
			for r in range(hc_size):

				# plot feature map of filter r in the c-th hidden layer
				plt.subplot(rows,hc_size, (c + 1) * hc_size + r + 1)
				plt.imshow(walkthrough[c][i,:,:,r], cmap='gray', interpolation='none')
				plt.axis('off')
				plt.colorbar(orientation="horizontal",fraction=0.07)


		# plt.tight_layout()
		plt.savefig(os.path.join('digit_reconstructions', folder_name, run_name, '{}_{}_feature_maps.png'.format(run_name,i)))
		plt.close(fig)

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


if __name__ == '__main__':
	main()