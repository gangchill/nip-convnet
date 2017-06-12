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

	# restore weights from file if an autoencoder with the same architecture has already been trained before
	restore_weights_if_existant = False
	# TODO: adapt filename to the more complex setup 

	# import mnist data set
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# input variables: x (images)
	x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')

	# reshape the input to NHWD format
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# AUTOENCODER SPECIFICATIONS
	filter_dims 	= [(5,5), (5,5)]
	hidden_channels = [16, 32] 
	pooling_type 	= 'max_pooling'
	strides = None # other strides should not work yet
	activation_function = 'sigmoid'

	batch_size 		= 100
	max_iterations 	= 2000
	chk_iterations  = 100
	step_size 		= 0.0001

	tie_conv_weights = True

	weight_file_name = get_weight_file_name(filter_dims, hidden_channels, pooling_type, activation_function, batch_size, max_iterations)


	folder_name = 'dcae_reference_try'
	run_name 	= 'dcae_tied_weights{}'.format(weight_file_name)


	# construct autoencoder (5x5 filters, 3 feature maps)
	autoencoder = CAE(x_image, filter_dims, hidden_channels, step_size, strides, pooling_type, activation_function, tie_conv_weights, store_model_walkthrough = True)

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	print("Begin autencoder training")
	

	
	writer = tf.summary.FileWriter("logs/{}/{}".format(folder_name, run_name), sess.graph)

	if restore_weights_if_existant:
		# only train a new autoencoder if no weights file is found

		cwd = os.getcwd()
		chkpnt_file_path = os.path.join(cwd, cae_weights_dir, weight_file_name)

		if os.path.exists(chkpnt_file_path + '.index'):
			print('Model file for same configuration was found ... load weights')

			autoencoder.load_model_from_file(sess, chkpnt_file_path)

		else:
			train_ae(sess, writer, x, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name, batch_size, max_iterations, chk_iterations)

	else:
		# always train a new autoencoder 
		train_ae(sess, writer, x, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name, batch_size, max_iterations, chk_iterations)
	

	# print('Test the training:')

	# visualize_cae_filters(sess, autoencoder)
	# visualize_ae_representation(sess, x_image, autoencoder, mnist, 2)


	# add logwriter for tensorboard
	writer.close()

	sess.close()

def get_weight_file_name(filter_dims, hidden_channels, pooling_type, activation_function, batch_size, max_iterations):
	# define unique file name for architecture + training combination

	# architecture:
	filter_dims_identifier 		= reduce(lambda x,y: '{}|{}'.format(x,y), map(lambda xy: '{},{}'.format(xy[0],xy[1]), filter_dims))
	hidden_channels_identifier 	= reduce(lambda x,y: '{}|{}'.format(x,y), hidden_channels)
	
	mp_identifier = pooling_type

	architecture_identifier = '({}-{}{}-{})'.format(filter_dims_identifier, hidden_channels_identifier, mp_identifier, activation_function)

	# training:
	training_identifier = '({},{})'.format(batch_size, max_iterations)

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


def visualize_ae_representation(sess, input_placeholder, autoencoder, mnist, num_images = 100, use_training_set = False, common_scaling = False):

	# initialize folder structure if not yet done
	print('...checking folder structure')
	folders = ['digit_reconstructions']
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

	encoding, reconst, error, walkthrough = sess.run([autoencoder.encoding, autoencoder.reconstruction, autoencoder.error, autoencoder.model_walkthrough], feed_dict={input_placeholder: dataset[0:num_images].reshape(num_images, 28, 28, 1)})

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
		plt.imshow(dataset[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')

		# plot reconstruction
		plt.subplot(rows,1 , rows)
		plt.imshow(reconst[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')

		for c in range(hidden_layer_count):
			hc_size = walkthrough[c].shape[3]
			for r in range(hc_size):

				# plot feature map of filter r in the c-th hidden layer
				plt.subplot(rows,hc_size, (c + 1) * hc_size + r + 1)
				plt.imshow(walkthrough[c][i,:,:,r], cmap='gray', interpolation='none')
				plt.axis('off')


		# plt.tight_layout()
		plt.savefig(os.path.join('digit_reconstructions', 'cae_example{}_feature_maps.png'.format(i)))
		plt.close(fig)

		print('filter + representation')
		fig = plt.figure(figsize=(10 * num_filters , 40))



		plt.subplot(4,1,1)
		# plt.title('input image', fontsize=fontsize)
		plt.imshow(dataset[i].reshape(28, 28), cmap='gray', interpolation='None')
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
		plt.imshow(reconst[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')

		plt.tight_layout()

		plt.savefig(os.path.join('digit_reconstructions', 'cae_example{}.png'.format(i)))

		plt.close(fig)


if __name__ == '__main__':
	main()