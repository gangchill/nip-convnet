# --------------------------------------------------------------------------------------
# train and test a convolutional autoencoder with one hidden layer for the MNIST dataset
# --------------------------------------------------------------------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# import the simple autoencoder class from SAE.py
from models.cae.convolutional_autoencoder import CAE

########
# MAIN #
########

def main():

	# directory containing the autoencoder file
	cae_dir 		= os.path.join('models', 'cae')
	cae_weights_dir	= os.path.join(cae_dir, 'weights')

	# restore weights from file if an autoencoder with the same architecture has already been trained before
	restore_weights_if_existant = True # ATTENTION: currently simply restores the weights from the last run if True
	# TODO: adapt filename to the more complex setup 

	# import mnist data set
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# input variables: x (images)
	x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')

	# reshape the input to NHWD format
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# AUTOENCODER SPECIFICATIONS
	filter_dims 	= [(7,7), (3,3)]
	hidden_channels = [4,6] 
	use_max_pooling = True
	strides = None # other strides should not work yet
	activation_function = 'sigmoid'

	# construct autoencoder (5x5 filters, 3 feature maps)
	autoencoder = CAE(x_image, filter_dims, hidden_channels, strides, use_max_pooling, activation_function, store_model_walkthrough = True)

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	print("Begin autencoder training")
	batch_size 		= 100
	max_iterations 	= 100
	chk_iterations  = 100


	weight_file_name = 'last_weights' # = 'cae_weights_{}_{}_{}_{}.ckpt'.format(filter_height, filter_width, num_feature_maps, max_iterations)

	if restore_weights_if_existant:
		# only train a new autoencoder if no weights file is found

		cwd = os.getcwd()
		chkpnt_file_path = os.path.join(cwd, cae_weights_dir, weight_file_name)

		if os.path.exists(chkpnt_file_path + '.index'):
			print 'Model file for same configuration was found ... load weights'

			autoencoder.load_model_from_file(sess, chkpnt_file_path)			

		else:
			train_ae(sess, x, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name, batch_size, max_iterations, chk_iterations)

	else:
		# always train a new autoencoder 
		train_ae(sess, x, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name, batch_size, max_iterations, chk_iterations)
	

	print 'Test the training:'

	# visualize_cae_filters(sess, autoencoder)

	visualize_ae_representation(sess, x_image, autoencoder, mnist, 3)


	# add logwriter for tensorboard
	writer = tf.summary.FileWriter("logs", sess.graph)
	writer.close()

	sess.close()


def train_ae(sess, input_placeholder, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name, batch_size=100, max_iterations=1000, chk_iterations=500):

	print('...checking folder structure')
	folders = ['models', cae_dir, cae_weights_dir]
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	print("Training for {} iterations with batchsize {}".format(max_iterations, batch_size))

	for i in range(max_iterations):

	  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	  sess.run(autoencoder.optimize, feed_dict={input_placeholder: batch_xs})

	  if chk_iterations > 100 and i % 100 == 0:
	  	print '...iteration {}'.format(i)

	  
	  if i % chk_iterations == 0:

		avg_r_e = sess.run(autoencoder.error, feed_dict={input_placeholder: mnist.test.images})

		print('it {} avg_re {}'.format(i, np.mean(avg_r_e)))


	print '...finished training'

	autoencoder.store_model_to_file(sess, os.path.join(cae_weights_dir, weight_file_name))
	print '...saved model to file'


def visualize_cae_filters(sess, autoencoder): 

	folders = ['filters']
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	print 'save the filters to file:'

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

	print 'jener error: ', np.mean(error)


	with sess.as_default():
		for cw in autoencoder.conv_weights:
			cae_filters.append(cw.eval())


	print len(cae_filters), len(walkthrough)

	# workaround to make old code work
	# TODO: change the visualization to be able to show all filters + feature maps
	cae_filters = cae_filters[0]


	num_filters = cae_filters.shape[3]


	if autoencoder.use_max_pooling:
		code_dimx = 7
	else:
		code_dimx = 28

	code_dimy = code_dimx

	print 'cae_filters.shape = ', cae_filters[0].shape
	print 'encoding.shape    = ', encoding.shape
	print 'reconst.shape     = ', reconst.shape

	print 'save {} example images to file'.format(num_images)

	for i in range(num_images):

		print '...treating image {}'.format(i)

		fig = plt.figure(figsize=(10 * num_filters , 40))

		plt.subplot(4,1,1)
		# plt.title('input image', fontsize=fontsize)
		plt.imshow(dataset[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')

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
				plt.imshow(encoding[i,:,:,f].reshape(code_dimx, code_dimy), cmap='gray', interpolation='None', norm=norm_encodings)
			else:
				plt.imshow(encoding[i,:,:,f].reshape(code_dimx, code_dimy), cmap='gray', interpolation='None')

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