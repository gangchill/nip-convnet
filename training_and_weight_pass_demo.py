import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# import the autoencoder and cnn class
from models.cnn.simple_cnn import SCNN
from models.cae.convolutional_autoencoder import CAE

# import the training procedures
from scripts.train_cae import train_ae
from scripts.train_cnn import train_cnn

def main():

	cae_dir 		= os.path.join('models', 'cae')
	cae_weights_dir	= os.path.join(cae_dir, 'weights')

	# load mnist
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	with tf.name_scope('Input'):
		# input variables: x (images), y_ (labels), keep_prob (dropout rate)
		x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')
		# reshape the input to NHWD format
		x_image = tf.reshape(x, [-1, 28, 28, 1], name='input_digits_NHWD')
		y_ = tf.placeholder(tf.float32, [None, 10], name='target_labels')

		keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

	

	## ##################### ##
	# ARCHITECTURE PARAMETERS #
	## ##################### ##

	# feature extraction (both CAE and CNN)
	filter_dims 	= [(5,5), (5,5)]
	hidden_channels = [16, 32] 
	pooling_type  = 'strided_conv' # dont change, std::bac_alloc otherwise (TODO: understand why)
	strides = None # other strides should not work yet
	activation_function = 'sigmoid'

	# CAE only:
	tie_conv_weights = True

	# CNN only:
	dense_depths = []

	## ################# ##
	# TRAINING PARAMETERS #
	## ################# ##

	# currently, the same parameters are used for the training of the cae and the cnn
	batch_size 		= 100
	max_iterations	= 6001
	chk_iterations 	= 300
	dropout_k_p		= 0.5
	step_size 		= 0.0001

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

	autoencoder = CAE(x_image, filter_dims, hidden_channels, step_size, strides, pooling_type, activation_function, tie_conv_weights)
	cnn = SCNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function, scope_name='pre_trained_CNN')

	# second cnn with the same structure that will be trained independently from the autoencoder
	comparison_cnn = SCNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type, activation_function, scope_name='reference_CNN')

	## ###### ##
	# TRAINING #
	## ###### ##

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# add logwriter for tensorboard
	writer = tf.summary.FileWriter(log_path, sess.graph)

	# train the autoencoder
	train_ae(sess, writer, x, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name_cae, batch_size, max_iterations, chk_iterations)
	print('...finished training the cae')

	# save autoencoder weights to file
	autoencoder.store_encoding_weights(sess, encoding_weights_path)
	print('...saved cae encoding weights to file')

	# load the same weights into the cnn
	cnn.load_encoding_weights(sess, encoding_weights_path)
	print('...loaded the cae weights into the cnn')

	# train the cnn
	train_cnn(sess, cnn, mnist, x, y_, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations, writer, fine_tuning_only)
	print('...finished training the cnn')

	train_cnn(sess, comparison_cnn, mnist, x, y_, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations, writer, fine_tuning_only)
	print('...finished training comparison cnn')

	# train the comparison cnn



	writer.close()
	sess.close()


if __name__ == '__main__':
	main()