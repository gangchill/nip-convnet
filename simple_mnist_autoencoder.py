# ----------------------------------------------------------------
# a simple autoencoder with one hidden layer for the MNIST dataset
# ----------------------------------------------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.sparse


class SAE: 
	# simple autoencoder

	def __init__(self, data, hidden_layer_size, input_layer_size = 784):

		print 'Initializing autoencoder with hidden layer size {}'.format(hidden_layer_size)

		self.data = data

		self.input_layer_size  = input_layer_size
		self.hidden_layer_size = hidden_layer_size

		self._encoding 			= None
		self._optimize			= None
		self._reconstruction	= None
		self._error				= None


	@property
	def encoding(self):
		# returns the hidden layer representation (encoding) of the autoencoder

		if self._encoding is None:

			print 'initialize encoding'

			with tf.name_scope('autoencoder_network'):
				# TODO: switch to gaussian or xavier init instead of uniform
				current_layer = 1
				lim_value = 1. / (self.hidden_layer_size ** (current_layer - 1) )

				# model variables W, b:
				self.W = tf.Variable(tf.random_uniform([self.input_layer_size, self.hidden_layer_size], minval=-lim_value, maxval=lim_value), name='encoding_weights')
				b = tf.Variable(tf.zeros([self.hidden_layer_size]), name='encoding_bias')
				self.c = tf.Variable(tf.zeros([self.input_layer_size]), name='reconstruction_bias')
				
				# hidden layer representation:
				self._encoding = tf.nn.sigmoid(tf.matmul(self.data, self.W) + b, name='encoding')

		return self._encoding

	@property
	def error(self):
		# returns the training error node (cross-entropy) used for the training and testing

		if self._error is None:
			print 'initialize error'

			reconstruction = tf.add(tf.matmul(self.encoding, tf.transpose(self.W)), self.c, name='reconstruction_without_sigmoid')
			self._error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.data, logits=reconstruction, name='cross-entropy_error')

		return self._error

	@property
	def optimize(self):
		# returns the cross-entropy node we use for the optimization

		if self._optimize is None:
			print 'initialize optimizer'

			# TODO: make step size modifiable
			step_size = 0.001

			self._optimize = tf.train.GradientDescentOptimizer(step_size).minimize(self.error)

		return self._optimize

	@property
	def reconstruction(self):
		# returns the reconstruction node that contains the reconstruction of the input

		if self._reconstruction is None:
			print 'initialize reconstruction'

			self._reconstruction = tf.nn.sigmoid(tf.matmul(self.encoding, tf.transpose(self.W)) + self.c, name='reconstruction')
		return self._reconstruction

	def store_model_to_file(self, sess, path_to_file):

		# TODO: add store / save function to the class
		saver = tf.train.Saver()
		save_path = saver.save(sess, path_to_file)

		print 'Model was saved in {}'.format(save_path)

		return save_path

	def load_model_from_file(self, sess, path_to_file):

		saver = tf.train.Saver()
		saver.restore(sess, path_to_file)

		print 'Restored model from {}'.format(path_to_file)



########
# MAIN #
########

def main():

	# restore weights from file if an autoencoder with the same architecture has already been trained before
	restore_weights_if_existant = True

	# import mnist data set
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	hidden_layer_size = 5*5

	# input variables: x (images)
	x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')

	# construct autoencoder
	autoencoder = SAE(x, hidden_layer_size)

	print 'call the properties to initialize the graph'
	autoencoder.optimize
	autoencoder.reconstruction

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	# TODO: add store / save function to the class
	# saver = tf.train.Saver()
	# save_path = saver.save(sess, "models/weights.ckpt")

	print("Begin autencoder training")
	batch_size 		= 100
	max_iterations 	= 1000
	chk_iterations  = 500

	if restore_weights_if_existant:
		# only train a new autoencoder if no weights file is found

		cwd = os.getcwd()
		chkpnt_file_path = os.path.join(cwd, 'models', '{}_autoencoder_{}it.ckpt'.format(autoencoder.hidden_layer_size, max_iterations))

		if os.path.exists(chkpnt_file_path + '.index'):
			print 'Model file for same configuration was found ... load weights'

			autoencoder.load_model_from_file(sess, chkpnt_file_path)			

		else:
			train_ae(sess, x, autoencoder, mnist, batch_size, max_iterations, chk_iterations)

	else:
		# always train a new autoencoder 
		train_ae(sess, x, autoencoder, mnist, batch_size, max_iterations, chk_iterations)
	

	print 'Test the training:'

	visualize_ae_representation(sess, x, autoencoder, mnist, 1)


	# add logwriter for tensorboard
	writer = tf.summary.FileWriter("logs", sess.graph)
	writer.close()

	sess.close()


def train_ae(sess, input_placeholder, autoencoder, mnist, chkpnt_file_path, batch_size=100, max_iterations=1000, chk_iterations=500):

	print('...checking folder structure')
	folders = ['models']
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

	autoencoder.store_model_to_file(sess, chkpnt_file_path)
	print '...saved model to file'

def visualize_ae_representation(sess, input_placeholder, autoencoder, mnist, num_images):

	# initialize folder structure if not yet done
	print('...checking folder structure')
	folders = ['digit_reconstructions']
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	encoding, reconst = sess.run([autoencoder.encoding, autoencoder.reconstruction], feed_dict={input_placeholder: mnist.test.images[0:num_images]})

	code_dimx = int(autoencoder.hidden_layer_size**.5)

	print 'save {} example images to file'.format(num_images)

	for i in range(num_images):

		fontsize = 30

		fig = plt.figure(figsize=(20,10))

		plt.subplot(1,3,1)
		plt.title('input image', fontsize=fontsize)
		plt.imshow(mnist.test.images[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')


		plt.subplot(1,3,2)
		plt.title('encoded representation', fontsize=fontsize)
		plt.imshow(encoding[i].reshape(code_dimx, code_dimx), cmap='gray', interpolation='None')
		plt.axis('off')

		plt.subplot(1,3,3)
		plt.title('reconstruction', fontsize=fontsize)
		plt.imshow(reconst[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')

		plt.tight_layout()

		plt.savefig(os.path.join('digit_reconstructions', 'class_ae_example{}.png'.format(i)), dpi=400)


if __name__ == '__main__':
	main()