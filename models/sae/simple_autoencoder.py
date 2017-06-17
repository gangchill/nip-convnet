import tensorflow as tf 

class SAE: 
	# simple autoencoder

	def __init__(self, data, hidden_layer_size, step_size = 0.0001):

		print('Initializing autoencoder with hidden layer size {}'.format(hidden_layer_size))

		self.data = data

		self.input_layer_size  		= int(data.shape[1]) # assuming the input data is in NxD format
		self.hidden_layer_size 	= hidden_layer_size

		self.step_size = step_size

		self._encoding 				= None
		self._optimize				= None
		self._logit_reconstruction 	= None
		self._reconstruction		= None
		self._error					= None

		with tf.name_scope('autoencoder'):
			self.optimize
			self.reconstruction


	@property
	def encoding(self):
		# returns the hidden layer representation (encoding) of the autoencoder

		if self._encoding is None:

			print('initialize encoding')

			with tf.name_scope('encoding'):
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
	def logit_reconstruction(self):
		# returns the reconstruction node that contains the reconstruction of the input

		if self._logit_reconstruction is None:
			print('initialize logit reconstruction')
			with tf.name_scope('reconstruction'):
				self._logit_reconstruction = tf.add(tf.matmul(self.encoding, tf.transpose(self.W)),self.c, name='logit_reconstruction')
		
		return self._logit_reconstruction

	@property
	def reconstruction(self):

		if self._reconstruction is None:
			print('initialize reconstruction')
			self._reconstruction = tf.nn.sigmoid(self.logit_reconstruction)

		return self._reconstruction


	@property
	def error(self):
		# returns the training error node (cross-entropy) used for the training and testing

		if self._error is None:
			print('initialize error')

			self._error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.data, logits=self.logit_reconstruction, name='cross-entropy_error')

		return self._error

	@property
	def optimize(self):
		# returns the cross-entropy node we use for the optimization

		if self._optimize is None:
			print('initialize optimizer')

			step_size = self.step_size

			self._optimize = tf.train.GradientDescentOptimizer(step_size).minimize(self.error)

		return self._optimize

	def store_model_to_file(self, sess, path_to_file):

		# TODO: add store / save function to the class
		saver = tf.train.Saver()
		save_path = saver.save(sess, path_to_file)

		print('Model was saved in {}'.format(save_path))

		return save_path

	def load_model_from_file(self, sess, path_to_file):

		saver = tf.train.Saver()
		saver.restore(sess, path_to_file)

		print('Restored model from {}'.format(path_to_file))
