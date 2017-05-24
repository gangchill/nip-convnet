import tensorflow as tf 

class CAE: 
	# convolutional autoencoder 

	def __init__(self, data, filter_height, filter_width, out_channels, strides = [1,1,1,1], batch_size_workaround=100):

		# TODO: init all values defining the shape of the convolutions
		self.filter_height 	= filter_height
		self.filter_width 	= filter_width
		self.in_channels 	= int(data.shape[3])
		self.out_channels	= out_channels
		self.strides 		= strides 

		# batch size used for conv2d_transposed, TODO: find a nicer way to define this at runtime
		self.batch_size_workaround = batch_size_workaround

		print 'Initializing conv autoencoder with {}x{} kernels and {} feature maps'.format(filter_height, filter_width, out_channels)

		self.data = data # we assume data in NHWC format 

		self._encoding 			= None
		self._optimize			= None
		self._logit_reconstruction = None
		self._reconstruction	= None
		self._error				= None

		with tf.name_scope('CAE'):
			self.optimize
			self.reconstruction

	@property
	def encoding(self):
		# returns the hidden layer representation (encoding) of the autoencoder

		if self._encoding is None:

			print 'initialize encoding'

			# initialize filter and bias variables for the encoding:
			filter_shape = [self.filter_height, self.filter_width, self.in_channels, self.out_channels]
			self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='conv1_weights')

			# TODO: bias for each output channel or for each output dimension? 
			b = tf.Variable(tf.constant(0.1, shape=[self.out_channels]), name='conv1_bias')

			# hidden layer representation:
			# TODO: check relu function, how would we invert it for the reconstruction? Do we nee to?
			# TODO: tanh / sigmoid??

			post_conv_act = tf.nn.sigmoid( tf.nn.conv2d(self.data, self.W, strides = self.strides, padding='SAME') + b)

			# add max-pooling for dimensionality reduction
			# self._encoding = tf.nn.max_pool(post_conv_act, [1,2,2,1], [1,2,2,1], padding='SAME')
			self._encoding = post_conv_act

		return self._encoding

	@property
	def error(self):
		# returns the training error node (cross-entropy) used for the training and testing

		if self._error is None:
			print 'initialize error'

			self._error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.data, logits=self.logit_reconstruction, name='cross-entropy_error')

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
	def logit_reconstruction(self):
		# returns the node containing the reconstruction before applying the sigmoid function 
		# the logit_reconstruction is separated to use the sigmoid_cross_entropy_with_logits function for the error

		if self._logit_reconstruction is None:

			# upsample_strides = [1,2,2,1]

			# self.reconst_W = tf.Variable(tf.truncated_normal([3, 3, 1, 10] , stddev=0.1))

			self.c = tf.Variable(tf.constant(0.1, shape=[self.in_channels]), name='reconstruction_bias')

			self._logit_reconstruction = tf.add( tf.nn.conv2d_transpose(self.encoding, self.W, tf.shape(self.data), self.strides), self.c, name='logit_reconstruction')

		return self._logit_reconstruction

	@property
	def reconstruction(self):
		# returns the reconstruction node that contains the reconstruction of the input

		if self._reconstruction is None:
			print 'initialize reconstruction'

			self._reconstruction = tf.nn.sigmoid(self.logit_reconstruction, name='reconstruction')
		
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
