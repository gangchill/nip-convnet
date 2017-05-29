import tensorflow as tf 

class SCNN: 
	# simple convolutional neural network (same structure as cae with added fully-connected layers)

	def __init__(self, data, target, keep_prob, filter_dims, hidden_channels, dense_depths,  strides = None, use_max_pooling = True, activation_function = 'sigmoid', store_model_walkthrough = False):

		# TODO:
		# 	- add assertion that test whether filter_dims, hidden_channels and strides have the right dimensions
		# 	- add the possibility of variable strides (currently only strides = [1,1,1,1] for all dimensions should work if use_max_pooling = True)
		# 	  (the upsampling_strides need to be adapted for the upsampling)
		# 	- verify the bias treatment (currently: the same bias for every pixel in a given feature map)

		self.data = data # we assume data in NHWC format 
		self.target = target	# labels (assumed to be in one-hot encoding)

		self.keep_prob = keep_prob # input probability for dropout regularization (set to 1.0 for inference)

		# filter_dims, out_channels and strides (if specified) are lists containing the specifications for each of the consecutive layers
		# the choice of mac pooling and activation function is used for the whole network (the last activation function is always a sigmoid)

		self.filter_dims 		= filter_dims 		# height and width of the conv kernels 	for each layer
		self.hidden_channels 	= hidden_channels	# number of feature maps 				for each layer
		if strides is None:
			self.strides = [[1,1,1,1] for filter in filter_dims]

		# layer sizes for the dense layers (decision making)
		self.dense_depths = dense_depths

		# add a dense shape for the readout layer
		self.dense_depths.append(self.target.get_shape().as_list()[1])

		self.use_max_pooling 		= use_max_pooling
		self.activation_function	= activation_function

		# init lists that will store weights and biases for the convolution operations
		self.conv_weights 	= []
		self.conv_biases	= []

		# TODO: implement storage of model walkthrough
		self.store_model_walkthrough = store_model_walkthrough
		if self.store_model_walkthrough:
			# initialize list that stores all the intermediate tensors in a forward path (probably high memory consumption, set flag to False if any problems occur)
			self.model_walkthrough = []

		# private attributes used by the properties
		self._encoding 		= None
		self._logits 		= None
		self._prediction 	= None
		self._error			= None
		self._optimize		= None
		self._accuracy		= None
		

		print('Initializing simple CNN')
		with tf.name_scope('CNN'):
			self.optimize
		self.accuracy

	@property
	def encoding(self):
		# returns the feature extraction of the CNN (same architecture as the encoding of the CAE)

		if self._encoding is None:

			print('initialize encoding')

			tmp_tensor = self.data

			for layer in range(len(self.filter_dims)):

				# CONVOLUTION
				if layer == 0:
					in_channels = int(self.data.shape[3])
				else:

					if self.store_model_walkthrough:
						# store intermediate results
						self.model_walkthrough.append(tmp_tensor)

					in_channels = self.hidden_channels[layer - 1]
				out_channels = self.hidden_channels[layer]

				# initialize weights and biases:
				filter_shape = [self.filter_dims[layer][0], self.filter_dims[layer][1], in_channels, out_channels]

				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='conv{}_weights'.format(layer))
				b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name='conv{}_bias'.format(layer))

				self.conv_weights.append(W)
				self.conv_biases.append(b)

				# PREACTIVATION
				conv_preact = tf.add(tf.nn.conv2d(tmp_tensor, W, strides = self.strides[layer], padding='SAME'),  b, name='conv_{}_preactivation'.format(layer))

				# ACTIVATION
				if self.activation_function == 'relu':
					conv_act = tf.nn.relu(conv_preact, name='conv_{}_activation'.format(layer))

				else:
					conv_act = tf.nn.sigmoid(conv_preact, name='conv_{}_activation'.format(layer))

				# POOLING (2x2 max pooling)
				if self.use_max_pooling:
					pool_out = tf.nn.max_pool(conv_act, [1,2,2,1], [1,2,2,1], padding='SAME', name='max_pool_{}'.format(layer))
					tmp_tensor = pool_out

				else:
					tmp_tensor = conv_act

			self._encoding = tmp_tensor

		return self._encoding

	@property
	def logits(self):

		if self._logits is None:

			print('Initialize dense layers')

			encoding_shape = self.encoding.get_shape().as_list()

			encoding_dim = encoding_shape[1] * encoding_shape[2] * encoding_shape[3]

			tmp_tensor = tf.reshape(self.encoding, [-1, encoding_dim], name='last_conv_output_flattened')

			for d_ind, d in enumerate(self.dense_depths):

				layer_size = self.dense_depths[d_ind]

				weight_shape = [tmp_tensor.get_shape().as_list()[1], layer_size]
				bias_shape = [layer_size]

				print 'weight_shape: ', weight_shape

				W = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name='dense_{}_weights'.format(d_ind))
				b = tf.Variable(tf.constant(0.1, shape=bias_shape), name='dense_{}_bias'.format(d_ind))

				dense_preact 	= tf.add(tf.matmul(tmp_tensor, W), b, name='dense_{}_preact'.format(d_ind))
				
				if d_ind != len(self.dense_depths) - 1:

					if self.activation_function =='relu':
						dense_act = tf.nn.relu(dense_preact, name='dense_{}_act'.format(d_ind))
					
					else:
						dense_act = tf.nn.sigmoid(dense_preact, name='dense_{}_act'.format(d_ind))

					# add dropout regularization
					dense_act_drop = tf.nn.dropout(dense_act, self.keep_prob)

					tmp_tensor = dense_act_drop

				else:

					tmp_tensor = dense_preact

			self._logits = tmp_tensor

		return self._logits

	@property
	def prediction(self):

		if self._prediction is None:

			self._prediction = tf.nn.softmax(self.logits, name='softmax_prediction')

		return self._prediction


	@property
	def error(self):
		# returns the training error node (cross-entropy) used for the training and testing

		if self._error is None:
			print('initialize error')

			self._error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='cross-entropy_error'))

		return self._error

	@property
	def optimize(self):
		# returns the cross-entropy node we use for the optimization

		if self._optimize is None:
			print('initialize optimizer')

			# TODO: make step size modifiable
			step_size = 0.0001

			self._optimize = tf.train.AdamOptimizer(step_size).minimize(self.error)

		return self._optimize

	@property
	def accuracy(self):

		if self._accuracy is None:
			print('initialize accuracy')

			correct_prediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.target,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			self._accuracy = accuracy 


		return self._accuracy

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
