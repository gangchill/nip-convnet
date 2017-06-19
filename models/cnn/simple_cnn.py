import tensorflow as tf
import csv, os
import collections

class SCNN: 
	# simple convolutional neural network (same structure as cae with added fully-connected layers)

	def __init__(self, data, target, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type = 'strided_conv', activation_function = 'sigmoid', add_tensorboard_summary = True, scope_name='CNN', one_hot_labels = True):

		# TODO:
		# 	- add assertion that test whether filter_dims, hidden_channels and strides have the right dimensions
		# 	  (the upsampling_strides need to be adapted for the upsampling)
		# 	- verify the bias treatment (currently: the same bias for every pixel in a given feature map)

		self.data = data # we assume data in NHWC format 
		self.target = target	# labels (assumed to be in one-hot encoding)

		self.one_hot_labels = one_hot_labels

		self.keep_prob = keep_prob # input probability for dropout regularization (set to 1.0 for inference)

		# filter_dims, out_channels and strides (if specified) are lists containing the specifications for each of the consecutive layers
		# the choice of mac pooling and activation function is used for the whole network (the last activation function is always a sigmoid)

		self.filter_dims 		= filter_dims 		# height and width of the conv kernels 	for each layer
		self.hidden_channels 	= hidden_channels	# number of feature maps 				for each layer

		if pooling_type == 'strided_conv':
			# use strides as downsampling
			self.strides = [[1,2,2,1] for filter in filter_dims]
		else:
			# default value
			self.strides = [[1,1,1,1] for filter in filter_dims]

		# layer sizes for the dense layers (decision making)
		self.dense_depths = []
		self.dense_depths.extend(dense_depths)

		# list that will store all dense layer variables for fine tuning 
		self.dense_layer_variables = []

		# add a dense shape for the readout layer
		if one_hot_labels:
			self.dense_depths.append(self.target.get_shape().as_list()[1])
		else:
			self.dense_depths.append(10) # hardcoded for cifar-10

		self.pooling_type 			= pooling_type
		self.activation_function	= activation_function

		# init lists that will store weights and biases for the convolution operations
		self.conv_weights 	= []
		self.conv_biases	= []


		self.add_tensorboard_summary = add_tensorboard_summary

		self.conv_weights_dict = {}
		self.conv_biases_dict = {}
		self.conv_merged_dict = {}

		# private attributes used by the properties
		self._encoding 		= None
		self._logits 		= None
		self._prediction 	= None
		self._error			= None
		self._optimize 		= None
		self._optimize_dense_layers = None
		self._accuracy		= None

		self.weight_init_stddev 	= 0.05 # 0.000015
		self.weight_init_mean 		= 0.   # 0.0001
		self.initial_bias_value 	= 0. # 0.0001
		self.step_size 				= 0.0001 # 0.0001

		self._summaries = []
		

		print('Initializing simple CNN')
		with tf.name_scope(scope_name):
			self.optimize
			self.optimize_dense_layers

		with tf.name_scope('accuracy_' + scope_name):
			self.accuracy

		if self.add_tensorboard_summary:
			self.merged = tf.summary.merge(self._summaries)

		print('...finished initialization')

	@property
	def encoding(self):
		# returns the hidden layer representation (encoding) of the autoencoder

		print('encoding called')

		if self._encoding is None:

			print('initialize encoding')

			tmp_tensor = self.data

			for layer in range(len(self.filter_dims)):

				# CONVOLUTION
				if layer == 0:
					in_channels = int(self.data.shape[3])
				else:

					in_channels = self.hidden_channels[layer - 1]
				out_channels = self.hidden_channels[layer]

				# initialize weights and biases:
				filter_shape = [self.filter_dims[layer][0], self.filter_dims[layer][1], in_channels, out_channels]

				W = tf.Variable(tf.truncated_normal(filter_shape, mean=self.weight_init_mean, stddev=self.weight_init_stddev), name='conv{}_weights'.format(layer))
				b = tf.Variable(tf.constant(self.initial_bias_value, shape=[out_channels]), name='conv{}_bias'.format(layer))

				if self.add_tensorboard_summary and layer == 0:
					# visualize first layer filters

					for fltr_indx in range(out_channels):
						self._summaries.append(tf.summary.image('first layer filter {}'.format(fltr_indx), tf.reduce_mean(W, 2)[None, :,:,fltr_indx, None]))


				self.conv_weights.append(W)
				self.conv_biases.append(b)

				# self.pre_conv_shapes.append(tf.shape(tmp_tensor))

				# PREACTIVATION
				conv_preact = tf.add(tf.nn.conv2d(tmp_tensor, W, strides = self.strides[layer], padding='SAME'),  b, name='conv_{}_preactivation'.format(layer))

				self._summaries.append(tf.summary.histogram('layer {} preactivations'.format(layer), conv_preact))

				# ACTIVATION
				if self.activation_function == 'relu':
					conv_act = tf.nn.relu(conv_preact, name='conv_{}_activation'.format(layer))

					alive_neurons = tf.count_nonzero(conv_act, name='active_neuron_number_{}'.format(layer))
					self._summaries.append(tf.summary.scalar('nb of relu neurons alive in layer {}'.format(layer), alive_neurons))

				else:
					conv_act = tf.nn.sigmoid(conv_preact, name='conv_{}_activation'.format(layer))

				# POOLING (2x2 max pooling)
				if self.pooling_type == 'max_pooling':
					pool_out = tf.nn.max_pool(conv_act, [1,2,2,1], [1,2,2,1], padding='SAME', name='max_pool_{}'.format(layer))
					tmp_tensor = pool_out

				elif self.pooling_type == 'max_pooling_k3':
					# max pooling with larger kernel (as in AlexNet)
					pool_out = tf.nn.max_pool(conv_act, [1,3,3,1], [1,2,2,1], padding='SAME', name='max_pool_{}'.format(layer))
					tmp_tensor = pool_out

				else:
					tmp_tensor = conv_act

			self._encoding = tmp_tensor

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.histogram('encoding histogram', self._encoding))

		return self._encoding

	@property
	def logits(self):

		print('logits called')

		if self._logits is None:

			print('Initialize dense layers')

			encoding_shape = self.encoding.get_shape().as_list()

			encoding_dim = encoding_shape[1] * encoding_shape[2] * encoding_shape[3]

			tmp_tensor = tf.reshape(self.encoding, [-1, encoding_dim], name='last_conv_output_flattened')

			for d_ind, d in enumerate(self.dense_depths):

				layer_size = self.dense_depths[d_ind]

				weight_shape = [tmp_tensor.get_shape().as_list()[1], layer_size]
				bias_shape = [layer_size]

				print('weight_shape: ', weight_shape)

				W = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name='dense_{}_weights'.format(d_ind))
				b = tf.Variable(tf.constant(0.1, shape=bias_shape), name='dense_{}_bias'.format(d_ind))

				# save dense variables to list to use them in fine-tuning
				self.dense_layer_variables.append(W)
				self.dense_layer_variables.append(b)

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

		print('error called')

		if self._error is None:
			print('initialize error')

			if self.one_hot_labels:
				self._error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='cross-entropy_error'))
			else:
				self._error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='cross-entropy_error'))

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.scalar('cross entropy error', self._error))

		return self._error

	@property
	def optimize(self):
		# minimize the error function tuning all variables

		if self._optimize is None:

			print('init optimization')

			self._optimize = tf.train.AdamOptimizer(self.step_size).minimize(self.error)

		return self._optimize

	@property
	def optimize_dense_layers(self):
		# minimize the error function tuning only the variables of the dense layers 

		if self._optimize_dense_layers is None:

			print('init dense layer optimization')

			self._optimize_dense_layers = tf.train.AdamOptimizer(self.step_size).minimize(self.error, var_list = self.dense_layer_variables)

		return self._optimize_dense_layers

	@property
	def accuracy(self):

		if self._accuracy is None:
			print('initialize accuracy')

			if self.one_hot_labels:
				correct_prediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.target,1))
			else:
				correct_prediction = tf.equal(tf.argmax(self.prediction,1), self.target)

			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			self._accuracy = accuracy 

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.scalar('accuracy', self._accuracy))


		return self._accuracy

	def merge_two_dicts(x, y):
		"""Given two dicts, merge them into a new dict as a shallow copy."""
		z = x.copy()
		z.update(y)
		return z

	# store weights and biases separately in two simple dicts, and do a simple merge to create a new dict of weights and biases
	def store_model_to_file_v1(self, sess, path_to_file):

		if len(self.conv_weights) > 0:
			weights = dict(zip(['conv_w_' + str(i) for i in enumerate(self.conv_weights)], self.conv_weights))
			if weights is not None:
				self.conv_weights_dict = weights

		if len(self.conv_biases) > 0:
			biases = dict(zip(['conv_b_' + str(i) for i in enumerate(self.conv_biases)], self.conv_biases))
			if biases is not None:
				self.conv_biases_dict = biases

		if self.conv_weights_dict is not None and self.conv_biases_dict is not None:
			self.conv_merged_dict=self.merge_two_dicts(self.conv_weights_dict, self.conv_biases_dict)

		saver = tf.train.Saver(self.conv_merged_dict)
		save_path = saver.save(sess, path_to_file)
		return

	# store weights and biases separately in two ordered dicts, and do a merge on same key to create a new dict of pairs of weights and biases
	def store_model_to_file_v2(self, sess, path_to_file):

		if len(self.conv_weights) > 0:
			weights = collections.OrderedDict(zip(['conv_wb_' + str(i) for i in enumerate(self.conv_weights)], self.conv_weights))
			if weights is not None:
				self.conv_weights_dict = weights

		if len(self.conv_biases) > 0:
			biases = collections.OrderedDict(zip(['conv_wb_' + str(i) for i in enumerate(self.conv_biases)], self.conv_biases))
			if biases is not None:
				self.conv_biases_dict = biases

		if self.conv_weights_dict is not None and self.conv_biases_dict is not None:
			dicts = self.conv_weights_dict, self.conv_biases_dict
			self.conv_merged_dict={k:[d.get(k) for d in dicts] for k in {k for d in dicts for k in d}}

		saver = tf.train.Saver(self.conv_merged_dict)
		save_path = saver.save(sess, path_to_file)
		return


	def store_model_to_file(self, sess, path_to_file):
		# store the whole model to file

		saver = tf.train.Saver()
		save_path = saver.save(sess, path_to_file)

		print('Model was saved in {}'.format(save_path))

		return save_path

	def load_model_from_file(self, sess, path_to_file):


		saver = tf.train.Saver()
		saver.restore(sess, path_to_file)

		print('Restored model from {}'.format(path_to_file))


	def load_encoding_weights(self, sess, path_to_file):

		# load the encoding (feature extraction) weights from a given file (init encoding with the weights learned by a DCAE)
		# similar to the CAE.store_encoding_weights() function
		
		conv_w_d = zip(['conv_W_{}'.format(i) for i,j in enumerate(self.conv_weights)], self.conv_weights)
		conv_b_d = zip(['conv_b_{}'.format(i) for i,j in enumerate(self.conv_biases )], self.conv_biases)

		conv_variable_dict = dict(conv_w_d + conv_b_d)

		saver = tf.train.Saver(conv_variable_dict)
		saver.restore(sess, path_to_file)

		print('Restored model from {}'.format(path_to_file))

