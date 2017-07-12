import tensorflow as tf
import csv, os
import collections

class CNN: 
	# convolutional neural network (same structure as cae with added fully-connected layers)

	def __init__(self, data, target, keep_prob, filter_dims, hidden_channels, dense_depths, pooling_type = 'strided_conv', activation_function = 'sigmoid', add_tensorboard_summary = True, scope_name='CNN', one_hot_labels = True, step_size = 0.1, decay_steps = 10000, decay_rate = 0.1, weight_decay_regularizer = 0, weight_init_stddev = 0.2, weight_init_mean = 0, initial_bias_value = 0):

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
		self.dense_weights	= []
		self.dense_biases 	= []
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

		self.dense_weights = []

		self.add_tensorboard_summary 		= add_tensorboard_summary
		self.track_gradients_in_tensorboard = True

		# private attributes used by the properties
		self._encoding 		= None
		self._logits 		= None
		self._prediction 	= None
		self._error			= None
		self._optimizer  	= None
		self._optimize 		= None
		self._optimize_dense_layers = None
		self._accuracy		= None

		self.weight_init_stddev 	= weight_init_stddev # 0.2 		
		self.weight_init_mean 		= weight_init_mean   # 0.   	
		self.initial_bias_value 	= initial_bias_value # 0. 
		self.step_size 				= step_size			 # 0.0001

		self.dense_stddev = 0.001
		self.dense_mean   = 0.0
		self.dense_b_init = 0.0001

		self.decay_steps = decay_steps
		self.decay_rate = decay_rate

		self._losses = []

		self._summaries = []


		self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
		self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1)
		self.global_step_setter_input 	= tf.placeholder(tf.int32, shape=[])
		self.set_global_step_op 		= tf.assign(self.global_step, self.global_step_setter_input)

		if weight_decay_regularizer == 0 and type(weight_decay_regularizer) == int:
		    print('WARNING: weight_decay_regularizer was int 0, set to float')
		    weight_decay_regularizer = 0.

		self.decay_factor = weight_decay_regularizer

		print('decay factor is {}'.format(self.decay_factor))
		self.decay_terms = []

		print('Initializing simple CNN')
		with tf.name_scope(scope_name):

			self.logits # needs to be called first for self.decay_terms to be filled
			if self.decay_factor > 0:
			    self.decay_sum = tf.add_n(self.decay_terms) / len(self.decay_terms)
			else:
			    self.decay_sum = 0

			self._summaries.append(tf.summary.scalar('decay_term_sum', self.decay_sum))

			self.optimize
			self.optimize_dense_layers

			if self.track_gradients_in_tensorboard:
				for i, conv_weight in enumerate(self.conv_weights):
					self._summaries.append(tf.summary.histogram('c-e loss gradient conv weight {}'.format(i), self.optimizer.compute_gradients(self.error, [conv_weight])))
				for i, conv_bias in enumerate(self.conv_biases):
					self._summaries.append(tf.summary.histogram('c-e loss gradient conv bias {}'.format(i), self.optimizer.compute_gradients(self.error, [conv_bias])))


		with tf.name_scope('accuracy_' + scope_name):
			self.accuracy

		if self.add_tensorboard_summary:
			self.update_summaries()


		# initialize the weights and conv dictionaries used to store the weights
		# encoding:
		encoding_w_d = list(zip(['conv_W_{}'.format(i) for i,j in enumerate(self.conv_weights)], self.conv_weights))
		encoding_b_d = list(zip(['conv_b_{}'.format(i) for i,j in enumerate(self.conv_biases )], self.conv_biases ))

		dense_w_d = list(zip(['dense_W_{}'.format(i) for i,j in enumerate(self.dense_weights)], self.dense_weights))
		dense_b_d = list(zip(['dense_b_{}'.format(i) for i,j in enumerate(self.dense_biases )], self.dense_biases ))

		# g_s_d = {'global_step': self.global_step}

		self.encoding_variables_dict = dict(encoding_w_d + encoding_b_d)
		self.all_variables_dict = dict(encoding_w_d + encoding_b_d + dense_w_d + dense_b_d)

		# print(self.encoding_variables_dict)

		print('...finished initialization')


	def update_summaries(self):
		self.merged = tf.summary.merge(self._summaries)

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

				if self.decay_factor > 0:
					self.decay_terms.append(tf.nn.l2_loss(W))

				if self.add_tensorboard_summary and layer == 0:
					# visualize first layer filters

					for fltr_indx in range(out_channels):
						self._summaries.append(tf.summary.image('first layer filter {}'.format(fltr_indx), W[None, :,:,:,fltr_indx]))


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
				elif self.activation_function == 'scaled_tanh':
					conv_act = tf.add(tf.nn.tanh(conv_preact) / 2, 0.5, 'conv_{}_activation'.format(layer))
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


				print('ADD DENSE LAYER OF DEPTH {}'.format(self.dense_depths[d_ind]))

				layer_size = self.dense_depths[d_ind]

				weight_shape = [tmp_tensor.get_shape().as_list()[1], layer_size]
				bias_shape = [layer_size]

				# print('weight_shape: ', weight_shape)

				W = tf.Variable(tf.truncated_normal(weight_shape, mean=self.dense_mean,  stddev=self.dense_stddev), name='dense_{}_weights'.format(d_ind))
				b = tf.Variable(tf.constant(self.dense_b_init, shape=bias_shape), name='dense_{}_bias'.format(d_ind))

				if self.decay_factor > 0:
					self.decay_terms.append(tf.nn.l2_loss(W))

				# save dense variables to list to use them in fine-tuning
				self.dense_weights.append(W)
				self.dense_biases.append(b)
				self.dense_layer_variables.append(W)
				self.dense_layer_variables.append(b)

				dense_preact 	= tf.add(tf.matmul(tmp_tensor, W), b, name='dense_{}_preact'.format(d_ind))
		
			
	
				self._summaries.append(tf.summary.histogram('dense_layer_{}_preact'.format(d_ind), dense_preact))
	
				if d_ind != len(self.dense_depths) - 1:

					if self.activation_function =='relu':
						dense_act = tf.nn.relu(dense_preact, name='dense_{}_act'.format(d_ind))
					
					elif self.activation_function == 'scaled_tanh':
						dense_act = tf.add(tf.nn.tanh(dense_preact) / 2, 0.5, 'dense_{}_act'.format(d_ind))

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
				ce_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='cross-entropy_error'))
			else:
				ce_error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='cross-entropy_error'))


			self._error = ce_error + self.decay_factor * self.decay_sum

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.scalar('cross entropy error', ce_error))
				if self.decay_factor > 0:
					self._summaries.append(tf.summary.scalar('total error (incl weight decay)', self._error))

		return self._error

	@property
	def optimizer(self):

		if self._optimizer is None:
			print('init optimizer')

			if self.decay_steps:

				print('learning rate decay enabled')

				# Decay the learning rate exponentially based on the number of steps.
				lr = tf.train.exponential_decay(self.step_size,
						self.global_step,
						self.decay_steps, # 10000
						self.decay_rate, # 0.1
						staircase=True)
			else:

				lr = self.step_size

				print('learning rate decay disabled, lr = {}'.format(lr))


			self._summaries.append(tf.summary.scalar('learning_rate', lr))
			self._optimizer = tf.train.GradientDescentOptimizer(lr)

		return self._optimizer

	@property
	def optimize(self):
		# minimize the error function tuning all variables

		if self._optimize is None:

			print('init optimization')

			self._optimize = self.optimizer.minimize(self.error)

		self.global_step += 1
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

			#f self.add_tensorboard_summary:
			#	self._summaries.append(tf.summary.scalar('accuracy', self._accuracy))

		return self._accuracy

	def store_model_to_file(self, sess, path_to_file, step = None, saver = None):

		if saver == None:
			saver = tf.train.Saver(self.all_variables_dict)

		if step is None:
			save_path = saver.save(sess, path_to_file)
		else:
			save_path = saver.save(sess, path_to_file, global_step = step)

		print('Model was saved in {}'.format(save_path))

		return save_path

	def load_model_from_file(self, sess, path_to_file):


		saver = tf.train.Saver(self.all_variables_dict)
		saver.restore(sess, path_to_file)

		print('Restored model from {}'.format(path_to_file))


	def load_encoding_weights(self, sess, path_to_file):

		# load the encoding (feature extraction) weights from a given file (init encoding with the weights learned by a DCAE)
		# similar to the CAE.store_encoding_weights() function

		saver = tf.train.Saver(self.encoding_variables_dict)
		saver.restore(sess, path_to_file)

		print('Restored model from {}'.format(path_to_file))

