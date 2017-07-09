import tensorflow as tf 
import numpy as np

from lib.activations import l_relu

class CAE:
	# convolutional autoencoder 

	def __init__(self, data, filter_dims, hidden_channels, step_size = 0.0001, weight_init_stddev = 0.0001, weight_init_mean = 0.0001, initial_bias_value = 0.0001, strides = None, pooling_type = 'strided_conv', activation_function = 'sigmoid', tie_conv_weights = True, store_model_walkthrough = False, add_tensorboard_summary = True, relu_leak = 0.2, optimizer_type = 'gradient_descent', output_reconstruction_activation = 'sigmoid', regularization_factor = 0, decay_steps = None, decay_rate = 0.1, intialization_debug_output = True):

		if intialization_debug_output:
			print('-----------------------------------------')
			print('Initializing CAE:')
			print('->Architecture:')
			print('Filter sizes    		: {}'.format(' '.join(map(lambda x: '({},{})'.format(x[0], x[1]), filter_dims))))
			print('Hidden channels 		: {}'.format(', '.join(map(str, hidden_channels))))
			print('Pooling         		: {}'.format(pooling_type))
			print('Weight tying 		: {}'.format(tie_conv_weights))

			if activation_function != output_reconstruction_activation:
				if activation_function == 'lrelu':
					print('Hidden activation: lrelu (leak {})'.format(relu_leak))
				else:
					print('Hidden activations	: {}'.format(activation_function))
				print('Output activation    : {}'.format(output_reconstruction_activation))
			else:
				print('Activation function  : {}'.format(activation_function))


			print('->Training:')
			print('Weight init parameters: W~N({},{}), b = {}'.format(weight_init_mean, weight_init_stddev, initial_bias_value))
			print('Step size       : {}'.format(str(step_size)))
			print('Optimizer type  : {}'.format(optimizer_type))

			if regularization_factor > 0:
				print('Encoding regularization (L1) factor: {}'.format(regularization_factor))

			print('-----------------------------------------')


		self.data = data # we assume data in NHWC format 

		self.filter_dims 		= filter_dims 		# height and width of the conv kernels 	for each layer
		self.hidden_channels 	= hidden_channels	# number of feature maps 				for each layer
		
		self.strided_conv_strides 	= [1,2,2,1]
		self.std_strides 			= [1,1,1,1] 

		if str(strides) == 'None':
			if pooling_type == 'strided_conv':
				self.strides = [self.strided_conv_strides 	for filter in filter_dims]
			else:
				self.strides = [self.std_strides 			for filter in filter_dims]

		self.pooling_type 			= pooling_type
		self.activation_function	= activation_function
		self.relu_leak = relu_leak # only used if activation function is leaky relu

		self.hl_reconstruction_activation_function = self.activation_function

		self.output_reconstruction_activation	= output_reconstruction_activation

		self.tie_conv_weights = tie_conv_weights

		self.add_tensorboard_summary = add_tensorboard_summary
		self.track_gradients_in_tensorboard = True

		# init lists that will store weights and biases for the convolution operations
		self.conv_weights 	= []
		self.conv_biases	= []
		self.reconst_weights= []
		self.reconst_biases = []

		self.weight_init_stddev 	= weight_init_stddev
		self.weight_init_mean 		= weight_init_mean
		self.initial_bias_value 	= initial_bias_value

		self.step_size 				= step_size
		self.decay_steps 			= decay_steps
		self.decay_rate 			= decay_rate

		self.optimizer_type = optimizer_type

		# sparsity regularization (L1 norm):
		if regularization_factor < 0:
			regularization_factor = 0
		self.regularization_factor = regularization_factor
		self.regularization_terms = []

		# init list to store the shapes in the forward pass for the conv2d_transpose operations
		self.pre_conv_shapes = []

		self.store_model_walkthrough = store_model_walkthrough

		if self.store_model_walkthrough:
			print('MODEL WALKTHROUGH ENABLED')
			# initialize list that stores all the intermediate tensors in a forward path (probably high memory consumption, set flag to False if any problems occur)
			self.model_walkthrough = []

		else:
			print('model walkthrough disabled')

		# private attributes used by the properties
		self._encoding 				= None
		self._logit_reconstruction 	= None
		self._reconstruction		= None
		self._error					= None
		self._ce_error 				= None
		self._optimizer 			= None
		self._optimize				= None
		self._optimize_mse 			= None

		self._summaries = []


		print('Initializing conv autoencoder')
		with tf.name_scope('CAE'):
			self.optimize
			self.error

			if self.track_gradients_in_tensorboard:
				for i, conv_weight in enumerate(self.conv_weights):
					self._summaries.append(tf.summary.histogram('c-e loss gradient conv weight {}'.format(i), self.optimizer.compute_gradients(self.ce_error, [conv_weight])))
				for i, conv_bias in enumerate(self.conv_biases):
					self._summaries.append(tf.summary.histogram('c-e loss gradient conv bias {}'.format(i), self.optimizer.compute_gradients(self.ce_error, [conv_bias])))

		if self.add_tensorboard_summary:
			self.update_summaries()


		# initialize the weights and conv dictionaries used to store the weights
		# encoding:
		encoding_w_d = list(zip(['conv_W_{}'.format(i) for i,j in enumerate(self.conv_weights)], self.conv_weights))
		encoding_b_d = list(zip(['conv_b_{}'.format(i) for i,j in enumerate(self.conv_biases )], self.conv_biases))
		# decoding:
		reconst_w_d = list(zip(['reconstruction_W_{}'.format(i) for i,j in enumerate(self.reconst_weights)], self.reconst_weights))
		reconst_b_d = list(zip(['reconstruction_b_{}'.format(i) for i,j in enumerate(self.reconst_biases )], self.reconst_biases))

		self.encoding_variables_dict = dict(encoding_w_d + encoding_b_d)
		self.all_variables_dict = dict(encoding_w_d + encoding_b_d + reconst_w_d + reconst_b_d)

		print('Initialization finished')
			

	def add_summary(self, summary):
		self._summaries.append(summary)

	def update_summaries(self):
		self.merged = tf.summary.merge(self._summaries)

	@property
	def encoding(self):
		# returns the hidden layer representation (encoding) of the autoencoder

		if self._encoding is None:

			print('initialize encoding')

			tmp_tensor = self.data

			depth =  len(self.filter_dims)

			for layer in range(depth):

				# CONVOLUTION
				if layer == 0:
					in_channels = int(self.data.shape[3])
				else:

					if self.store_model_walkthrough:
						# store intermediate results
						self.model_walkthrough.append(tmp_tensor)

					in_channels = self.hidden_channels[layer - 1]
				out_channels = self.hidden_channels[layer]

				# print('init layer ', layer, 'conv', ' in-out:', in_channels, out_channels)

				# initialize weights and biases:
				filter_shape = [self.filter_dims[layer][0], self.filter_dims[layer][1], in_channels, out_channels]

				bias_shape = [out_channels]

				W = tf.Variable(tf.truncated_normal(filter_shape, mean=self.weight_init_mean, stddev=self.weight_init_stddev), name='conv{}_weights'.format(layer))
				b = tf.Variable(tf.constant(self.initial_bias_value, shape=bias_shape), name='conv{}_bias'.format(layer))

				self._summaries.append(tf.summary.histogram('ENCODING: layer {} weight'.format(layer), W))
				self._summaries.append(tf.summary.histogram('ENCODING: layer {} bias'.format(layer), b))

				if self.add_tensorboard_summary and layer == 0 and self.filter_dims[layer] != (1,1):
					# visualize first layer filters

					for fltr_indx in range(out_channels):
						self._summaries.append(tf.summary.image('first layer filter {}'.format(fltr_indx), W[None, :,:,:,fltr_indx]))

				self.conv_weights.append(W)
				self.conv_biases.append(b)

				self.pre_conv_shapes.append(tf.shape(tmp_tensor))

				# PREACTIVATION
				if self.pooling_type == 'strided_conv':
					strides = self.strided_conv_strides
				else:
					strides = self.std_strides

				conv_preact = tf.add(tf.nn.conv2d(tmp_tensor, W, strides = self.strides[layer], padding='SAME'),  b, name='conv_{}_preactivation'.format(layer))

				if self.add_tensorboard_summary:
					self._summaries.append(tf.summary.histogram('layer {} preactivations'.format(layer), conv_preact))

				# ACTIVATION
				if self.activation_function == 'relu':
					conv_act = tf.nn.relu(conv_preact, name='conv_{}_activation'.format(layer))

					alive_neurons = tf.count_nonzero(conv_act, name='active_neuron_number_{}'.format(layer))
					self._summaries.append(tf.summary.scalar('nb of relu neurons alive in layer {}'.format(layer), alive_neurons))

				elif self.activation_function == 'lrelu':
					# leaky relu to avoid the dying relu problem
					conv_act = l_relu(conv_preact, leak = self.relu_leak,  name='conv_{}_activation'.format(layer))

					alive_neurons = tf.count_nonzero(conv_act, name='active_neuron_number_{}'.format(layer))
					self._summaries.append(tf.summary.scalar('nb of relu neurons alive in layer {}'.format(layer), alive_neurons))

				elif self.activation_function == 'scaled_tanh':
					conv_act = tf.add(tf.nn.tanh(conv_preact) / 2, 0.5, name='conv_{}_activation'.format(layer))

				else:
					conv_act = tf.nn.sigmoid(conv_preact, name='conv_{}_activation'.format(layer))

				# POOLING (2x2 max pooling)
				if self.pooling_type == 'max_pooling':
					pool_out = tf.nn.max_pool(conv_act, [1,2,2,1], [1,2,2,1], padding='SAME', name='max_pool_{}'.format(layer))
					tmp_tensor = pool_out

				else:
					tmp_tensor = conv_act

			self._encoding = tmp_tensor

			# append L1 norm of hidden representation to enforce sparsity in the hidden representation
			encoding_norm = tf.norm(self._encoding, ord=1, name='L1_encoding_norm')
			self.regularization_terms.append(encoding_norm)
			self._summaries.append(tf.summary.scalar('encoding L1 norm', encoding_norm))

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.histogram('encoding histogram', self._encoding))

		return self._encoding

	@property
	def error(self):
		# returns the training error mean_squared error used for the training and testing

		if self._error is None:
			print('initialize error')

			mse = tf.reduce_mean(tf.squared_difference(self.reconstruction, self.data), name='mean-squared_error')

			self._error = mse 

			# add regularization to the total error
			for reg_term in self.regularization_terms:
				self._error += self.regularization_factor * reg_term

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.scalar('total_error_with_regularization', self._error))
				self._summaries.append(tf.summary.scalar('mean squared error', mse))

		return self._error

	@property
	def ce_error(self):
		# cross-entropy error
		if self._ce_error is None:
			if self.output_reconstruction_activation == 'scaled_tanh':

				ce_error = -tf.reduce_sum(self.data * tf.log(tf.clip_by_value(self.reconstruction, 1e-10, 1.0)), name='cross_entropy_on_scaled_tanh')

			else:

				ce_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.data, logits=self.logit_reconstruction, name='cross_entropy_error')

			self._ce_error = ce_error

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.scalar('avg cross entropy', tf.reduce_mean(self._ce_error)))

		return self._ce_error

	@property
	def optimizer(self):
		if self._optimizer is None:
			print('initialize {} optimizer'.format(self.optimizer_type))

			if self.optimizer_type == 'ada_grad':
				self._optimizer = tf.train.AdagradOptimizer(self.step_size)

			else:
				# default: gradient descent optimizer
				self._optimizer = tf.train.GradientDescentOptimizer(self.step_size)


		return self._optimizer

	@property
	def optimize_mse(self):
		if self._optimize_mse is None:

			optimizer = self.optimizer
			self._optimize_mse = optimizer.minimize(self.error)

		return self._optimize_mse

	@property
	def optimize(self):
		# returns the cross-entropy node we use for the optimization

		if self._optimize is None:
			print('initialize optimize call')

			ce_error  = self.ce_error
			optimizer = self.optimizer

			self._optimize = optimizer.minimize(ce_error)

		return self._optimize

	@property
	def logit_reconstruction(self):
		# returns the node containing the reconstruction before applying the sigmoid function 
		# the logit_reconstruction is separated to use the sigmoid_cross_entropy_with_logits function for the error

		if self._logit_reconstruction is None:
			print('initialize logit_reconstruction')

			tmp_tensor = self.encoding
			for layer in range(len(self.filter_dims))[::-1]:
				# go through the layers in reverse order to reconstruct the image

				#print('layer {} reconstruction'.format(layer))
				#print(self.conv_weights[layer])

				if self.store_model_walkthrough:
					# store intermediate results
					self.model_walkthrough.append(tmp_tensor)

				# CONV_TRANSPOSE (AND UPSAMPLING)
				if layer == 0:
					channels = int(self.data.shape[3])
				else:
					channels = self.hidden_channels[layer - 1]

					# print('channels', channels)


				if not self.tie_conv_weights: #  and layer == 0:
					# TODO: why layer == 0
					W = tf.Variable(tf.truncated_normal(tf.shape(self.conv_weights[layer]), mean=self.weight_init_mean, stddev=self.weight_init_stddev), name='conv{}_weights'.format(layer))
					self.reconst_weights.append(W)
					self._summaries.append(tf.summary.histogram('DECODING: layer {} weight'.format(layer), W))

				else:
					W =  self.conv_weights[layer]

				# init reconstruction bias
				bias_shape = [channels]
				c = tf.Variable(tf.constant(self.initial_bias_value, shape=bias_shape), name='reconstruction_bias_{}'.format(layer))
				self.reconst_biases.append(c)

				self._summaries.append(tf.summary.histogram('DECODING: layer {} bias'.format(layer), c))


				if self.pooling_type == 'max_pooling':
					# conv2d_transpose with upsampling
					upsampling_strides = [1,2,2,1]
					reconst_preact = tf.add( tf.nn.conv2d_transpose(tmp_tensor, W, self.pre_conv_shapes[layer], upsampling_strides), c, name='reconstruction_preact_{}'.format(layer))

				else:
					# conv2d_transpose without upsampling 
					if self.pooling_type == 'strided_conv':
						strides = self.strided_conv_strides
					else:
						strides = self.std_strides
					reconst_preact = tf.add( tf.nn.conv2d_transpose(tmp_tensor, W, self.pre_conv_shapes[layer], self.strides[layer]), c, name='reconstruction_preact_{}'.format(layer))

				self._summaries.append(tf.summary.histogram('layer {} reconstruction preactivations'.format(layer), reconst_preact))

				# ACTIVATION
				if layer > 0:
					# do not use the activation function in the last layer because we want the logits
					if self.hl_reconstruction_activation_function == 'relu':
						reconst_act = tf.nn.relu(reconst_preact, name='reconst_act')

						alive_neurons = tf.count_nonzero(reconst_act, name='alive_relus_in_reconstruction_layer_{}'.format(layer))
						self._summaries.append(tf.summary.scalar('alive neurons in reconstruction layer {}'.format(layer), alive_neurons))

					elif self.hl_reconstruction_activation_function == 'lrelu':
						reconst_act = l_relu(reconst_preact, leak = self.relu_leak, name='reconst_act')

					elif self.hl_reconstruction_activation_function == 'scaled_tanh':
						reconst_act = tf.add(tf.nn.tanh(reconst_preact) / 2, 0.5, name='reconst_act')

					else:
						reconst_act = tf.nn.sigmoid(reconst_preact ,name='reconst_act')

					tmp_tensor = reconst_act

				else:
					self._logit_reconstruction = reconst_preact

			if self.add_tensorboard_summary:
				self._summaries.append(tf.summary.histogram('logit reconstruction', self._logit_reconstruction))

		return self._logit_reconstruction

	@property
	def reconstruction(self):
		# returns the reconstruction node that contains the reconstruction of the input

		if self._reconstruction is None:
			print('initialize reconstruction')

			if self.output_reconstruction_activation == 'scaled_tanh':

				self._reconstruction = tf.add(tf.nn.tanh(self.logit_reconstruction) / 2, 0.5, name='scaled_tanh_reconstruction')

			else:

				self._reconstruction = tf.nn.sigmoid(self.logit_reconstruction, name='reconstruction')
		
		
			self._summaries.append(tf.summary.image('input', self.data))
			self._summaries.append(tf.summary.image('reconstruction', self._reconstruction))

			self._summaries.append(tf.summary.histogram('reconstruction_hist', self._reconstruction))

		return self._reconstruction


	def store_encoding_weights(self, sess, path_to_file):

		saver = tf.train.Saver(self.encoding_variables_dict)
		save_path = saver.save(sess, path_to_file)

		print('Saved encoding weights to {}'.format(save_path))

	def store_model_to_file(self, sess, path_to_file, step = None, saver = None):

		if saver is None:
			saver = tf.train.Saver(self.all_variables_dict)

		if step is None:
			save_path = saver.save(sess, path_to_file)
		else:
			save_path = saver.save(sess, path_to_file, global_step = step)

		# print('Model was saved in {}'.format(save_path))

		return save_path

	def load_model_from_file(self, sess, path_to_file):

		saver = tf.train.Saver(self.all_variables_dict)
		saver.restore(sess, path_to_file)

		print('Restored model from {}'.format(path_to_file))
