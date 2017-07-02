import configs.config as config
import os

def main():

	assert(test_cnn_config() == True)
	assert(test_cae_config() == True)

def test_cae_config():

	# TODO: reproduce the same functionality as in test_cnn_config() for the cae
	config_loader 	= config.ConfigLoader()
	# store all variables to a file, reload them and make sure they are still the same
	filter_dims 	= [(5,5), (5,5)]
	hidden_channels = [100, 150]
	pooling_type 	= 'strided_conv'
	strides = 'vier' # other strides should not work yet
	activation_function = 'relu'
	relu_leak = 0.2 # only for leaky relus
	error_function 	= 'mse' 					# default is cross-entropy
	optimizer_type 	= 'gradient_descent' 		# default is gradient descent
	output_reconstruction_activation = 'scaled_tanh'
	weight_init_mean 	= 0.001
	weight_init_stddev 	= 0.05
	initial_bias_value  = 0.001
	batch_size 		= 128
	max_iterations 	= 100001
	chk_iterations  = 500
	step_size 		= 0.1
	tie_conv_weights = True

	# store to config dict:
	config_dict = {}
	config_dict['filter_dims'] 			= filter_dims
	config_dict['hidden_channels'] 		= hidden_channels
	config_dict['pooling_type']  		= pooling_type
	config_dict['strides'] 				= strides
	config_dict['activation_function'] 	= activation_function
	config_dict['relu_leak'] 			= relu_leak
	config_dict['error_function'] 		= error_function
	config_dict['optimizer_type'] 		= optimizer_type
	config_dict['output_reconstruction_activation'] = output_reconstruction_activation
	config_dict['weight_init_mean'] 	= weight_init_mean
	config_dict['weight_init_stddev'] 	= weight_init_stddev
	config_dict['initial_bias_value'] 	= initial_bias_value
	config_dict['batch_size'] 			= batch_size
	config_dict['max_iterations'] 		= max_iterations
	config_dict['chk_iterations'] 		= chk_iterations
	config_dict['step_size'] 			= step_size
	config_dict['tie_conv_weights'] 	= int(tie_conv_weights)

	config_loader.configuration_dict 	= config_dict

	os.mkdir('tmp')

	config_loader.store_config_file('tmp/cae_config.ini', 'CAE')


	# RESTORING

	config_loader_2 = config.ConfigLoader()
	config_loader_2.load_config_file('tmp/cae_config.ini', 'CAE')

	# delete temporary folder
	os.remove('tmp/cae_config.ini')
	os.rmdir('tmp')

	config_dict = config_loader_2.configuration_dict

	# restore variables from file
	filter_dims_l 					= config_dict['filter_dims']
	hidden_channels_l 				= config_dict['hidden_channels']
	pooling_type_l  				= config_dict['pooling_type']
	strides_l 						= config_dict['strides']
	activation_function_l 			= config_dict['activation_function']
	relu_leak_l 					= float(config_dict['relu_leak'])
	error_function_l				= config_dict['error_function']
	optimizer_type_l 				= config_dict['optimizer_type']
	output_reconstruction_activation_l 	= config_dict['output_reconstruction_activation']
	weight_init_mean_l 				= float(config_dict['weight_init_mean'])
	weight_init_stddev_l 			= float(config_dict['weight_init_stddev'])
	initial_bias_value_l 			= float(config_dict['initial_bias_value'])
	batch_size_l 					= int(config_dict['batch_size'])
	max_iterations_l				= int(config_dict['max_iterations'])
	chk_iterations_l 				= int(config_dict['chk_iterations'])
	step_size_l						= float(config_dict['step_size'])
	tie_conv_weights_l				= bool(int(config_dict['tie_conv_weights']))

	# test
	assert(filter_dims 		== filter_dims_l)
	assert(hidden_channels 	== hidden_channels_l)
	assert(pooling_type 	== pooling_type_l)
	assert(strides 			== strides_l)
	assert(activation_function == activation_function_l)
	assert(relu_leak == relu_leak_l)
	assert(error_function == error_function_l)
	assert(optimizer_type == optimizer_type_l)
	assert(output_reconstruction_activation == output_reconstruction_activation_l)
	assert(weight_init_mean == weight_init_mean_l)
	assert(weight_init_stddev == weight_init_stddev_l)
	assert(initial_bias_value == initial_bias_value_l)
	assert(batch_size == batch_size_l)
	assert(max_iterations == max_iterations_l)
	assert(chk_iterations == chk_iterations_l)
	assert(step_size == step_size_l)
	assert(tie_conv_weights == tie_conv_weights_l)

	return True


def test_cnn_config():

	# SAVING
	config_loader 	= config.ConfigLoader()
	# example config parameters
	filter_dims 		= [(5,3), (5,5), (1,5)]
	hidden_channels 	= [64, 64, 77] 
	pooling_type  		= 'strided_conv' 
	strides 			= 'vier'
	activation_function = 'relu'
	dense_depths 		= [384, 192]
	batch_size 			= 128
	max_iterations		= 1001
	chk_iterations 		= 100
	dropout_k_p			= 0.5
	step_size 			= 0.1
	fine_tuning_only 	= False

	# store to config dict:
	config_dict = {}
	config_dict['filter_dims'] 			= filter_dims
	config_dict['hidden_channels'] 		= hidden_channels
	config_dict['pooling_type']  		= pooling_type
	config_dict['strides'] 				= strides
	config_dict['activation_function'] 	= activation_function
	config_dict['dense_depths'] 		= dense_depths
	config_dict['batch_size'] 			= batch_size
	config_dict['max_iterations'] 		= max_iterations
	config_dict['chk_iterations'] 		= chk_iterations
	config_dict['dropout_k_p'] 			= dropout_k_p 
	config_dict['fine_tuning_only'] 	= int(fine_tuning_only)
	config_dict['step_size'] 			= step_size
	config_loader.configuration_dict = config_dict

	os.mkdir('tmp')

	config_loader.store_config_file('tmp/cnn_config.ini', 'CNN')


	# RESTORING

	config_loader_2 = config.ConfigLoader()
	config_loader_2.load_config_file('tmp/cnn_config.ini', 'CNN')

	# delete temporary folder
	os.remove('tmp/cnn_config.ini')
	os.rmdir('tmp')

	config_dict = config_loader_2.configuration_dict

	# restore variables from file
	filter_dims_l 			= config_dict['filter_dims']
	hidden_channels_l 		= config_dict['hidden_channels'] 
	pooling_type_l  			= config_dict['pooling_type'] 
	strides_l 				= config_dict['strides']
	activation_function_l 	= config_dict['activation_function']
	dense_depths_l 			= config_dict['dense_depths']
	batch_size_l 				= int(config_dict['batch_size'])
	max_iterations_l			= int(config_dict['max_iterations'])
	chk_iterations_l 			= int(config_dict['chk_iterations'])
	dropout_k_p_l				= float(config_dict['dropout_k_p']) 
	fine_tuning_only_l 		= bool(int(config_dict['fine_tuning_only']))
	step_size_l				= float(config_dict['step_size'])


	# test 
	assert(filter_dims 		== filter_dims_l)
	assert(hidden_channels 	== hidden_channels_l)
	assert(pooling_type 	== pooling_type_l)
	assert(strides 			== strides_l)
	assert(activation_function == activation_function_l)
	assert(dense_depths == dense_depths_l)
	assert(batch_size == batch_size_l)
	assert(max_iterations == max_iterations_l)
	assert(chk_iterations == chk_iterations_l)
	assert(dropout_k_p == dropout_k_p_l)
	assert(fine_tuning_only == fine_tuning_only_l) 
	assert(step_size == step_size_l)

	return True



if __name__ == '__main__':
	main()
