from pathlib import Path
import sys

for path in Path(__file__).resolve().parents:
    if path.name == 'nip-convnet':
        sys_path = str(path)
        break
sys.path.append(sys_path)
print(sys_path)

from configobj import ConfigObj
from collections import OrderedDict

class ConfigLoader:

	def __init__(self):
		self.configuration_dict = OrderedDict()

	def load_config_file(self, path=sys_path+'/configs/simple_cae_config.ini', config_version='CAE'):
		config = ConfigObj(path)
		local_dict = OrderedDict()
		valid_config=0
		if config_version.upper()=='CAE':
			valid_config=1
			local_dict['filter_dims'] =  list(zip(map(lambda x:int(x),config[config_version]['filter_dims_x']), map(lambda x:int(x),config[config_version]['filter_dims_y']))),
			local_dict['hidden_channels'] =  list(map(int, config[config_version]['hidden_channels'])),
			local_dict['pooling_type'] =  config[config_version]['pooling_type'],
			local_dict['strides'] =  config[config_version]['strides'],
			local_dict['activation_function'] =  config[config_version]['activation_function'],
			local_dict['relu_leak'] =  config[config_version]['relu_leak'],
			local_dict['error_function'] =  config[config_version]['error_function'],
			local_dict['optimizer_type'] =  config[config_version]['optimizer_type'],
			local_dict['output_reconstruction_activation'] =  config[config_version]['output_reconstruction_activation'],
			local_dict['weight_init_mean'] =  config[config_version]['weight_init_mean'],
			local_dict['weight_init_stddev'] =  config[config_version]['weight_init_stddev'],
			local_dict['initial_bias_value'] =  config[config_version]['initial_bias_value'],
			local_dict['batch_size'] =  config[config_version]['batch_size'],
			local_dict['max_iterations'] =  config[config_version]['max_iterations'],
			local_dict['chk_iterations'] =  config[config_version]['chk_iterations'],
			local_dict['step_size'] =  config[config_version]['step_size'],
			local_dict['tie_conv_weights'] =  config[config_version]['tie_conv_weights']


		elif config_version.upper()=='CNN':
			valid_config=1
			local_dict['filter_dims'] 			= list(zip(map(lambda x:int(x),config[config_version]['filter_dims_x']), map(lambda x:int(x),config[config_version]['filter_dims_y']))),
			local_dict['hidden_channels'] 		= list(map(int, config[config_version]['hidden_channels'])),
			local_dict['pooling_type'] 			= config[config_version]['pooling_type'],
			local_dict['strides'] 				= config[config_version]['strides'],
			local_dict['activation_function'] 	= config[config_version]['activation_function'],
			local_dict['dense_depths'] 			= list(map(int, config[config_version]['dense_depths'])),
			local_dict['batch_size'] 			= config[config_version]['batch_size'],
			local_dict['max_iterations'] 		= config[config_version]['max_iterations'],
			local_dict['chk_iterations'] 		= config[config_version]['chk_iterations'],
			local_dict['dropout_k_p'] 			= config[config_version]['dropout_k_p'],
			local_dict['fine_tuning_only'] 		= config[config_version]['fine_tuning_only'],
			local_dict['step_size'] 			= config[config_version]['step_size'],
			local_dict['decay_steps'] 			= config[config_version]['decay_steps'],
			local_dict['decay_rate'] 			= config[config_version]['decay_rate'],
			local_dict['weight_init_stddev'] 	= config[config_version]['weight_init_stddev'],
			local_dict['weight_init_mean']		= config[config_version]['weight_init_mean'],
			local_dict['initial_bias_value']	= config[config_version]['initial_bias_value'],

			local_dict['weight_decay_regularizer'] = config[config_version]['weight_decay_regularizer'],

		# LOAD CNN ARCHITECTURE
		elif config_version.upper()=='CNN_ARC':
			valid_config = 1

			local_dict['filter_dims'] 			=  list(zip(map(lambda x:int(x),config[config_version]['filter_dims_x']), map(lambda x:int(x),config[config_version]['filter_dims_y']))),
			local_dict['hidden_channels'] 		=  list(map(int, config[config_version]['hidden_channels'])),
			local_dict['pooling_type'] 			=  config[config_version]['pooling_type'],
			local_dict['activation_function'] 	=  config[config_version]['activation_function'],
			local_dict['dense_depths'] 			=  list(map(int, config[config_version]['dense_depths'])),

		# LOAD CNN TRAINING
		elif config_version.upper() == 'CNN_TR':
			valid_config = 1

			local_dict['batch_size'] 		=  config[config_version]['batch_size'],
			local_dict['max_iterations'] 	=  config[config_version]['max_iterations'],
			local_dict['chk_iterations'] 	=  config[config_version]['chk_iterations'],
			local_dict['dropout_k_p'] 		=  config[config_version]['dropout_k_p'],
			local_dict['fine_tuning_only'] 	=  config[config_version]['fine_tuning_only'],
			local_dict['step_size'] 		=  config[config_version]['step_size'],
			local_dict['decay_steps'] 		=  config[config_version]['decay_steps'],
			local_dict['decay_rate'] 		=  config[config_version]['decay_rate'],
			local_dict['test_set_bool'] 	=  config[config_version]['test_set_bool'],

		else:
			print("Unknown config version (loading)")

		if valid_config==1:
			for k, v in local_dict.items():
				self.configuration_dict[k] = v[0]

			print('Succesfully loaded config file, values are:')
			for k, v in self.configuration_dict.items():
				print(k, v)

	def store_config_file(self, path=sys_path+'/configs/custom_cae.ini', config_version='CAE'):

		# store content of current configuration_dict to file
		config = ConfigObj()
		config.filename = path
		config[config_version] = {}
		if config_version.upper()=='CAE':
			if self.configuration_dict and len(self.configuration_dict)==17:
				config[config_version]['filter_dims_x'] = [int(i[0]) for i in self.configuration_dict.get('filter_dims')]
				config[config_version]['filter_dims_y'] = [int(i[1]) for i in self.configuration_dict.get('filter_dims')]
				config[config_version]['hidden_channels'] = self.configuration_dict.get('hidden_channels')
				config[config_version]['pooling_type'] = self.configuration_dict.get('pooling_type')
				config[config_version]['strides'] = self.configuration_dict.get('strides')
				config[config_version]['activation_function'] = self.configuration_dict.get('activation_function')
				config[config_version]['relu_leak'] = self.configuration_dict.get('relu_leak')
				config[config_version]['error_function'] = self.configuration_dict.get('error_function')
				config[config_version]['optimizer_type'] = self.configuration_dict.get('optimizer_type')
				config[config_version]['output_reconstruction_activation'] = self.configuration_dict.get('output_reconstruction_activation')
				config[config_version]['weight_init_mean'] = self.configuration_dict.get('weight_init_mean')
				config[config_version]['weight_init_stddev'] = self.configuration_dict.get('weight_init_stddev')
				config[config_version]['initial_bias_value'] = self.configuration_dict.get('initial_bias_value')
				config[config_version]['batch_size'] = self.configuration_dict.get('batch_size')
				config[config_version]['max_iterations'] = self.configuration_dict.get('max_iterations')
				config[config_version]['chk_iterations'] = self.configuration_dict.get('chk_iterations')
				config[config_version]['step_size'] = self.configuration_dict.get('step_size')
				config[config_version]['tie_conv_weights'] = self.configuration_dict.get('tie_conv_weights')
				print(config)
				config.write()
			else:
				print('# of configurations need to be 17!')

		elif config_version.upper()=='CNN':
			if self.configuration_dict and len(self.configuration_dict)==18:
				config[config_version]['filter_dims_x'] 		= [int(i[0]) for i in self.configuration_dict.get('filter_dims')]
				config[config_version]['filter_dims_y'] 		= [int(i[1]) for i in self.configuration_dict.get('filter_dims')]
				config[config_version]['hidden_channels'] 		= self.configuration_dict.get('hidden_channels')
				config[config_version]['pooling_type'] 			= self.configuration_dict.get('pooling_type')
				config[config_version]['strides'] 				= self.configuration_dict.get('strides')
				config[config_version]['activation_function'] 	= self.configuration_dict.get('activation_function')
				config[config_version]['dense_depths'] 			= self.configuration_dict.get('dense_depths')
				config[config_version]['batch_size'] 			= self.configuration_dict.get('batch_size')
				config[config_version]['max_iterations'] 		= self.configuration_dict.get('max_iterations')
				config[config_version]['chk_iterations'] 		= self.configuration_dict.get('chk_iterations')
				config[config_version]['dropout_k_p'] 			= self.configuration_dict.get('dropout_k_p')
				config[config_version]['fine_tuning_only'] 		= self.configuration_dict.get('fine_tuning_only')
				config[config_version]['step_size'] 			= self.configuration_dict.get('step_size')
				config[config_version]['decay_steps'] 			= self.configuration_dict.get('decay_steps')
				config[config_version]['decay_rate'] 			= self.configuration_dict.get('decay_rate')
				config[config_version]['weight_init_stddev']	= self.configuration_dict.get('weight_init_stddev')
				config[config_version]['weight_init_mean'] 		= self.configuration_dict.get('weight_init_mean')
				config[config_version]['initial_bias_value']	= self.configuration_dict.get('initial_bias_value')

				config[config_version]['weight_decay_regularizer'] = self.configuration_dict.get('weight_decay_regularizer')

				config.write()
				print(config)
			else:
				print('# of configurations need to be 14!')

		# STORE CNN ARCHITECTURE
		elif config_version.upper() == 'CNN_ARC':

			if self.configuration_dict: # TODO: split into two dictionaries and do a lenght check? # and len(self.configuration_dict)==5:
				config[config_version]['filter_dims_x'] = [int(i[0]) for i in self.configuration_dict.get('filter_dims')]
				config[config_version]['filter_dims_y'] = [int(i[1]) for i in self.configuration_dict.get('filter_dims')]
				config[config_version]['hidden_channels'] = self.configuration_dict.get('hidden_channels')
				config[config_version]['pooling_type'] = self.configuration_dict.get('pooling_type')
				config[config_version]['activation_function'] = self.configuration_dict.get('activation_function')
				config[config_version]['dense_depths'] = self.configuration_dict.get('dense_depths')

				config.write()
				print('Wrote CNN architecture into config file')
				print(config)
			else:
				print('Error: configuration dict empty! (occured while storing cnn architecture')

		# STORE CNN TRAINING 
		elif config_version.upper() == 'CNN_TR':

			if self.configuration_dict: # TODO: split into two dicts?? Same as above
				config[config_version]['batch_size'] 		= self.configuration_dict.get('batch_size')
				config[config_version]['max_iterations'] 	= self.configuration_dict.get('max_iterations')
				config[config_version]['chk_iterations'] 	= self.configuration_dict.get('chk_iterations')
				config[config_version]['dropout_k_p'] 		= self.configuration_dict.get('dropout_k_p')
				config[config_version]['fine_tuning_only'] 	= self.configuration_dict.get('fine_tuning_only')
				config[config_version]['step_size'] 		= self.configuration_dict.get('step_size')
				config[config_version]['decay_steps'] 		= self.configuration_dict.get('decay_steps')
				config[config_version]['decay_rate'] 		= self.configuration_dict.get('decay_rate')
				config[config_version]['test_set_bool']		= self.configuration_dict.get('test_set_bool')
			
				config.write()
				print('Wrote CNN training parameters into config file')
				print(config)
			else:
				print('Error: configuration dict empty! (occured while storing cnn training parameters)')


		else:
			print("Unknown config version (storing)")

# config = ConfigLoader()
# print(config.configuration_dict)
# config.load_config_file('simple_cnn_config.ini', 'CNN')
# print(config.configuration_dict)
# config.store_config_file('custom_cnn.ini', 'CNN')
