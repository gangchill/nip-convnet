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


	def load_config_file(self, path=sys_path+'/configs/config.ini', config_version='cnn_default'):
		config = ConfigObj(path)
		local_dict = OrderedDict()
		# load config information from file and store in configuarion_dict
		local_dict['filter_dims'] 			= list(zip(list(map(int, config[config_version]['filter_dims_x'])), list(map(int,config[config_version]['filter_dims_y'])))),
		local_dict['hidden_channels'] 		=  list(map(int, config[config_version]['hidden_channels'])),
		local_dict['pooling_type'] 			=  config[config_version]['pooling_type'],
		local_dict['strides'] 				=  config[config_version]['strides'],
		local_dict['activation_function'] 	=  config[config_version]['activation_function'],
		local_dict['dense_depths'] 			=  list(map(int, config[config_version]['dense_depths'])),
		local_dict['batch_size'] 			=  config[config_version]['batch_size'],
		local_dict['max_iterations'] 		=  config[config_version]['max_iterations'],
		local_dict['chk_iterations'] 		=  config[config_version]['chk_iterations'],
		local_dict['dropout_k_p'] 			=  config[config_version]['dropout_k_p'],
		local_dict['fine_tuning_only'] 		=  config[config_version]['fine_tuning_only'],
		local_dict['step_size'] 			= config[config_version]['step_size'],

		for k, v in local_dict.items():
			self.configuration_dict[k] = min(v)

		print('Succesfully loaded config file, values are:')
		for k, v in self.configuration_dict.items():
			print(k, v)

	def store_config_file(self, path=sys_path+'/configs/custom.ini', config_version='custom'):

		# store content of current configuration_dict to file
		config = ConfigObj()
		config.filename = path
		config[config_version] = {}

		if self.configuration_dict and len(self.configuration_dict)==12:
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
			config.write()
		# print(config)

# config = ConfigLoader()
# print(config.configuration_dict)
# config.load_config_file('config.ini', 'default')
# print(config.configuration_dict)