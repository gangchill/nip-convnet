from configobj import ConfigObj
from collections import OrderedDict

class ConfigLoader:

	def __init__(self):
		self.configuration_dict = OrderedDict()

	def load_config_file(self, path='config.ini', config_version='default'):
		config = ConfigObj(path)
		local_dict = OrderedDict()
		# load config information from file and store in configuarion_dict
		local_dict['filter_dims'] = list(zip(list(map(int, config[config_version]['filter_dims_x'])), list(map(int,config[config_version]['filter_dims_y'])))),
		local_dict['hidden_channels'] =  list(map(int, config[config_version]['hidden_channels'])),
		local_dict['pooling_type'] =  config[config_version]['pooling_type'],
		local_dict['strides'] =  config[config_version]['strides'],
		local_dict['activation_function'] =  config[config_version]['activation_function'],
		local_dict['dense_depths'] =  list(map(int, config[config_version]['dense_depths'])),
		local_dict['batch_size'] =  config[config_version]['batch_size'],
		local_dict['max_iterations'] =  config[config_version]['max_iterations'],
		local_dict['chk_iterations'] =  config[config_version]['chk_iterations'],
		local_dict['dropout_k_p'] =  config[config_version]['dropout_k_p'],
		local_dict['fine_tuning_only'] =  config[config_version]['fine_tuning_only']

		for k, v in local_dict.items():
			self.configuration_dict[k] = min(v)

		for k, v in self.configuration_dict.items():
			print(k, v)

	def store_config_file(self, path='config.ini', config_version='default'):

		# store content of current configuration_dict to file
		pass

c = ConfigLoader()
c.load_config_file()