import tensorflow as tf 

def l_relu(x, leak = 0.2, name='lrelu'):
	# elegant leaky relu formulation inspired by 
	# https://github.com/pkmital/tensorflow_tutorials/blob/master/python/libs/activations.py

	with tf.name_scope(name):
		return 0.5 * ( (1 + leak) * x + (1 - leak) * abs(x) )