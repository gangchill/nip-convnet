# ----------------------------------------------------
# train and test a simple convolutional neural network
# ----------------------------------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# import the  SCNN class from simple_.py
from models.cnn.simple_cnn import SCNN

########
# MAIN #
########

def main():

	# load mnist
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# input variables: x (images), y_ (labels), keep_prob (dropout rate)
	x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')
	y_ = tf.placeholder(tf.float32, [None, 10], name='target_labels')

	keep_prob = tf.placeholder(tf.float32)

	# reshape the input to NHWD format
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# CNN parameters:

	# feature extraction parameters
	filter_dims 	= [(5,5), (5,5)]
	hidden_channels = [32,64] 
	use_max_pooling = True
	strides = None # other strides should not work yet
	activation_function = 'relu'

	# fc-layer parameters:
	dense_depths = [1024]

	cnn = SCNN(x_image, y_, keep_prob, filter_dims, hidden_channels, dense_depths, strides, use_max_pooling, activation_function)

	# training parameters:
	batch_size 		= 50
	max_iterations	= 20000
	chk_iterations 	= 1000
	dropout_k_p		= 0.5

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	train_cnn(sess, cnn, mnist, x, y_, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations)

	# add logwriter for tensorboard
	writer = tf.summary.FileWriter("logs", sess.graph)
	writer.close()

	sess.close()


def train_cnn(sess, cnn, mnist, x, y, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations):

	print("Training SCNN for {} iterations with batchsize {}".format(max_iterations, batch_size))

	for i in range(max_iterations):

		batch_xs, batch_ys = mnist.train.next_batch(batch_size)

		sess.run(cnn.optimize, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))


		if i % chk_iterations == 0:

			avg_r_e = sess.run(cnn.accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})

			print('it {} avg_re {}'.format(i, np.mean(avg_r_e)))


	print('...finished training')








if __name__ == '__main__':
	main()