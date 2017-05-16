# -----------------------------
# tensor flow: autoencoder test
# -----------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.sparse

# import mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('...checking folder structure')
folders = ['models', 'digit_reconstructions']
cwd = os.getcwd()
for folder in folders:
	dir_path = os.path.join(cwd, folder)
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

# training parameters:
error_function 	= 'cross-entropy' # options: 'cross-entropy', 'MSE'
step_size 		= 0.01
batch_size 		= 100
max_iterations 	= 100
chk_iterations  = 100

single_image_output = False

# autoencoder parameters
current_layer = 1 # first hidden layer, used for deeper constructions later
hidden_layer_size = 28*28

# input variables: x (images), y_ (labels)
x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')


with tf.name_scope('autoencoder_network'):

	# model variables W, b:
	lim_value = 1. / (hidden_layer_size ** (current_layer - 1) )

	W = tf.Variable(tf.random_uniform([784, hidden_layer_size], minval=-lim_value, maxval=lim_value), name='encoding_weights')
	# W = tf.Variable(tf.zeros([784, hidden_layer_size], name='encoding_weights'))
	b = tf.Variable(tf.zeros([hidden_layer_size]), name='encoding_bias')
	c = tf.Variable(tf.zeros([784]), name='reconstruction_bias')

	# reconstruction W not needed, we tie the weights
	# W_=tf.Variable(tf.random_uniform([hidden_layer_size, 784], minval=-lim_value, maxval=lim_value))

	# hidden layer representation:
	hl_1 = tf.nn.sigmoid(tf.matmul(x, W) + b, name='encoding')

	# reconstruction:
	if error_function == 'cross-entropy':
		reconstruction = tf.add(tf.matmul(hl_1, tf.transpose(W)), c, name='reconstruction_without_sigmoid')
	else:
		reconstruction = tf.nn.sigmoid(tf.matmul(hl_1, tf.transpose(W)) + c, naprintme='reconstruction')

# error function:
if error_function == 'cross-entropy':
	reconstruction_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=reconstruction, name='cross-entropy_error')

else: # default is MSE (mean squared error)
	reconstruction_error = tf.reduce_mean(tf.squared_difference(x, reconstruction), name = 'MSE')


# training step:
train_step = tf.train.GradientDescentOptimizer(step_size).minimize(reconstruction_error)

print("begin autoencoder training test")

# training: 
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

saver = tf.train.Saver()

for i in range(max_iterations):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={x: batch_xs})

  if chk_iterations > 100 and i % 100 == 0:
  	print '...iteration {}'.format(i)

  if i % chk_iterations == 0:

	avg_r_e = sess.run(reconstruction_error, feed_dict={x: mnist.test.images})

	print('it {} avg_re {}'.format(i, np.mean(avg_r_e)))


print("...training done")
print ("...save weights to file")

save_path = saver.save(sess, "models/weights.ckpt")


# print 'mnist interval: {} {}'.format(np.max(mnist.test.images), np.min(mnist.test.images))

if error_function == 'cross-entropy':
	test_reconstruction = tf.nn.sigmoid(reconstruction, name='reconstruction_images')
else:
	test_reconstruction = reconstruction

encoding, reconst = sess.run([hl_1, test_reconstruction], feed_dict={x: mnist.test.images[0:10]})

code_dimx = int(hidden_layer_size**.5)

print 'save 10 example images to file'

# save 10 example images reconstructions: 
for i in range(3):

	fontsize = 30

	fig = plt.figure(figsize=(20,20))

	plt.subplot(1,3,1)
	plt.title('input image', fontsize=fontsize)
	plt.imshow(mnist.test.images[i].reshape(28, 28), cmap='gray', interpolation='None')
	plt.axis('off')


	plt.subplot(1,3,2)
	plt.title('encoded representation', fontsize=fontsize)
	plt.imshow(encoding[i].reshape(code_dimx, code_dimx), cmap='gray', interpolation='None')
	plt.axis('off')

	plt.subplot(1,3,3)
	plt.title('reconstruction', fontsize=fontsize)
	plt.imshow(reconst[i].reshape(28, 28), cmap='gray', interpolation='None')
	plt.axis('off')

	plt.tight_layout()

	plt.savefig(os.path.join('digit_reconstructions', '{}_example{}.png'.format(error_function, i)), dpi=400)

	if single_image_output:

		fig = plt.figure(figsize=(20,20))
		plt.imshow(mnist.test.images[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')
		plt.savefig(os.path.join('digit_reconstructions', 'single_image_output', '{}_0_input.png'.format(i)))
		plt.close(fig)

		fig = plt.figure(figsize=(20,20))
		plt.imshow(encoding[i].reshape(code_dimx, code_dimx), cmap='gray', interpolation='None')
		plt.axis('off')
		plt.savefig(os.path.join('digit_reconstructions', 'single_image_output', '{}_1_encoding.png'.format(i)))
		plt.close(fig)


		fig = plt.figure(figsize=(20,20))
		plt.imshow(reconst[i].reshape(28, 28), cmap='gray', interpolation='None')
		plt.axis('off')
		plt.savefig(os.path.join('digit_reconstructions', 'single_image_output', '{}_2_reconstruction.png'.format(i)))
		plt.close(fig)


# add logwriter for tensorboard
writer = tf.summary.FileWriter("logs", sess.graph)
writer.close()

sess.close()


