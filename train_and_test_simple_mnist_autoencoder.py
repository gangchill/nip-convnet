# -------------------------------------------------------------------------------
# train and test a simple autoencoder with one hidden layer for the MNIST dataset
# -------------------------------------------------------------------------------

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

# import the simple autoencoder class from SAE.py
from models.sae.simple_autoencoder import SAE

########
# MAIN #
########

def main():

	# directory containing the autoencoder file
	sae_dir 		= os.path.join('models', 'sae')
	sae_weights_dir	= os.path.join(sae_dir, 'weights')

	# restore weights from file if an autoencoder with the same architecture has already been trained before
	restore_weights_if_existant = True

	# import mnist data set
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	hidden_layer_size = 6*6

	# input variables: x (images)
	x  = tf.placeholder(tf.float32, [None, 784], name='input_digits')

	# construct autoencoder
	autoencoder = SAE(x, hidden_layer_size)

	print 'call the properties to initialize the graph'
	autoencoder.optimize
	autoencoder.reconstruction

	sess = tf.Session() 
	sess.run(tf.global_variables_initializer())

	print("Begin autencoder training")
	batch_size 		= 100
	max_iterations 	= 100
	chk_iterations  = 100

	if restore_weights_if_existant:
		# only train a new autoencoder if no weights file is found

		cwd = os.getcwd()
		chkpnt_file_path = os.path.join(cwd, sae_weights_dir, '{}_autoencoder_{}it.ckpt'.format(autoencoder.hidden_layer_size, max_iterations))

		if os.path.exists(chkpnt_file_path + '.index'):
			print 'Model file for same configuration was found ... load weights'

			autoencoder.load_model_from_file(sess, chkpnt_file_path)			

		else:
			train_ae(sess, x, autoencoder, mnist, sae_dir, sae_weights_dir, batch_size, max_iterations, chk_iterations)

	else:
		# always train a new autoencoder 
		train_ae(sess, x, autoencoder, mnist, sae_dir, sae_weights_dir, batch_size, max_iterations, chk_iterations)
	

	print 'Test the training:'

	visualize_ae_representation(sess, x, autoencoder, mnist, 1)


	# add logwriter for tensorboard
	writer = tf.summary.FileWriter("logs", sess.graph)
	writer.close()

	sess.close()


def train_ae(sess, input_placeholder, autoencoder, mnist, sae_dir, sae_weights_dir, batch_size=100, max_iterations=1000, chk_iterations=500):

	print('...checking folder structure')
	folders = ['models', sae_dir, sae_weights_dir]
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	print("Training for {} iterations with batchsize {}".format(max_iterations, batch_size))

	for i in range(max_iterations):

	  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	  sess.run(autoencoder.optimize, feed_dict={input_placeholder: batch_xs})

	  if chk_iterations > 100 and i % 100 == 0:
	  	print '...iteration {}'.format(i)

	  if i % chk_iterations == 0:

		avg_r_e = sess.run(autoencoder.error, feed_dict={input_placeholder: mnist.test.images})

		print('it {} avg_re {}'.format(i, np.mean(avg_r_e)))


	print '...finished training'

	autoencoder.store_model_to_file(sess, os.path.join(sae_weights_dir, '{}_sae_{}it'.format(autoencoder.hidden_layer_size, max_iterations)))
	print '...saved model to file'

def visualize_ae_representation(sess, input_placeholder, autoencoder, mnist, num_images):

	# initialize folder structure if not yet done
	print('...checking folder structure')
	folders = ['digit_reconstructions']
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	encoding, reconst = sess.run([autoencoder.encoding, autoencoder.reconstruction], feed_dict={input_placeholder: mnist.test.images[0:num_images]})

	code_dimx = int(autoencoder.hidden_layer_size**.5)

	print 'save {} example images to file'.format(num_images)

	for i in range(num_images):

		fontsize = 30

		fig = plt.figure(figsize=(20,10))

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

		plt.savefig(os.path.join('digit_reconstructions', 'class_ae_example{}.png'.format(i)), dpi=400)


if __name__ == '__main__':
	main()