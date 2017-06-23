import tensorflow 	as tf 
import numpy 		as np 
import os

import cifar_10_input

CIFAR_LOCATION = 'cifar10_data/cifar-10-batches-bin'

def train_cnn(sess, cnn, data, x, y, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations, writer, fine_tuning_only):

	print("Training SCNN for {} iterations with batchsize {}".format(max_iterations, batch_size))

	for i in range(max_iterations):

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

		if data == 'cifar_10':
			batch_xs, batch_ys = cifar_10_input.distorted_inputs(CIFAR_LOCATION, batch_size)

			with sess.as_default():
				batch_xs = batch_xs.eval()
				batch_ys = batch_ys.eval()

		else:
			batch_xs, batch_ys = data.train.next_batch(batch_size)

		if fine_tuning_only:
			sess.run(cnn.optimize_dense_layers, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})
		else:
			sess.run(cnn.optimize, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})


		if i % chk_iterations == 0:

			if data == 'cifar_10':
				test_images, test_labels = cifar_10_input.inputs(False, CIFAR_LOCATION, batch_size)

				with sess.as_default():
					test_images = test_images.eval()
					test_labels = test_labels.eval()
			else:
				test_images, test_labels = data.test.images, data.test.labels

			avg_r_e, summary = sess.run([cnn.accuracy, cnn.merged], feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})

			print('it {} accuracy {}'.format(i, np.mean(avg_r_e)))

			writer.add_summary(summary, i)



	print('...finished training') 
