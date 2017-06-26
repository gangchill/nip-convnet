import tensorflow 	as tf 
import numpy 		as np 
import os

import from_github.cifar10_input as cifar10_input

CIFAR_LOCATION = 'cifar10_data/cifar-10-batches-bin'

def train_cnn(sess, cnn, data, x, y, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_prefix = None, best_accuracy_so_far = 0):

	print("Training CNN for {} iterations with batchsize {}".format(max_iterations, batch_size))

	if data == 'cifar_10':

		coord = tf.train.Coordinator()

		image_batch, label_batch = cifar10_input.distorted_inputs(CIFAR_LOCATION, batch_size)
		test_image_node, test_label_node = cifar10_input.inputs(False, CIFAR_LOCATION, batch_size)

		# add the some test images to the summary 
		cnn._summaries.append(tf.summary.image('some example input', test_image_node))
		cnn.update_summaries()

		threads = tf.train.start_queue_runners(sess=sess, coord=coord)


	current_top_accuracy = best_accuracy_so_far

	for i in range(init_iteration, max_iterations):

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

		if data == 'cifar_10':
			batch_xs, batch_ys = sess.run([image_batch, label_batch])

		else:
			batch_xs, batch_ys = data.train.next_batch(batch_size)

		if fine_tuning_only:
			sess.run(cnn.optimize_dense_layers, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})
		else:
			sess.run(cnn.optimize, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})


		if i % chk_iterations == 0:

			if data == 'cifar_10':
				test_images, test_labels = sess.run([test_image_node, test_label_node])

			else:
				test_images, test_labels = data.test.images, data.test.labels

			avg_accuracy, summary = sess.run([cnn.accuracy, cnn.merged], feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})

			print('it {} accuracy {}'.format(i, avg_accuracy))

			if avg_accuracy > current_top_accuracy:
				print('...new top accuracy found')

				current_top_accuracy = avg_accuracy

				if save_prefix is not None:
					file_path = os.path.join(save_prefix, 'CNN-acc-{}'.format(current_top_accuracy))
					print('...save new found best weights to file ')
					cnn.store_model_to_file(sess, file_path, i)

			writer.add_summary(summary, i)


	if data == 'cifar_10':
		coord.request_stop()
		coord.join(threads)

	print('...finished training') 