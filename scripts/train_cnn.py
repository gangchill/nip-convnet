import tensorflow 	as tf 
import numpy 		as np 
import os

import scripts.from_github.cifar10_input as cifar10_input

CIFAR_LOCATION = 'cifar10_data/cifar-10-batches-bin'

def train_cnn(sess, cnn, data, x, y, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_prefix = None, best_accuracy_so_far = 0, num_test_images = -1, test_batch_size = 1024):

	print("Training CNN for {} iterations with batchsize {}".format(max_iterations, batch_size))

	cnn.global_step = init_iteration

	if data == 'cifar_10':

		coord = tf.train.Coordinator()

		image_batch, label_batch = cifar10_input.distorted_inputs(CIFAR_LOCATION, batch_size)
		test_image_node, test_label_node = cifar10_input.inputs(True, CIFAR_LOCATION, test_batch_size)

		# add the some test images to the summary 
		cnn._summaries.append(tf.summary.image('some example input', test_image_node))
		cnn.update_summaries()

		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		# determine test set size (cifar10 test set is 10000)
		if num_test_images <= 0:
			total_test_images = 10000
		else:
			total_test_images = min(num_test_images, 10000)

	else:
		max_test_images = data.test.images.shape[0]
		if num_test_images <= 0:
			total_test_images = max_test_images
		else:
			total_test_images = min(num_test_images, max_test_images)


	current_top_accuracy = best_accuracy_so_far

	# create two different savers (always store the model from the last 5 check iterations and the current model with the best accuracy)
	chk_it_saver 	= tf.train.Saver(cnn.all_variables_dict, max_to_keep = 1)
	best_it_saver 	= tf.train.Saver(cnn.all_variables_dict, max_to_keep = 1)

	# 

	for i in range(init_iteration, max_iterations):

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

		if data == 'cifar_10':
			batch_xs, batch_ys = sess.run([image_batch, label_batch])

		else:
			batch_xs, batch_ys = data.train.next_batch(batch_size)

		if fine_tuning_only:
			print('ATTENTION: Fine-TUNING-ONLY')
			sess.run(cnn.optimize_dense_layers, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})
		else:
			sess.run(cnn.optimize, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})


		if i % chk_iterations == 0:

			print total_test_images, test_batch_size

			# batch the test data (prevent memory overflow)
			last_batch_size = total_test_images % test_batch_size
			num_batches 	= total_test_images / test_batch_size + int(last_batch_size > 0)
			
			if last_batch_size == 0:
				last_batch_size = test_batch_size

			print('---> Test Iteration')
			print('Test batch size is {}'.format(test_batch_size))
			print('We want to average over {} test images in total'.format(total_test_images))
			print('This gives us {} batches, the last one having only {} images'.format(num_batches, last_batch_size))

			total_accuracy = 0

			for batch_indx in range(num_batches): 

				print('...treating batch {}'.format(batch_indx))

				if batch_indx == num_batches - 1:
					current_batch_size = last_batch_size
				else:
					current_batch_size = test_batch_size

				if data == 'cifar_10':
					test_images, test_labels = sess.run([test_image_node, test_label_node])

				else:
					test_images, test_labels = data.test.next_batch(current_batch_size)

				avg_accuracy, summary = sess.run([cnn.accuracy, cnn.merged], feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})

				total_accuracy += avg_accuracy * current_batch_size

			total_accuracy = total_accuracy / total_test_images
			'''
			TODO: accumulate summary
			# batch_acc_summary = tf.summary.scalar('acc_over_all_test_batches', total_accuracy)

			# total_summary = tf.summary.merge([summary, batch_acc_summary], name='total_summary')

			b_summary = tf.Summary()
			b_summary.value.add(tag="batch_averaged_acc", simple_value=total_accuracy)

			print type(b_summary)

			total_summary = tf.summary.merge([summary, b_summary])

			'''

			print('it {} accuracy {}'.format(i, total_accuracy))
			

			# always keep the models from the last 5 iterations stored
			if save_prefix is not None:
					file_path = os.path.join(save_prefix, 'CNN-acc-{}'.format(total_accuracy))
					print('...save current iteration weights to file ')
					cnn.store_model_to_file(sess, file_path, i, saver=chk_it_saver)

			if total_accuracy > current_top_accuracy:
				print('...new top accuracy found')

				current_top_accuracy = total_accuracy

				if save_prefix is not None:
					file_path = os.path.join(save_prefix, 'best', 'CNN-acc-{}'.format(current_top_accuracy))
					print('...save new found best weights to file ')
					cnn.store_model_to_file(sess, file_path, i, saver=best_it_saver)

			writer.add_summary(summary, i)


	if data == 'cifar_10':
		coord.request_stop()
		coord.join(threads)

	print('...finished training') 