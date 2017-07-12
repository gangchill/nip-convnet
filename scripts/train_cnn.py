import tensorflow 	as tf 
import numpy 		as np 
import os

import scripts.from_github.cifar10_input as cifar10_input

CIFAR_LOCATION = 'cifar10_data/cifar-10-batches-bin'

def train_cnn(sess, cnn, data, x, y, keep_prob, dropout_k_p, batch_size, init_iteration, max_iterations, chk_iterations, writer, fine_tuning_only, save_prefix = None, best_accuracy_so_far = 0, num_test_images = 1024, test_batch_size = 1024, evaluate_using_test_set = False, final_test_evaluation = True, best_model_for_test = True):

	print("Training CNN for {} iterations with batchsize {}".format(max_iterations, batch_size))

	sess.run(cnn.set_global_step_op, feed_dict = {cnn.global_step_setter_input: init_iteration})
	print('Set gobal step to {}'.format(init_iteration))

	if evaluate_using_test_set and final_test_evaluation:
		print('Attention: 	we are currently using the test set during training time.')
		print('				Therefore, the last test iteration is not needed and will not be executed.')
		print('Consider using the train / validation set to track progress during training and evaluate the accuracy using the test set in the end.')
		final_test_evaluation = False

	if data == 'cifar_10':

		coord = tf.train.Coordinator()

		image_batch, label_batch = cifar10_input.distorted_inputs(CIFAR_LOCATION, batch_size)
		iteration_evaluation_image_node, iteration_evaluation_label_node = cifar10_input.inputs(evaluate_using_test_set, CIFAR_LOCATION, test_batch_size)

		if final_test_evaluation:
			test_image_node, test_label_node = cifar10_input.inputs(True, CIFAR_LOCATION, test_batch_size)

		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		# determine test set size (cifar10 test set is 10000)
		if num_test_images <= 0:
			total_test_images = 10000
		else:
			total_test_images = min(num_test_images, 10000)

		if evaluate_using_test_set:
			iteration_evaluation_name = 'test'
		else:
			iteration_evaluation_name = 'train'

	else:

		# choose dataset used for testing every chk_iterations-th iteration
		if evaluate_using_test_set:
			iteration_evaluation_set = data.test 
			iteration_evaluation_name = 'test'
		else:
			iteration_evaluation_set = data.validation 
			iteration_evaluation_name = 'validation'

			if final_test_evaluation:
				test_set = data.test

		max_test_images = iteration_evaluation_set.images.shape[0]
		if num_test_images <= 0:
			total_test_images = max_test_images
		else:
			total_test_images = min(num_test_images, max_test_images)


	current_top_accuracy = best_accuracy_so_far

	# create two different savers (always store the model from the last 5 check iterations and the current model with the best accuracy)
	chk_it_saver 	= tf.train.Saver(cnn.all_variables_dict, max_to_keep = 1)
	best_it_saver 	= tf.train.Saver(cnn.all_variables_dict, max_to_keep = 1)

	#
	total_test_set_accuracy = tf.Variable(0, '{}_set_accuracy'.format(iteration_evaluation_name))

	for i in range(init_iteration, max_iterations):

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

		if data == 'cifar_10':
			batch_xs, batch_ys = sess.run([image_batch, label_batch])

		else:
			batch_xs, batch_ys = data.train.next_batch(batch_size)

		if i % chk_iterations == 0:

			# batch the test data (prevent memory overflow)
			last_batch_size = total_test_images % test_batch_size
			num_batches 	= total_test_images / test_batch_size + int(last_batch_size > 0)
			
			if last_batch_size == 0:
				last_batch_size = test_batch_size

			print('---> Test Iteration')

			if fine_tuning_only:
				print('BE AWARE: we are currently only optimizing the dense layer weights, convolution weights and biases stay unchanged')
			print('Current performance is evaluated using the {}-set'.format(iteration_evaluation_name))
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
					test_images, test_labels = sess.run([iteration_evaluation_image_node, iteration_evaluation_label_node])

				else:
					test_images, test_labels = iteration_evaluation_set.next_batch(current_batch_size)

				avg_accuracy, summary = sess.run([cnn.accuracy, cnn.merged], feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})

				total_accuracy += avg_accuracy * current_batch_size

			total_accuracy = total_accuracy / total_test_images

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

			with tf.name_scope('CNN'):
				total_batch_acc_summary = tf.Summary()
				total_batch_acc_summary.value.add(tag='acc_over_all_ {}_batches'.format(iteration_evaluation_name), simple_value=total_accuracy)
				writer.add_summary(total_batch_acc_summary, i)

			writer.add_summary(summary, i)

		# perform one training step
		if fine_tuning_only:
			sess.run([cnn.optimize_dense_layers, cnn.increment_global_step_op], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})
		else:
			sess.run([cnn.optimize, cnn.increment_global_step_op], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})



	print('...finished training') 

	if final_test_evaluation:
		print('The network was trained without presence of the test set.')
		print('...Performing test set evaluation')

		print('loading best model')
		best_model_folder = os.path.join(save_prefix, 'best')
		print('looking for best weights in {}'.format(best_model_folder))

		latest_checkpoint = tf.train.latest_checkpoint(best_model_folder)

		best_it_saver.restore(sess, latest_checkpoint)


		if data == 'cifar_10':
			total_test_images = 10000
			print('Treating cifar10: the total test set size is {} images'.format(num_test_images))
		else:
			total_test_images = test_set.images.shape[0]
			print('Test set size is {}'.format(total_test_images))

		# batch the test data (prevent memory overflow)
		last_batch_size = total_test_images % test_batch_size
		num_batches 	= total_test_images / test_batch_size + int(last_batch_size > 0)
		
		if last_batch_size == 0:
			last_batch_size = test_batch_size

		print('-------------------------------------')
		print('---> FINAL TEST SET EVALUATION <-----')
		print('-------------------------------------')
		print('Test batch size is {}'.format(test_batch_size))
		print('We want to average over {} test images in total'.format(num_test_images))
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
				test_images, test_labels = test_set.next_batch(current_batch_size)

			avg_accuracy, summary = sess.run([cnn.accuracy, cnn.merged], feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})

			total_accuracy += avg_accuracy * current_batch_size

		total_accuracy = total_accuracy / total_test_images

		with tf.name_scope('CNN'):
				total_batch_acc_summary = tf.Summary()
				total_batch_acc_summary.value.add(tag='final_test_set_accuracy', simple_value=total_accuracy)
				writer.add_summary(total_batch_acc_summary, max_iterations)


	if data == 'cifar_10':
		coord.request_stop()
		coord.join(threads)

	