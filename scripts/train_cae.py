import tensorflow 	as tf
import numpy		as np
import sys, os

import scripts.from_github.cifar10_input as cifar10_input

CIFAR_LOCATION = 'cifar10_data/cifar-10-batches-bin'

def train_ae(sess, writer,  input_placeholder, autoencoder, data, cae_dir, weight_file_name, error_function = 'cross_entropy', batch_size=100, init_iteration = 0, max_iterations=1000, chk_iterations=500, save_prefix = None, minimal_reconstruction_error = sys.maxsize):

	if data == 'cifar_10':

		coord = tf.train.Coordinator()

		image_batch, label_batch = cifar10_input.distorted_inputs(CIFAR_LOCATION, batch_size)
		test_image_node, test_label_node = cifar10_input.inputs(False, CIFAR_LOCATION, batch_size)

		# add the some test images to the summary 
		autoencoder.add_summary(tf.summary.image('some example input', test_image_node))
		autoencoder.update_summaries()

		threads = tf.train.start_queue_runners(sess=sess, coord=coord)



	print("Training for {} iterations with batchsize {}".format(max_iterations, batch_size))
	print("Error function is {}".format(error_function))

	if error_function == 'mse':
		optimizer_node = autoencoder.optimize_mse
	else:
		optimizer_node = autoencoder.optimize


	# create two different savers (always store the model from the last 5 check iterations and the current model with the best accuracy)
	chk_it_saver 	= tf.train.Saver(autoencoder.all_variables_dict, max_to_keep = 1)
	best_it_saver 	= tf.train.Saver(autoencoder.all_variables_dict, max_to_keep = 1)


	for i in range(init_iteration, max_iterations):

		if data == 'cifar_10':
			batch_xs, batch_ys = sess.run([image_batch, label_batch])

		else:
			batch_xs, batch_ys = data.train.next_batch(batch_size)

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

	  
		if i % chk_iterations == 0:


			if data == 'cifar_10':
				test_images, test_labels = sess.run([test_image_node, test_label_node])

			else:
				test_images, test_labels = data.validation.images[:128], data.validation.labels[:128]


			summary, reconstruction_error = sess.run([autoencoder.merged, autoencoder.error], feed_dict={input_placeholder: test_images})

			average_reconstruction_error = np.mean(reconstruction_error)

			print('it {} avg_re {}'.format(i, average_reconstruction_error))

			if save_prefix is not None:
				file_path = os.path.join(save_prefix, 'cae_model-mre-{}'.format(minimal_reconstruction_error))
				print('...save iteration weights to file ')
				autoencoder.store_model_to_file(sess, file_path, i, saver=chk_it_saver)

			if average_reconstruction_error < minimal_reconstruction_error:
				print('...found new weight configuration with minimal reconstruction error')

				minimal_reconstruction_error = average_reconstruction_error

				if save_prefix is not None:
					file_path = os.path.join(save_prefix, 'best', 'cae_model-mre-{}'.format(minimal_reconstruction_error))
					print('...save new found best weights to file ')
					autoencoder.store_model_to_file(sess, file_path, i, saver=best_it_saver)


			writer.add_summary(summary, i)
			
		sess.run(optimizer_node, feed_dict={input_placeholder: batch_xs})


	if data == 'cifar_10':
		coord.request_stop()
		coord.join(threads)



	print('...finished training')