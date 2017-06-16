import tensorflow 	as tf
import numpy		as np
import os

def train_ae(sess, writer,  input_placeholder, autoencoder, mnist, cae_dir, cae_weights_dir, weight_file_name, batch_size=100, max_iterations=1000, chk_iterations=500):

	print('...checking folder structure')
	folders = ['models', cae_dir, cae_weights_dir]
	cwd = os.getcwd()
	for folder in folders:
		dir_path = os.path.join(cwd, folder)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	print("Training for {} iterations with batchsize {}".format(max_iterations, batch_size))

	for i in range(max_iterations):

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

	  
		if i % chk_iterations == 0:

			summary, avg_r_e = sess.run([autoencoder.merged, autoencoder.error], feed_dict={input_placeholder: mnist.test.images})

			print('it {} avg_re {}'.format(i, np.mean(avg_r_e)))

			writer.add_summary(summary, i)

		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(autoencoder.optimize, feed_dict={input_placeholder: batch_xs})


	print('...finished training')

	autoencoder.store_model_to_file(sess, os.path.join(cae_weights_dir, weight_file_name))
	print('...saved model to file') 
