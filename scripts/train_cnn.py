import tensorflow 	as tf 
import numpy 		as np 
import os

def train_cnn(sess, cnn, mnist, x, y, keep_prob, dropout_k_p, batch_size, max_iterations, chk_iterations, writer, fine_tuning_only):

	print("Training SCNN for {} iterations with batchsize {}".format(max_iterations, batch_size))

	for i in range(max_iterations):

		if chk_iterations > 100 and i % 100 == 0:
			print('...iteration {}'.format(i))

		batch_xs, batch_ys = mnist.train.next_batch(batch_size)

		if fine_tuning_only:
			sess.run(cnn.optimize_dense_layers, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})
		else:
			sess.run(cnn.optimize, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_k_p})


		if i % chk_iterations == 0:

			avg_r_e, summary = sess.run([cnn.accuracy, cnn.merged], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})

			print('it {} avg_re {}'.format(i, np.mean(avg_r_e)))

			writer.add_summary(summary, i)



	print('...finished training') 
