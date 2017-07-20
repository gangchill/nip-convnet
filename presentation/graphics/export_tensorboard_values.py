"""
This script has to be called with the event file as argument
The resulting values can be plotted
"""
import sys
import tensorflow 			as tf
import matplotlib 			as mpl
import matplotlib.pyplot 	as plt
import numpy 				as np


def main():
	CEEs = []
	my_tag = "CAE/avg_cross_entropy"
	for e in tf.train.summary_iterator(sys.argv[1]):
	    for v in e.summary.value:
	        if v.tag == my_tag:
	            CEEs.append((e.step, v.simple_value))
	# print(CEEs)

	MSEs = []
	my_tag = "CAE/mean_squared_error"
	for e in tf.train.summary_iterator(sys.argv[1]):
	    for v in e.summary.value:
	        if v.tag == my_tag:
	            MSEs.append((e.step, v.simple_value))
	# print(MSEs)

	it, avg_ce 	= zip(*CEEs)
	it_2, mse 	= zip(*MSEs)

	assert(it == it_2)

	it 		= np.array(it)
	avg_ce 	= np.array(avg_ce)
	mse 	= np.array(mse)

	n_datapoints =  it.shape
	print ('got {} datapoints'.format(n_datapoints))


	# presentation:

	sclng_fctr = 3

	max_its 	= 10 	* sclng_fctr
	linewidth 	= 3		* sclng_fctr
	ticksize 	= 15	* sclng_fctr
	fontsize 	= 20	* sclng_fctr

	fig = plt.figure(figsize=(40,20))


	mpl.rcParams['xtick.labelsize'] = ticksize
	mpl.rcParams['ytick.labelsize'] = ticksize

	if max_its > 0:
		it 		= it[:max_its]
		avg_ce 	= avg_ce[:max_its]
		mse 	= mse[:max_its]

	plt.plot(it, avg_ce / np.max(avg_ce), 	label='(normalized) mean cross-entropy', linewidth = linewidth, color='red')
	plt.plot(it, mse    / np.max(mse), 		label='(normalized) mean squared-error', linewidth = linewidth, color='green')
	plt.legend(fontsize=fontsize)
	plt.xlabel('training iteration', fontsize=fontsize)
	plt.savefig('mse_ce_comparison.png', dpi=300)


if __name__ == '__main__':
	main()