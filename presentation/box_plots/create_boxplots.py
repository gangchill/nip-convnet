import numpy 				as np
import matplotlib 			as mpl
from scipy import stats

mpl.use('agg')

import matplotlib.pyplot 	as plt

def main():

	create_mnist_boxplot()
	create_cifar_boxplot()
	create_ckplus_boxplot()

def create_cifar_boxplot():
	## ### ##
	# CIFAR #
	## ### ##

	print('\nCIFAR')

	# cifar_1k_pre_trained = [0.3655, 0.3703, 0.3681, 0.3634, 0.3650, 0.3698, 0.3800, 0.3679, 0.3652] 
	# cifar_1k_random_init = [0.3628, 0.3754, 0.3712, 0.3614, 0.3721, 0.3729, 0.3648, 0.3793, 0.3704, 0.3737]
	
	cifar_1k_pre_trained = [0.3679, 0.3618, 0.3694 ,0.3662, 0.3656, 0.3681, 0.3683, 0.3636, 0.3679, 0.3680]
	cifar_1k_random_init = [0.3635, 0.3752, 0.3705, 0.3704, 0.3561, 0.3643, 0.3678, 0.3773, 0.3695, 0.3725]


	cifar_1k = [cifar_1k_random_init, cifar_1k_pre_trained]
	cifar_1k_trial_count = min(len(cifar_1k_pre_trained), len(cifar_1k_random_init))

	_, c1_pvalue = stats.ttest_ind(cifar_1k_pre_trained, cifar_1k_random_init, equal_var = False)


	avg_improvement = (np.mean(cifar_1k_pre_trained) - np.mean(cifar_1k_random_init)) * 100
	print('--> CIFAR 1k <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('cifar_1k p-value: {}'.format(c1_pvalue))

	# cifar_10k_pre_trained = [0.5577, 0.5547, 0.5485, 0.5459, 0.5481, 0.5537, 0.5583, 0.5491]
	# cifar_10k_random_init = [0.5196, 0.5304, 0.5199, 0.5160, 0.5241, 0.5190, 0.5226, 0.5200, 0.5259]
	
	cifar_10k_pre_trained = [0.5552, 0.5566, 0.5619, 0.5584, 0.5555, 0.5611, 0.5586, 0.5614, 0.5617, 0.5559]
	cifar_10k_random_init = [0.5254, 0.5141, 0.5178, 0.5101, 0.5097, 0.5206, 0.5160, 0.5094, 0.5290, 0.5090]
	cifar_10k = [cifar_10k_random_init, cifar_10k_pre_trained]
	cifar_10k_trial_count = min(len(cifar_10k_random_init), len(cifar_10k_pre_trained))

	_, c10_pvalue = stats.ttest_ind(cifar_10k_pre_trained, cifar_10k_random_init, equal_var = False)
	

	avg_improvement = (np.mean(cifar_10k_pre_trained) - np.mean(cifar_10k_random_init)) * 100
	print('--> CIFAR 10k <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('cifar_1k p-value: {}'.format(c10_pvalue))

	# cifar_full_pre_trained = [0.6572, 0.6592, 0.6535, 0.6557, 0.6300, 0.6544]
	# cifar_full_random_init = [0.6292, 0.6287, 0.6235, 0.6293, 0.6284, 0.6252]

	cifar_full_pre_trained = [0.6572, 0.6592, 0.6535, 0.6557, 0.6300, 0.6544, 0.6612, 0.6603, 0.6544, 0.6563]
	cifar_full_random_init = [0.6292, 0.6287, 0.6235, 0.6293, 0.6284, 0.6252, 0.6181, 0.6252, 0.6248, 0.6250]
	cifar_full = [cifar_full_random_init, cifar_full_pre_trained]
	cifar_full_trial_count = min(len(cifar_full_random_init), len(cifar_full_pre_trained))

	_, cf_pvalue = stats.ttest_ind(cifar_full_pre_trained, cifar_full_random_init, equal_var = False)
	
	avg_improvement = (np.mean(cifar_full_pre_trained) - np.mean(cifar_full_random_init)) * 100
	print('--> CIFAR FULL <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('cifar_1k p-value: {}'.format(cf_pvalue))

	cifar_ylims = 0.3, 0.7

	data 			= [cifar_1k, cifar_10k, cifar_full]
	trial_counts 	= [cifar_1k_trial_count, cifar_10k_trial_count, cifar_full_trial_count]

	visualize_boxplot(data, 'cifar', [1,10,50], trial_counts, cifar_ylims)

def create_ckplus_boxplot():
	
	## #### ##
	# CKPLUS #
	## #### ##

	print('\nCKPLUS')

	# ckplus_full_pre_trained = [0.7323, 0.7374, 0.7475, 0.7374, 0.7273] 
	# ckplus_full_random_init = [0.6919, 0.7172, 0.6869, 0.7172, 0.7071]
	
	ckplus_full_pre_trained = [0.7475, 0.7374, 0.7273, 0.7525, 0.7424, 0.7323, 0.7374, 0.7424, 0.7374, 0.7374]
	ckplus_full_random_init = [0.6970, 0.6818, 0.7020, 0.6717, 0.7020, 0.7020, 0.6818, 0.7020, 0.7121, 0.6869]
	ckplus_full = [ckplus_full_random_init, ckplus_full_pre_trained]
	ckplus_full_trial_count = min(len(ckplus_full_pre_trained), len(ckplus_full_random_init))

	_, ckplus_pvalue = stats.ttest_ind(ckplus_full_pre_trained, ckplus_full_random_init, equal_var = False)

	avg_improvement = (np.mean(ckplus_full_pre_trained) - np.mean(ckplus_full_random_init)) * 100
	print('--> CKPLUS <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('cifar_1k p-value: {}'.format(ckplus_pvalue))

	ckplus_ylims = 0.6, 0.8

	data 			= [ckplus_full]
	trial_counts 	= [ckplus_full_trial_count]

	visualize_boxplot(data, 'ckplus', [0.696], trial_counts, ckplus_ylims)


def create_mnist_boxplot():

	## ### ##
	# MNIST #
	## ### ##

	print('\nMNIST')

	#mnist_1k_pre_trained = [0.9038, 0.9009, 0.9009]
	#mnist_1k_random_init = [0.8980, 0.8913, 0.9003]
	#mnist_1k_pre_trained = [0.9109, 0.9129, 0.9135, 0.9091, 0.9083, 0.9123, 0.9117, 0.9126, 0.9119, 0.9132]
	#mnist_1k_random_init = [0.9136, 0.9127, 0.9097, 0.9108, 0.9085, 0.9114, 0.9113, 0.9151, 0.9144, 0.9101]
	
	mnist_1k_pre_trained = [0.9234, 0.9231, 0.9233, 0.9221, 0.9242, 0.9249, 0.9238, 0.9232, 0.9239, 0.9234]
	mnist_1k_random_init = [0.9221, 0.9181, 0.9190, 0.9174, 0.9222, 0.9194, 0.9188, 0.9232, 0.9204, 0.9176]

	_, m1k_pvalue = stats.ttest_ind(mnist_1k_pre_trained, mnist_1k_random_init, equal_var = False)
	
	avg_improvement = (np.mean(mnist_1k_pre_trained) - np.mean(mnist_1k_random_init)) * 100
	print('--> MNIST 1k <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('mnist_1k p-value: {}'.format(m1k_pvalue))

	mnist_1k = [mnist_1k_random_init, mnist_1k_pre_trained]
	mnist_1k_trial_count = min(len(mnist_1k_pre_trained), len(mnist_1k_random_init))

	# trained too short
	#mnist_10k_pre_trained = [0.9560, 0.9514, 0.9496, 0.9553, 0.9512, 0.9529, 0.9565, 0.9532, 0.9532, 0.9474, 0.9570, 0.9518, 0.9576, 0.9499, 0.9553, 0.9554, .09542, 0.9552, 0.9587, 0.9470]
	#mnist_10k_random_init = [0.9588, 0.9565, 0.9571, 0.9602, 0.9538, 0.9530, 0.9582, 0.9592, 0.9526, 0.9564, 0.9574, 0.9549, 0.9566, 0.9608, 0.9605, 0.9569, 0.9577, 0.9565, 0.9602, 0.9565]
	# mnist_10k_pre_trained = [0.9768, 0.9789, 0.9768, 0.9751, 0.9780]
	# mnist_10k_random_init = [0.9713, 0.9750]

	mnist_10k_pre_trained = [0.9775, 0.9788, 0.9774, 0.9782, 0.9780, 0.9786, 0.9788, 0.9777, 0.9769, 0.9781]
	mnist_10k_random_init = [0.9692, 0.9740, 0.9760, 0.9753, 0.9739, 0.9714, 0.9757, 0.9748, 0.9753, 0.9754]

	_, m10k_pvalue = stats.ttest_ind(mnist_10k_pre_trained, mnist_10k_random_init, equal_var = False)
	
	avg_improvement = (np.mean(mnist_10k_pre_trained) - np.mean(mnist_10k_random_init)) * 100
	print('--> MNIST 10k <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('mnist 10k p-value: {}'.format(m10k_pvalue))

	mnist_10k = [mnist_10k_random_init, mnist_10k_pre_trained]
	mnist_10k_trial_count = min(len(mnist_10k_random_init), len(mnist_10k_pre_trained))

	# mnist_full_pre_trained = [0.9866, 0.9837, 0.9847, 0.9857, 0.9866, 0.9858, 0.9826, 0.9859, 0.9858]
	# mnist_full_random_init = [0.9841, 0.9826, 0.9824, 0.9860, 0.9817, 0.9814, 0.9823, 0.9842, 0.9829, 0.9848]
	
	mnist_full_pre_trained = [0.9856, 0.9884, 0.9881, 0.9870, 0.9887, 0.9886, 0.9872, 0.9881, 0.9870, 0.9882]
	mnist_full_random_init = [0.9834, 0.9762, 0.9850, 0.9848, 0.9830, 0.9855, 0.9875, 0.9870, 0.9863, 0.9859]

	_, mf_pvalue = stats.ttest_ind(mnist_full_pre_trained, mnist_full_random_init, equal_var = False)
	
	avg_improvement = (np.mean(mnist_full_pre_trained) - np.mean(mnist_full_random_init)) * 100
	print('--> MNIST full <--')
	print('avg improvement: {}'.format(avg_improvement))
	print('mnist full p-value: {}'.format(mf_pvalue))

	mnist_full = [mnist_full_random_init, mnist_full_pre_trained]
	mnist_full_trial_count = min(len(mnist_full_random_init), len(mnist_full_pre_trained))

	mnist_ylims = 0.9, 1.0

	data 			= [mnist_1k, mnist_10k, mnist_full]
	trial_counts 	= [mnist_1k_trial_count, mnist_10k_trial_count, mnist_full_trial_count]

	visualize_boxplot(data, 'mnist', [1,10,55], trial_counts, mnist_ylims)


def visualize_boxplot(data, dataset_name, in_k_sizes, trials, ylims, incl_trial_cnt = False, plot_mode = 'barplot'):

	fontsize 	= 27
	linewidth 	= 4
	ticksize 	= 30
	boxwidth 	= 0.5

	bxplt_linewidth = 3

	mpl.rcParams['xtick.labelsize'] = ticksize
	mpl.rcParams['ytick.labelsize'] = ticksize

	# presentation
	fig = plt.figure(figsize=(20 / 3. * len(data), 10))

	x_labels    = ['random-init', 'pre-trained']
	sets = len(data)


	widths = [boxwidth for i in range(len(data[0]))]

	for i in range(sets):

		plt.subplot(1, sets, i+1)

		if incl_trial_cnt:
			plt.title('{} {}k ({} trials)'.format(dataset_name, in_k_sizes[i], trials[i]), fontsize=fontsize)
		else:
			plt.title('{} {}k'.format(dataset_name, in_k_sizes[i]), fontsize=fontsize)

		plt.ylim(ylims)

		plt.yticks(np.linspace(ylims[0], ylims[1], 5))
		
		if plot_mode == 'boxplot':

			## ##### ##
			# BOXPLOT # 
			## ##### ##

			ax = plt.gca()
			bp = ax.boxplot(data[i], 0, '', patch_artist=True)

			## change outline color, fill color and linewidth of the boxes
			for box in bp['boxes']:
			    # change outline color
			    box.set( color='#7570b3', linewidth=bxplt_linewidth)
			    # change fill color
			    box.set( facecolor = '#1b9e77' )


			## change color and linewidth of the whiskers
			for whisker in bp['whiskers']:
			    whisker.set(color='#7570b3', linewidth=bxplt_linewidth)

			## change color and linewidth of the caps
			for cap in bp['caps']:
			    cap.set(color='#7570b3', linewidth=bxplt_linewidth)

			## change color and linewidth of the medians
			for median in bp['medians']:
			    median.set(color='#b2df8a', linewidth=bxplt_linewidth)

			'''
			## change the style of fliers and their fill
			for flier in bp['fliers']:
			    flier.set(marker='o', color='#e7298a', alpha=0.5)
			'''

			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()

			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(fontsize) 

			plt.xticks([1,2], x_labels)

		else:

			pt_data = np.array(data[i])[1]

			pt_mean 	= np.mean(pt_data)
			pt_stddev 	= np.std(pt_data) 

			ri_data = np.array(data[i])[0]

			ri_mean 	= np.mean(ri_data)
			ri_stddev 	= np.std(ri_data) 

			'''
			print 'handling {} {}k'.format(dataset_name, in_k_sizes[i])
			print 'ri_data: {}'.format(ri_data)
			print 'ri_mean: {}'.format(ri_mean)
			print 'ri_stdd: {}'.format(ri_stddev)

			print 'pt_data: {}'.format(pt_data)
			print 'pt_mean: {}'.format(pt_mean)
			print 'pt_stdd: {}'.format(pt_stddev)

			# print 'ttest_ind: {}'.format(stats.ttest_ind(ri_data, pt_data)[1])
			print ''
			'''

			ax = plt.gca()
			pt_bp = ax.bar([1.8], pt_mean, yerr=pt_stddev, error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))
			ri_bp = ax.bar([0.5], ri_mean, yerr=ri_stddev, error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))

			ax.set_xlim([0,3])

			for bar in pt_bp:

				# change outline color
				bar.set( color='#7570b3', linewidth=bxplt_linewidth)
				# change fill color
				bar.set( facecolor = '#cca677')

			for bar in ri_bp:

				# change outline color
				bar.set( color='#7570b3', linewidth=bxplt_linewidth)
				# change fill color
				bar.set( facecolor = '#1b9e77' )

			plt.xticks([0.9,2.2], x_labels, ha='center')

			# ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()

			plt.tick_params(
			    axis='x',          # changes apply to the x-axis
			    which='both',      # both major and minor ticks are affected
			    bottom='off',      # ticks along the bottom edge are off
			    top='off') # labels along the bottom edge are off

			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(fontsize) 


	filename = 'boxplots_{}.png'.format(dataset_name)

	plt.tight_layout()

	plt.savefig(filename, dpi=200)
	plt.close(fig) 

	print('Saved {} values to file {}'.format(dataset_name, filename))

if __name__ == '__main__':
	main()