import matplotlib.pyplot as plt
import numpy as np

def main():

	create_mnist_boxplot()
	create_cifar_boxplot()
	create_ckplus_boxplot()

def create_cifar_boxplot():
	## ### ##
	# CIFAR #
	## ### ##

	cifar_1k_pre_trained = []
	cifar_1k_random_init = []
	cifar_1k = [cifar_1k_random_init, cifar_1k_pre_trained]
	cifar_1k_trial_count = min(len(cifar_1k_pre_trained), len(cifar_1k_random_init))

	cifar_10k_pre_trained = [0.5577, 0.5547, 0.5485, 0.5459, 0.5481, 0.5537, 0.5583, 0.5491]
	cifar_10k_random_init = [0.5196, 0.5304, 0.5199, 0.5160, 0.5241, 0.5190, 0.5226, 0.5200, 0.5259]
	cifar_10k = [cifar_10k_random_init, cifar_10k_pre_trained]
	cifar_10k_trial_count = min(len(cifar_10k_random_init), len(cifar_10k_pre_trained))

	cifar_full_pre_trained = []
	cifar_full_random_init = []
	cifar_full = [cifar_full_random_init, cifar_full_pre_trained]
	cifar_full_trial_count = min(len(cifar_full_random_init), len(cifar_full_pre_trained))

	cifar_ylims = 0., 1.

	data 			= [cifar_1k, cifar_10k, cifar_full]
	trial_counts 	= [cifar_1k_trial_count, cifar_10k_trial_count, cifar_full_trial_count]

	visualize_boxplot(data, 'cifar', [1,10,50], trial_counts, cifar_ylims)

def create_ckplus_boxplot():
	
	print('ckplus not implemented, feed data!!!')

	pass


def create_mnist_boxplot():

	## ### ##
	# MNIST #
	## ### ##
	mnist_1k_pre_trained = [0.9038, 0.9009, 0.9009]
	mnist_1k_random_init = [0.8980, 0.8913, 0.9003]
	mnist_1k = [mnist_1k_random_init, mnist_1k_pre_trained]
	mnist_1k_trial_count = min(len(mnist_1k_pre_trained), len(mnist_1k_random_init))

	mnist_10k_pre_trained = [0.9560, 0.9514, 0.9496, 0.9553, 0.9512, 0.9529, 0.9565, 0.9532, 0.9532, 0.9474, 0.9570, 0.9518, 0.9576, 0.9499, 0.9553, 0.9554, 0.9542, 0.9552, 0.9587, 0.9470]
	mnist_10k_random_init = [0.9588, 0.9565, 0.9571, 0.9602, 0.9538, 0.9530, 0.9582, 0.9592, 0.9526, 0.9564, 0.9574, 0.9549, 0.9566, 0.9608, 0.9605, 0.9569, 0.9577, 0.9565, 0.9602, 0.9565]
	mnist_10k = [mnist_10k_random_init, mnist_10k_pre_trained]
	mnist_10k_trial_count = min(len(mnist_10k_random_init), len(mnist_10k_pre_trained))

	mnist_full_pre_trained = [0.9866, 0.9837, 0.9847, 0.9857, 0.9866, 0.9858, 0.9826, 0.9859, 0.9858]
	mnist_full_random_init = [0.9841, 0.9826, 0.9824, 0.9860, 0.9817, 0.9814, 0.9823, 0.9842, 0.9829, 0.9848]
	mnist_full = [mnist_full_random_init, mnist_full_pre_trained]
	mnist_full_trial_count = min(len(mnist_full_random_init), len(mnist_full_pre_trained))

	mnist_ylims = 0.89, 0.99

	data 			= [mnist_1k, mnist_10k, mnist_full]
	trial_counts 	= [mnist_1k_trial_count, mnist_10k_trial_count, mnist_full_trial_count]

	visualize_boxplot(data, 'mnist', [1,10,55], trial_counts, mnist_ylims)


def visualize_boxplot(data, dataset_name, in_k_sizes, trials, ylims):

	# presentation
	fig = plt.figure(figsize=(20, 10))

	x_labels    = ['random-init', 'pre-trained']
	sets = len(data)

	for i in range(sets):

		plt.subplot(1, sets, i+1)
		plt.title('{} {}k ({} trials)'.format(dataset_name, in_k_sizes[i], trials[i]))
		plt.ylim(ylims)
		plt.boxplot(data[i], 0, '')
		plt.xticks([1,2], x_labels)

	filename = 'boxplots_{}.png'.format(dataset_name)

	plt.savefig(filename, dpi=200)
	plt.close(fig) 

	print('Saved {} values to file {}'.format(dataset_name, filename))

if __name__ == '__main__':
	main()