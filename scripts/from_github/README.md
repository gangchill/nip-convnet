The files in this directory were found on github:
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

Changes made to the original files found on github: 

- To adjust the file to our folder structure, in cifar10.py, line 53 was changed from 

tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',

to 

tf.app.flags.DEFINE_string('data_dir', 'cifar10_data',

- Since our autoencoder has an output range in [0,1], we need to change the output range for the input images as well. We therefore removed the mean subtraction and whitening and rescaled the images to the [0,1] interval. The changes are made in lines 186 and 245 of cifar10_input.py.