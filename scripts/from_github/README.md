The files in this directory were found on github:
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

In cifar10.py, line 53 was changed from 
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
to 
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data',