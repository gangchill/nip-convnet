import numpy as np
import os.path
from sklearn.model_selection import train_test_split
import tarfile
import urllib

from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, dense_to_one_hot


NUM_CLASSES = 10

cifar_dir = "datasets/cifar-10-batches-py/"
batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]


def read_data_sets(validation_size = 5000, one_hot=True):
    cifar_filename = "datasets/" + "cifar-10-python.tar.gz"

    try:
        os.makedirs("datasets")
    except OSError:
        pass

    if not os.path.isfile(cifar_dir + batches[0]):
        # Download data
        print("Downloading ckplus dataset")
        urllib.urlretrieve("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", cifar_filename)
        tar = tarfile.open(cifar_filename)
        tar.extractall(path="datasets")
        tar.close()
        os.remove(cifar_filename)

    # Process batches
    all_batch_images = []
    all_batch_labels = []
    for batch_name in batches:
        batch = np.load(cifar_dir + batch_name)
        batch_images = batch['data']
        all_batch_images.append(batch_images)
        batch_labels = batch['labels']
        all_batch_labels.extend(batch_labels)

    all_batch_images = np.vstack(all_batch_images).reshape(-1, 3, 32, 32)
    all_batch_images = all_batch_images.transpose([0, 2, 3, 1])
    all_batch_labels = np.array(all_batch_labels)

    train_images, validation_images, train_labels, validation_labels = train_test_split(all_batch_images, all_batch_labels, test_size = validation_size, random_state=0)


    test_batch = np.load(cifar_dir + "test_batch")
    test_images = test_batch['data'].reshape(-1, 3, 32, 32)
    test_images = test_images.transpose([0, 2, 3, 1])

    test_labels = np.array(test_batch['labels'])


    if one_hot:
        train_labels = dense_to_one_hot(train_labels, NUM_CLASSES)
        validation_labels = dense_to_one_hot(validation_labels, NUM_CLASSES)
        test_labels = dense_to_one_hot(test_labels, NUM_CLASSES)

    train = DataSet(train_images, train_labels, reshape=False)
    validation = DataSet(validation_images, validation_labels, reshape=False)
    test = DataSet(test_images, test_labels, reshape=False)

    return Datasets(train=train, validation=validation, test=test)
