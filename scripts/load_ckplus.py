import os

import numpy as np
import pandas as pd
from PIL import Image
import random

import scipy.misc
from sklearn.model_selection import train_test_split

from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, dense_to_one_hot
import traceback

PATCH_SIZE = (325, 340)
INPUT_SIZE = (65,68)
NUM_CLASSES = 7


dataset_path = "datasets/cohn-kanade-images"
emotions_path = "datasets/Emotion"

landmarks = {}

def normalize_filename(fn):
    return '_'.join(fn.split('.')[0].split('_')[:3])

def identify_patch(landmarks):
    left = int(np.min(landmarks[:,0]))
    right = int(np.max(landmarks[:,0]))
    bottom = int(np.min(landmarks[:,1]))
    top = int(np.max(landmarks[:,1]))

    x_missing = PATCH_SIZE[0] - (right - left)
    y_missing = PATCH_SIZE[1] - (bottom - top)

    left -= np.ceil(x_missing/2.)
    right += np.floor(x_missing/2.)
    bottom += np.ceil(y_missing/2.)
    top -= np.floor(y_missing/2.)

    return (left, top, right, bottom)

def load_landmarks():
    print("Loading landmarks")
    dataset_path = "datasets/Landmarks"
    x_dists = []
    y_dists = []
    all_landmarks = {}
    for subject in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, subject)):
            for sequence in os.listdir(os.path.join(dataset_path, subject)):
                if os.path.isdir(os.path.join(dataset_path, subject, sequence)):
                    for landmarks in os.listdir(os.path.join(dataset_path, subject, sequence)):
                        l_data = np.genfromtxt(os.path.join(dataset_path, subject, sequence, landmarks))
                        all_landmarks[normalize_filename(os.path.join(subject, sequence, landmarks))] = l_data
                        try:
                            x_dists.append(np.max(l_data[:,0]) - np.min(l_data[:,0]))
                            y_dists.append(np.max(l_data[:,1]) - np.min(l_data[:,1]))
                        except:
                            print(traceback.format_exc())
    return all_landmarks

def create_image_thumbnail(img_path):
    global landmarks
    image_path = normalize_filename(img_path)
    thumb_size = INPUT_SIZE
    thumb_path = '.'.join(img_path.split('.')[:-1]) + '_patch_' + str(thumb_size[0]) + str(thumb_size[1]) + '.png'
    thumb_path = os.path.join(dataset_path, thumb_path)


    if not os.path.isfile(thumb_path):
        if not landmarks:
            # Load landmarks
            landmarks = load_landmarks()

        image = Image.open(os.path.join(dataset_path, image_path) + '.png')
        image = image.crop(identify_patch(landmarks[image_path]))
        image = image.resize(thumb_size, Image.ANTIALIAS)

        if not image.size == INPUT_SIZE:
            print("Image ratio not matching")
            raise Exception("Image ratio not matching")
        image.save(thumb_path)

    return thumb_path


def read_from_folders(folders, frames):
    # Load Emotions / Labels
    if frames:
        emotions = []
        for subject in folders:
            if os.path.isdir(os.path.join(emotions_path, subject)):
                for sequence in os.listdir(os.path.join(emotions_path, subject)):
                    if os.path.isdir(os.path.join(emotions_path, subject, sequence)):
                        if len(os.listdir(os.path.join(emotions_path, subject, sequence))) > 0:
                            emo_file = os.listdir(os.path.join(emotions_path, subject, sequence))[0]
                            filename = '_'.join(emo_file.split('.')[0].split('_')[:-1])

                            with open(os.path.join(emotions_path, subject, sequence, emo_file), 'r') as f:
                                emotion=int(float(f.read()))
                            emotions.append({'filename': filename, 'emotion': emotion})

                            if frames > 1:
                                base_filename = '_'.join(filename.split('_')[:-1])
                                frame_number = int(filename.split('_')[-1])

                                for i in range(frames-1):
                                    filename = base_filename + '_' + str(frame_number-i-1).rjust(8, '0')
                                    emotions.append({'filename': filename, 'emotion': emotion})

    # Load images
    data = []
    for subject in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, subject)):
            for sequence in os.listdir(os.path.join(dataset_path, subject)):
                if os.path.isdir(os.path.join(dataset_path, subject, sequence)):
                    for pngfile in os.listdir(os.path.join(dataset_path, subject, sequence)):
                        if frames is None and not ("thumb" in pngfile or "patch" in pngfile):
                            emotion = 0
                        elif frames:
                            try:
                                # Try to assign corresponding emotion, raises IndexError if not found
                                emotion = [e['emotion'] for e in emotions if e['filename'] == pngfile.split('.')[:-1][0]][0]

                                # Create thumbnail of image
                                img_path = os.path.join(subject, sequence, pngfile)
                                thumb_path = create_image_thumbnail(img_path)

                                image = scipy.misc.imread(thumb_path, mode='F').flatten()
                                if len(image) == 4420:

                                    # Normalize pixel values
                                    #image = image * (1. / 255)

                                    image_dict = dict(enumerate(image))
                                    image_dict['emotion'] = emotion - 1
                                    data.append(image_dict)
                                else:
                                    print("Image " + pngfile + " has wrong dimensions")
                            except IndexError:
                                pass
    return data


def read_data_sets(split=True, num_train_folders=90, num_test_folders=24, one_hot=True, frames=3):

    all_folders = os.listdir(emotions_path)
    random.Random(0).shuffle(all_folders)

    if split:
        train_folders = all_folders[:num_train_folders]
        test_folders = all_folders[num_train_folders:num_train_folders+num_test_folders]
        validation_folders = all_folders[num_train_folders+num_test_folders:]

        train_df = pd.DataFrame(read_from_folders(train_folders, frames))
        validation_df = pd.DataFrame(read_from_folders(validation_folders, frames))
        test_df = pd.DataFrame(read_from_folders(test_folders, frames))
        print("{} CK+ TRAIN datapoints loaded".format(len(train_df)))
        print("{} CK+ VALIDATION datapoints loaded".format(len(validation_df)))
        print("{} CK+ TEST datapoints loaded".format(len(test_df)))
    else:
        train_df = pd.DataFrame(read_from_folders(all_folders, frames))
        validation_df = train_df.copy()
        test_df = train_df.copy()
        print("{} CK+ TRAIN datapoints loaded".format(len(train_df)))

    if one_hot:
        train_labels = dense_to_one_hot(train_df['emotion'].values, NUM_CLASSES)
        validation_labels = dense_to_one_hot(validation_df['emotion'].values, NUM_CLASSES)
        test_labels = dense_to_one_hot(test_df['emotion'].values, NUM_CLASSES)
    else:
        train_labels = train_df['emotion']
        validation_labels = validation_df['emotion']
        test_labels = test_df['emotion']
    del train_df['emotion']
    del validation_df['emotion']
    del test_df['emotion']

    train_idx = np.arange(len(train_labels))
    validation_idx = np.arange(len(validation_labels))
    test_idx = np.arange(len(test_labels))
    np.random.shuffle(train_idx)
    np.random.shuffle(validation_idx)
    np.random.shuffle(test_idx)

    train_images = train_df.as_matrix()
    validation_images = validation_df.as_matrix()
    test_images = test_df.as_matrix()

    train = DataSet(train_images[train_idx,:], train_labels[train_idx], reshape=False)
    validation = DataSet(validation_images[validation_idx,:], validation_labels[validation_idx], reshape=False)
    test = DataSet(test_images[test_idx,:], test_labels[test_idx], reshape=False)

    return Datasets(train=train, validation=validation, test=test)
