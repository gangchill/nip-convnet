import os

import pandas as pd

import scipy.misc
from sklearn.model_selection import train_test_split

from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, dense_to_one_hot
from PIL import Image

INPUT_SIZE = (64,49)
NUM_CLASSES = 7

def create_image_thumbnail(img_path):
    thumb_size = INPUT_SIZE
    thumb_path = '.'.join(img_path.split('.')[:-1]) + '_thumb_' + str(thumb_size[0]) + str(thumb_size[1]) + '.png'

    if not os.path.isfile(thumb_path) or True:
        image = Image.open(img_path)
        image.thumbnail(thumb_size, Image.NEAREST)
        if not image.size == INPUT_SIZE:
            # TODO: How to deal with this?
            raise Exception("Image ratio not matching")
        image.save(thumb_path)

    return thumb_path


def read_data_sets(test_size = 50, one_hot=False, frames=3):
    # Load Emotions / Labels
    emotions_path = "datasets/Emotion"
    emotions = []
    for subject in os.listdir(emotions_path):
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
    dataset_path = "datasets/cohn-kanade-images"
    for subject in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, subject)):
            for sequence in os.listdir(os.path.join(dataset_path, subject)):
                if os.path.isdir(os.path.join(dataset_path, subject, sequence)):
                    for pngfile in os.listdir(os.path.join(dataset_path, subject, sequence)):
                        try:
                            # Try to assign corresponding emotion, raises IndexError if not found
                            emotion = [e['emotion'] for e in emotions if e['filename'] == pngfile.split('.')[:-1][0]][0]

                            # Create thumbnail of image
                            img_path = os.path.join(dataset_path, subject, sequence, pngfile)
                            thumb_path = create_image_thumbnail(img_path)

                            image = scipy.misc.imread(thumb_path).flatten()

                            # Normalize pixel values
                            #image = image * (1. / 255)

                            image_dict = dict(enumerate(image))
                            image_dict['emotion'] = emotion - 1
                            data.append(image_dict)
                        except:
                            pass

    df = pd.DataFrame(data)
    print("{} CK+ datapoints loaded".format(len(df)))

    train_df, test_df = train_test_split(df, test_size = test_size)

    if one_hot:
        train_labels = dense_to_one_hot(train_df['emotion'].values, NUM_CLASSES)
        test_labels = dense_to_one_hot(test_df['emotion'].values, NUM_CLASSES)
    else:
        train_labels = train_df['emotion']
        test_labels = test_df['emotion']
    del train_df['emotion']
    del test_df['emotion']

    train_images = train_df.as_matrix()
    test_images = test_df.as_matrix()

    train = DataSet(train_images, train_labels, reshape=False)
    test = DataSet(test_images, test_labels, reshape=False)

    return Datasets(train=train, validation=test, test=None)
