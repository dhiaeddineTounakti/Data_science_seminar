"""Cifar10 dataset preprocessing and specifications."""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
import pickle
from six.moves import urllib

import cv2
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

REMOTE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
LOCAL_DIR = os.path.join("data/cifar10/")
ARCHIVE_NAME = "cifar-10-python.tar.gz"
DATA_DIR = "cifar-10-batches-py/"
TRAIN_BATCHES = ["data_batch_%d" % (i + 1) for i in range(5)]
TEST_BATCHES = ["test_batch"]

IMAGE_SIZE = 32
NUM_CLASSES = 10

def get_params():
    """Return dataset parameters."""
    return {
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
    }

def prepare():
    """Download the cifar dataset."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
        print("Downloading...")
        urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
    if not os.path.exists(LOCAL_DIR + DATA_DIR):
        print("Extracting files...")
        tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
        tar.extractall(LOCAL_DIR)
        tar.close()
#%%

def read(split):
    """Create an instance of the dataset object and preprocess images"""
    """An iterator that reads and returns images and labels from cifar."""
    batches = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
        tf.estimator.ModeKeys.EVAL: TEST_BATCHES
    }[split]

    all_images = []
    all_labels = []
    
    for batch in batches:
        with open("%s%s%s" % (LOCAL_DIR, DATA_DIR, batch), "rb") as fo:
            diction = pickle.load(fo, encoding='bytes')
            images = np.array(diction[list(diction)[2]])
            labels = np.array(diction[list(diction)[1]])

            num = images.shape[0]
            images = np.reshape(images, [num, 3, IMAGE_SIZE, IMAGE_SIZE])
            images = np.transpose(images, [0, 2, 3, 1])

            print("Loaded %d examples." % num)

            for image in images:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                mean_y = np.mean(image[:,:,0])
                std_y = np.std(image[:,:,0])
                image[:,:,0] = (image[:,:,0]-mean_y)/std_y

            mean_u = np.mean(images[:,:,:,1])
            std_u = np.std(images[:,:,:,1])
            mean_v = np.mean(images[:,:,:,2])
            std_v = np.std(images[:,:,:,2])
            images[:,:,:,1] = (images[:,:,:,1]-mean_u)/std_u
            images[:,:,:,2] = (images[:,:,:,2]-mean_v)/std_v
            all_images.append(images.astype('float32'))
            lb = LabelBinarizer()
            if(split == tf.estimator.ModeKeys.TRAIN):
                all_labels.append(lb.fit_transform(labels))
            else:
                all_labels.append(lb.fit_transform(labels))
    
    # ! you are going to normalize y locally and normalize u and v globally
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)

    return tf.data.Dataset.from_tensor_slices((all_images, all_labels))
#%%
def gaussian1D(size=1, mean=0.25, std=0.5):
    """return a 1D gaussian kernel

    Keyword Arguments:
        size {int} -- size of the kernel (default: {1})
        mean {int} -- the kernel average (default: {0})
        std {int} -- the kernel  (default: {1})

    Returns:
        [type] -- [description]
    """
    assert (size % 2), "make sure that size is an odd number"
    assert (size > 0), "make sure that the size is strict positive"
    kernel = np.array([ i - size//2 for i in range(size)])
    kernel = (1/(np.sqrt(2*np.pi)*std))*np.exp((-1/2)*((kernel - mean)/std)**2)
    return kernel


