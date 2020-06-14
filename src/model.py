"""Simple convolutional neural network classififer."""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras import layers
import tensorflow as tf

FLAGS = tf.compat.v1.flags



def model():
    """VGG classifier """

    vgg = tf.keras.Sequential()

    def ConvBNReLU(input=-1, output=0):
        """this is the building block of the Vgg

        Args:
            input (int): number of input plane
            output (int): number of output plane

        Returns:
            None: only updates the model initialized above
        """
        if (input != -1):
            vgg.add(layers.Conv2D(output[2],kernel_size=(3,3),kernel_initializer='glorot_normal', input_shape=input, padding='same'))
            vgg.add(layers.BatchNormalization())
            vgg.add(layers.ReLU())
    
    
    ConvBNReLU((32,32,3),(32,32,64))
    vgg.add(layers.Dropout(0.3))
    ConvBNReLU((32,32,64),(32,32,64))
    vgg.add(layers.MaxPooling2D(strides=2))

    ConvBNReLU((32,32,64),(16,16,128))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((16,16,128),(16,16,128))
    vgg.add(layers.MaxPooling2D(strides=2))

    ConvBNReLU((16,16,128),(8,8,256))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((8,8,256),(8,8,256))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((8,8,256),(8,8,256))
    vgg.add(layers.MaxPooling2D(strides=2))

    ConvBNReLU((8,8,256),(4,4,512))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((4,4,512),(4,4,512))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((4,4,512),(4,4,512))
    vgg.add(layers.MaxPooling2D(strides=2))

    ConvBNReLU((4,4,512),(2,2,512))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((2,2,512),(2,2,512))
    vgg.add(layers.Dropout(0.4))
    ConvBNReLU((2,2,512),(2,2,512))
    vgg.add(layers.MaxPooling2D(strides=2))
    vgg.add(layers.Flatten())

    #Classifier

    vgg.add(layers.Dropout(0.5))
    vgg.add(layers.Dense(512, input_shape=(512,)))
    vgg.add(layers.BatchNormalization())
    vgg.add(layers.ReLU())
    vgg.add(layers.Dropout(0.5))
    vgg.add(layers.Dense(10,input_shape=(512,)))




    return vgg

def eval_metrics(unused_params):
    """Eval metrics."""
    return {
        "accuracy": tf.contrib.learn.MetricSpec(tf.metrics.accuracy)
    }
