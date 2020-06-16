"""Simple convolutional neural network classififer."""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras import layers
import tensorflow as tf


# ! you have to add get parameter function

class vgg(tf.keras.Model):
    def __init__(self):
        super(vgg, self).__init__()
        
        
        self.Conv2D_1 = layers.Conv2D(64,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_1 = layers.BatchNormalization()
        self.ReLU_1 = layers.ReLU()
        
        self.Dropout_1 = layers.Dropout(0.3)

        self.Conv2D_2 = layers.Conv2D(64,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_2 = layers.BatchNormalization()
        self.ReLU_2 = layers.ReLU()

        self.MaxPool2D_1 = layers.MaxPooling2D(strides=2)

        self.Conv2D_3 = layers.Conv2D(128,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_3 = layers.BatchNormalization()
        self.ReLU_3 = layers.ReLU()
        
        self.Dropout_2 = layers.Dropout(0.4)
        
        self.Conv2D_4 = layers.Conv2D(128,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_4 = layers.BatchNormalization()
        self.ReLU_4 = layers.ReLU()
        
        self.MaxPool2D_2 = layers.MaxPooling2D(strides=2)

        self.Conv2D_5 = layers.Conv2D(256,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_5 = layers.BatchNormalization()
        self.ReLU_5 = layers.ReLU()
        
        self.Dropout_3 = layers.Dropout(0.4)
        
        self.Conv2D_6 = layers.Conv2D(256,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_6 = layers.BatchNormalization()
        self.ReLU_6 = layers.ReLU()

        self.Dropout_4 = layers.Dropout(0.4)
        
        self.Conv2D_7 = layers.Conv2D(256,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_7 = layers.BatchNormalization()
        self.ReLU_7 = layers.ReLU()
        
        self.MaxPool2D_3 = layers.MaxPooling2D(strides=2)
        
        self.Conv2D_8 = layers.Conv2D(512,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_8 = layers.BatchNormalization()
        self.ReLU_8 = layers.ReLU()
        
        self.Dropout_5 = layers.Dropout(0.4)
        
        self.Conv2D_9 = layers.Conv2D(512,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_9 = layers.BatchNormalization()
        self.ReLU_9 = layers.ReLU()

        self.Dropout_6 = layers.Dropout(0.4)
        
        self.Conv2D_10 = layers.Conv2D(512,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_10 = layers.BatchNormalization()
        self.ReLU_10 = layers.ReLU()
        
        self.MaxPool2D_4 = layers.MaxPooling2D(strides=2)

        self.Conv2D_11 = layers.Conv2D(512,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_11 = layers.BatchNormalization()
        self.ReLU_11 = layers.ReLU()
        
        self.Dropout_7 = layers.Dropout(0.4)
        
        self.Conv2D_12 = layers.Conv2D(512,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_12 = layers.BatchNormalization()
        self.ReLU_12 = layers.ReLU()

        self.Dropout_8 = layers.Dropout(0.4)
        
        self.Conv2D_13 = layers.Conv2D(512,kernel_size=(3,3),kernel_initializer='glorot_normal', padding='same')
        self.BN_13 = layers.BatchNormalization()
        self.ReLU_13 = layers.ReLU()
        
        self.MaxPool2D_5 = layers.MaxPooling2D(strides=2)

        self.Flatten = layers.Flatten()

        #Classifier

        self.Dropout_9 = layers.Dropout(0.5)
        self.Dense_1 = layers.Dense(512)
        self.BN_14 = layers.BatchNormalization()
        self.ReLU_14 = layers.ReLU()
        self.Dropout_10 = layers.Dropout(0.5)
        self.Dense_2 = layers.Dense(10)

    def call(self, inputs):
        x = self.Conv2D_1(inputs)
        x = self.BN_1(x)
        x = self.ReLU_1(x)

        x = self.Dropout_1(x)

        x = self.Conv2D_2(x)
        x = self.BN_2(x)
        x = self.ReLU_2(x)

        x = self.MaxPool2D_1(x)

        x = self.Conv2D_3(x)
        x = self.BN_3(x)
        x = self.ReLU_3(x)

        x = self.Dropout_2(x)

        x = self.Conv2D_4(x)
        x = self.BN_4(x)
        x = self.ReLU_4(x)

        x = self.MaxPool2D_2(x)

        x = self.Conv2D_5(x)
        x = self.BN_5(x)
        x = self.ReLU_5(x)

        x = self.Dropout_3(x)

        x = self.Conv2D_6(x)
        x = self.BN_6(x)
        x = self.ReLU_6(x)

        x = self.Dropout_4(x)

        x = self.Conv2D_7(x)
        x = self.BN_7(x)
        x = self.ReLU_7(x)

        x = self.MaxPool2D_3(x)

        x = self.Conv2D_8(x)
        x = self.BN_8(x)
        x = self.ReLU_8(x)

        x = self.Dropout_5(x)

        x = self.Conv2D_9(x)
        x = self.BN_9(x)
        x = self.ReLU_9(x)

        x = self.Dropout_6(x)

        x = self.Conv2D_10(x)
        x = self.BN_10(x)
        x = self.ReLU_10(x)

        x = self.MaxPool2D_4(x)

        x = self.Conv2D_11(x)
        x = self.BN_11(x)
        x = self.ReLU_11(x)

        x = self.Dropout_7(x)

        x = self.Conv2D_12(x)
        x = self.BN_12(x)
        x = self.ReLU_12(x)

        x = self.Dropout_8(x)

        x = self.Conv2D_13(x)
        x = self.BN_13(x)
        x = self.ReLU_13(x)

        x = self.MaxPool2D_5(x)

        x = self.Flatten(x)

        x = self.Dropout_9(x)
        x = self.Dense_1(x)
        x = self.BN_14(x)
        x = self.ReLU_14(x)
        x = self.Dropout_10(x)
        return self.Dense_2(x)

model = vgg()

