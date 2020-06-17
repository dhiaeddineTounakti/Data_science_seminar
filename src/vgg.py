"""Simple convolutional neural network classififer."""
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense





class vgg(tf.keras.Model):
    def __init__(self):
        super(vgg, self).__init__()

        self.Conv2D_1 = Conv2D(64, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_1 = BatchNormalization()
        self.ReLU_1 = Activation("relu")

        self.Dropout_1 = Dropout(0.3)

        self.Conv2D_2 = Conv2D(64, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_2 = BatchNormalization()
        self.ReLU_2 = Activation("relu")

        self.MaxPool2D_1 = MaxPooling2D(pool_size=(2, 2))

        self.Conv2D_3 = Conv2D(128, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_3 = BatchNormalization()
        self.ReLU_3 = Activation("relu")

        self.Dropout_2 = Dropout(0.4)

        self.Conv2D_4 = Conv2D(128, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_4 = BatchNormalization()
        self.ReLU_4 = Activation("relu")

        self.MaxPool2D_2 = MaxPooling2D(pool_size=(2, 2))

        self.Conv2D_5 = Conv2D(256, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_5 = BatchNormalization()
        self.ReLU_5 = Activation("relu")

        self.Dropout_3 = Dropout(0.4)

        self.Conv2D_6 = Conv2D(256, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_6 = BatchNormalization()
        self.ReLU_6 = Activation("relu")

        self.Dropout_4 = Dropout(0.4)

        self.Conv2D_7 = Conv2D(256, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_7 = BatchNormalization()
        self.ReLU_7 = Activation("relu")

        self.MaxPool2D_3 = MaxPooling2D(pool_size=(2, 2))

        self.Conv2D_8 = Conv2D(512, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_8 = BatchNormalization()
        self.ReLU_8 = Activation("relu")

        self.Dropout_5 = Dropout(0.4)

        self.Conv2D_9 = Conv2D(512, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_9 = BatchNormalization()
        self.ReLU_9 = Activation("relu")

        self.Dropout_6 = Dropout(0.4)

        self.Conv2D_10 = Conv2D(512, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_10 = BatchNormalization()
        self.ReLU_10 = Activation("relu")

        self.MaxPool2D_4 = MaxPooling2D(pool_size=(2, 2))

        self.Conv2D_11 = Conv2D(512, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_11 = BatchNormalization()
        self.ReLU_11 = Activation("relu")

        self.Dropout_7 = Dropout(0.4)

        self.Conv2D_12 = Conv2D(512, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_12 = BatchNormalization()
        self.ReLU_12 = Activation("relu")

        self.Dropout_8 = Dropout(0.4)

        self.Conv2D_13 = Conv2D(512, kernel_size=(
            3, 3), kernel_initializer='glorot_normal', padding='same')
        self.BN_13 = BatchNormalization()
        self.ReLU_13 = Activation("relu")

        self.MaxPool2D_5 = MaxPooling2D(pool_size=(2, 2))

        self.Flatten = Flatten()

        # Classifier

        self.Dropout_9 = Dropout(0.5)
        self.Dense_1 = Dense(512)
        self.BN_14 = BatchNormalization()
        self.ReLU_14 = Activation("relu")
        self.Dropout_10 = Dropout(0.5)
        self.Dense_2 = Dense(10, activation="softmax")

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

