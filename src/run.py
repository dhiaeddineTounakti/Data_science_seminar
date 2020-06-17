# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import vgg as vgg
import provider as provider
import tensorflow as tf
import matplotlib.pyplot as plt
import Vgg_from_internet as vgg_internet

HPARAMS = {
    "Adam": {
        "class_name": "Adam",
        "config": {
            "learning_rate": 0.0003
        }
    },
    "RMSProp": {
        "class_name": "RMSProp",
        "config": {
            "learning_rate": 0.0003,
            "epsilon": 1e-08,
            "centred": True,
            "rho": 0.99
        }
    },
    "Adagrad": {
        "class_name": "Adagrad",
        "config": {
            "epsilon": 1e-10,
            "initial_accumulator_value": 0.0,
            "learning_rate": 0.01
        }

    },
    "SGD": {
        "class_name": "SGD",
        "config": {
            "learning_rate": 0.5,
        }
    },
    "HB": {
        "class_name": "SGD",
        "config": {
            "learning_rate": 0.5,
            "momentum": 0.9
        }

    }
}

batch_size = 128
epoch_number = 2
lr_update_epoch_steps = 25



def train_test(optimizer):
    model = vgg.vgg()

    train_data = provider.read(tf.estimator.ModeKeys.TRAIN).batch(batch_size)
    eval_data = provider.read(tf.estimator.ModeKeys.EVAL).batch(batch_size)

    checkpoint_filepath = './models/'+optimizer+'.weights.{epoch:02d}-{val_loss:.2f}.hdf5' 

    model.compile(
    optimizer=tf.keras.optimizers.get(HPARAMS[optimizer]),  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=['accuracy'])

    history = model.fit(train_data, batch_size=batch_size,
                        epochs=epoch_number, callbacks=[
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto',
                                min_delta=0.0001, cooldown=0, min_lr=0
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                save_weights_only=True, mode='auto', save_freq='epoch'
                            )
                        ], validation_data= eval_data)
    
    model.summary()

    return history


def experiment_optimizer(optimizer):
    history = train_test(optimizer)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    experiment_optimizer('Adam')

if __name__ == "__main__":
    main()
