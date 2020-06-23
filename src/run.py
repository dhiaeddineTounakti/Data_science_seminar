# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import vgg as vgg
import provider as provider
import tensorflow as tf
import matplotlib.pyplot as plt
import json
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
            "centered": True,
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
epoch_number = 250
lr_update_epoch_steps = 25



def train_test(optimizer):
    model = vgg.vgg()

    train_data = provider.read(tf.estimator.ModeKeys.TRAIN).batch(batch_size)
    eval_data = provider.read(tf.estimator.ModeKeys.EVAL).batch(batch_size)

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
                            )
                        ], validation_data= eval_data)
    
    model.summary()

    return history


def experiment_optimizer(optimizer, run_number):
    history = train_test(optimizer)
    output_file = open(optimizer + '_' + str(run_number)+'.json', 'w')
    json.dump(str(history.history), output_file, indent= 6)
    output_file.close()


def main():
    experiment_optimizer('Adagrad', 3)
    

if __name__ == "__main__":
    main()
