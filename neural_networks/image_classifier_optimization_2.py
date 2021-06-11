import os

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LR = 0.0001
EPOCHS_NUM = 10

training_data = keras.preprocessing.image_dataset_from_directory(
    '../dataset/v5/train', batch_size=32, image_size=(100, 100))

test_data = keras.preprocessing.image_dataset_from_directory(
    '../dataset/v5/test', batch_size=32, image_size=(100, 100))


def train_model(hparams):
    filter_type = hparams[HP_FILTER_TYPES]
    kernel_type_1 = hparams[HP_KERNEL_TYPES_1]
    kernel_type_2 = hparams[HP_KERNEL_TYPES_2]
    filters = FILTER_TYPE_MAP.get(filter_type)
    kernels_1 = KERNEL_TYPE_MAP_1.get(kernel_type_1)
    kernels_2 = KERNEL_TYPE_MAP_2.get(kernel_type_2)

    model = keras.Sequential()
    model.add(keras.Input((100, 100, 3)))
    model.add(Rescaling(1 / 255.0))
    model.add(layers.Conv2D(filters[0], kernels_2[0], activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(filters[1], kernels_2[1], activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(filters[2], kernels_1[0], activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(filters[3], kernels_2[1], activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(20, activation='softmax'))

    for batch_idx, (x, y) in enumerate(training_data):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)

    # Write to TB
    run_dir = (
            "logs/train/"
            + str(filter_type)
            + "filter_"
            + str(kernel_type_1)
            + "kernel1_"
            + str(kernel_type_2)
            + "kernel2"
    )
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = acc_metric.result()
        tf.summary.scalar("accuracy", accuracy, step=1)
    acc_metric.reset_states()


loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
optimizer = keras.optimizers.Adam(lr=LR)

HP_FILTER_TYPES = hp.HParam("filter_types", hp.Discrete([1, 2, 3, 4, 5, 6]))
HP_KERNEL_TYPES_1 = hp.HParam("kernel_types_1", hp.Discrete([1, 2, 3]))
HP_KERNEL_TYPES_2 = hp.HParam("kernel_types_2", hp.Discrete([1, 2, 3]))

FILTER_TYPE_MAP = {
    1: [16, 32, 32, 64],
    2: [16, 32, 64, 64],
    3: [16, 32, 64, 128],
    4: [32, 32, 64, 64],
    5: [32, 32, 64, 128],
    6: [32, 64, 64, 128]
}

KERNEL_TYPE_MAP_1 = {
    1: [3, 3],
    2: [5, 3],
    3: [5, 5]
}

KERNEL_TYPE_MAP_2 = {
    1: [7, 5],
    2: [7, 7],
    3: [9, 7]
}

if __name__ == "__main__":
    for filter_type in HP_FILTER_TYPES.domain.values:
        for kernel_type_1 in HP_KERNEL_TYPES_1.domain.values:
            for kernel_type_2 in HP_KERNEL_TYPES_2.domain.values:
                hparams = {
                    HP_FILTER_TYPES: filter_type,
                    HP_KERNEL_TYPES_1: kernel_type_1,
                    HP_KERNEL_TYPES_2: kernel_type_2
                }
                train_model(hparams)
