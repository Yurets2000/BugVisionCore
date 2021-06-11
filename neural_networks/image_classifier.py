import os
import shutil

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
CURR_DATASET_PATH = CURR_PATH + '/../dataset/' + Config.CURR_SPLIT_DATASET
CURR_MODEL_PATH = CURR_PATH + '/../neural_networks/' + Config.CURR_MODEL

LR = 0.0001
EPOCHS_NUM = 15


def get_model(image_size, classes_num):
    model = keras.Sequential([
        keras.Input(image_size),
        Rescaling(1 / 255.0),
        layers.Conv2D(16, kernel_size=(9, 9), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(classes_num, activation='softmax')
    ])
    return model


def get_model_2(image_size, classes_num):
    model = keras.Sequential([
        keras.Input(image_size),
        Rescaling(1 / 255.0),
        layers.Conv2D(16, kernel_size=(7, 7), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(classes_num, activation='softmax')
    ])
    return model


def get_model_3(image_size, classes_num):
    model = keras.Sequential([
        keras.Input(image_size),
        Rescaling(1 / 255.0),
        layers.Conv2D(16, kernel_size=(9, 9), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(classes_num, activation='softmax')
    ])
    return model


def train_model(dataset_path=CURR_DATASET_PATH,
                write_path=CURR_MODEL_PATH,
                lr=LR,
                epochs_num=EPOCHS_NUM,
                image_size=(100, 100, 3),
                classes_num=20):
    training_data = keras.preprocessing.image_dataset_from_directory(
        dataset_path + '/train', batch_size=32, image_size=image_size[:2])
    test_data = keras.preprocessing.image_dataset_from_directory(
        dataset_path + '/test', batch_size=32, image_size=image_size[:2])
    if (os.path.exists(write_path)):
        shutil.rmtree(write_path)
    model = get_model(image_size=image_size, classes_num=classes_num)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(training_data, epochs=epochs_num, verbose=2)
    model.evaluate(test_data, verbose=2)
    model.save(write_path)


if __name__ == "__main__":
    keras.utils.plot_model(get_model((100, 100, 3), 20), "plots/model_1.png", show_shapes=True, dpi=120)
    keras.utils.plot_model(get_model_2((100, 100, 3), 20), "plots/model_2.png", show_shapes=True, dpi=120)
    # TRAINING_DATA = keras.preprocessing.image_dataset_from_directory(
    #     CURR_DATASET_PATH + '/train', batch_size=32, image_size=(100, 100))
    #
    # TEST_DATA = keras.preprocessing.image_dataset_from_directory(
    #     CURR_DATASET_PATH + '/test', batch_size=32, image_size=(100, 100))
    #
    # model = get_model(image_size=(100, 100, 3), classes_num=20)
    # print(model.summary())
    #
    # model.compile(
    #     optimizer=keras.optimizers.Adam(lr=LR),
    #     loss=keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=['accuracy']
    # )
    #
    # model.fit(TRAINING_DATA, epochs=EPOCHS_NUM, verbose=2)
    # model.evaluate(TEST_DATA, verbose=2)
    #
    # model.save('model_1')
