import os

import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_data = keras.preprocessing.image_dataset_from_directory(
    '../dataset/v5/train', batch_size=32, image_size=(100, 100))

test_data = keras.preprocessing.image_dataset_from_directory(
    '../dataset/v5/test', batch_size=32, image_size=(100, 100))


def model_builder(hp):
    hp_lr = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    hp_dense_size = hp.Choice('dense_size', values=[512, 1024, 2048])

    model = keras.Sequential([
        keras.Input((100, 100, 3)),
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
        layers.Dense(hp_dense_size, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(hp_dense_size, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(20, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=hp_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # Створюємо оптимізатор гіперпараметрів
    tuner = kt.Hyperband(model_builder,
                         objective='accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='logs',
                         project_name='bug_vision')
    # Налаштовуємо можливість передчасної зупинки оптимізації
    stop_early = callbacks.EarlyStopping(monitor='loss', patience=5)
    # Запускаємо пошук оптимальних значень гіперпараметрів на 10 епохах навчання
    tuner.search(training_data, epochs=10)
    # Отримуємо оптимальні значення гіперпараметрів
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # Виводимо отримані результати
    print(f"""
    The hyperparameter search is complete. 
    Learning rate = {best_hps.get('lr')} 
    and dense layers size = {best_hps.get('dense_size')}.
    """)
