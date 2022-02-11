from tensorflow import keras
import numpy as np
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from tqdm import tqdm

train = True
model_file = "top_model_15.h5"

if train:
    data = pd.read_csv("train.csv")

    label = ["label"]
    y = data[label].values
    X = data.iloc[:, 1:].values
    X = scale(X)

    N_SPLITS = 10
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)
    y_pred_list, y_proba_list, scores = [], [], []

    """ Callbacks """
    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_accuracy',
                                          patience=3,
                                          verbose=1,
                                          factor=0.1,
                                          min_lr=0.00001)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20,
                                                            restore_best_weights=True)

    """ MODEL DEFINITION """
    model = keras.Sequential()

    # data augmentation
    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.07))
    model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.07, width_factor=0.07))

    # model itself
    model.add(keras.layers.Conv2D(filters=64, kernel_size=5, input_shape=(28, 28, 1), padding="valid"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(32, kernel_size=3, padding="valid"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(15, kernel_size=3, padding="valid"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

    h_frames = []
    for fold, (train_id, valid_id) in enumerate(tqdm(folds.split(X, y), total=N_SPLITS)):
        print('####### Fold: ', fold)

        # Splitting
        X_train, y_train = X[train_id], y[train_id]
        X_valid, y_valid = X[valid_id], y[valid_id]

        X_train = X_train.reshape(len(X_train), 28, 28).astype('float32')
        X_valid = X_valid.reshape(len(X_valid), 28, 28).astype('float32')

        X_train = np.expand_dims(X_train, axis=-1)
        X_valid = np.expand_dims(X_valid, axis=-1)
        y_train = np.expand_dims(y_train, axis=-1)
        y_valid = np.expand_dims(y_valid, axis=-1)

        history = model.fit(np.array(X_train), y_train, epochs=5, validation_data=[X_valid, y_valid],
                            callbacks=[early_stopping_callback, reduceLROnPlateau])

        history_frame = pd.DataFrame(history.history)
        h_frames.append(history_frame)

    for history_frame in h_frames:
        history_frame.loc[:, ['loss', 'val_loss']].plot()
        history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()

    matplotlib.pyplot.show()

    print("Done folding!")
    model.save(model_file)
    print("Done!")
