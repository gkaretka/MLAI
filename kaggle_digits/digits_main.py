import keras
import numpy as np
import sklearn.model_selection
from keras import layers
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import scale

train = False
model_file = "top_model_13.h5"

if train:
    data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    label = ["label"]
    y = data[label].values
    X = data.iloc[:, 1:].values

    X = scale(X)
    X = X.reshape(len(X), 28, 28).astype('float32')

    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(np.array(X), y, test_size=0.3)

    model = keras.Sequential()

    # data augmentation
    model.add(tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'))
    model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1))

    # model itself
    model.add(keras.layers.Conv2D(filters=30, kernel_size=5, input_shape=(28, 28, 1), padding='same', activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(15, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.80))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = model.fit(np.array(X_train), y_train, epochs=20, batch_size=300, validation_data=[X_test, y_test])

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    matplotlib.pyplot.show()

    model.save(model_file)
else:
    model: keras.Sequential
    model = keras.models.load_model(model_file)

    test_data = pd.read_csv("test.csv")

    X = test_data.values
    X = scale(X)

    X = X.reshape(len(X), 28, 28).astype('float32')
    X = np.expand_dims(X, axis=-1)

    y = model.predict(X)
    with open("output.csv", "w") as of:
        of.write("ImageId,Label\n")
        for i in range(len(y)):
            of.write(str(i+1) + "," + str(np.argmax(y[i])) + "\n")

    print("Done!")
