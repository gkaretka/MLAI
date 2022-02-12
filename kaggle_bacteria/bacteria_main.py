import keras
import matplotlib
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn import preprocessing
import numpy as np
import pandas as pd
from keras import layers
from sklearn.preprocessing import scale
import tensorflow as tf

train = True
save = True
use_cnn = True
show_data = False
model_name = "bacteria_model_19.h5"

if train:
    data = pd.read_csv("train.csv", index_col="row_id")

    # labels
    label = ["target"]
    y = np.array(data[label].values).ravel()
    y_unique = np.unique(y)
    y_unique_cnt = len(y_unique)

    # data
    X = scale(data.iloc[:, 0:len(data.columns) - 1].values)

    # reshape and fix
    def my_reshape(x):
        return x.reshape(len(X), 13, 22)
    X = my_reshape(X)
    X = np.expand_dims(X, axis=-1)

    if show_data:
        fig, axes = plt.subplots(
            nrows=3, ncols=3, figsize=(3, 3)
        )
        index = 111
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                print(i+(j*3))
                ax.imshow(X[i+(j*3)])
        plt.show()

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    y = np.expand_dims(y, axis=-1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    model = keras.Sequential()

    """ Preprocessor """
    # model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.10))
    model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.06, width_factor=0.04))
    model.add(keras.layers.GaussianNoise(input_shape=(13, 22, 1), stddev=0.1))

    """ Model """
    model.add(keras.layers.Conv2D(filters=64, kernel_size=5, input_shape=(13, 22, 1), activation="relu", padding="valid"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="valid"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(y_unique_cnt, activation="softmax"))

    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = model.fit(X_train, y_train, epochs=20, validation_data=[X_test, y_test])

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    matplotlib.pyplot.show()

    if save:
        model.save(model_name)
else:
    model: keras.Sequential
    model = keras.models.load_model(model_name)

    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")

    # row_id
    label = ["row_id"]
    row_id = np.array(test_data[label].values)

    # data
    if use_cnn:
        # data
        X = scale(test_data.iloc[:, 1:len(test_data.columns)].values)

        # reshape and fix
        def my_reshape(x):
            return x.reshape(len(x), 13, 22)

        X = my_reshape(X)
        X = np.expand_dims(X, axis=-1)
    else:
        X = scale(test_data.iloc[:, 1:len(test_data.columns)].values)

    # labels
    label = ["target"]
    y = np.array(train_data[label].values).ravel()
    le = preprocessing.LabelEncoder()
    le.fit(y)

    res = model.predict(X)
    results = []
    for i in range(len(res)):
        results.append(np.argmax(res[i]))

    results = le.inverse_transform(results)

    with open("output.csv", "w") as of:
        of.write("row_id,target\n")
        for i in range(len(results)):
            of.write(str(row_id[i][0]) + "," + str(results[i]) + "\n")

    print("Done!")
