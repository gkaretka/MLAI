import keras
import numpy as np
import sklearn.model_selection
from keras import layers
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import scale

train = True
model_file = "top_model_resnet_0.h5"

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

    # model = keras.Sequential()
    inputs = keras.Input(shape=(28, 28, 1))

    # data augmentation
    x = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')(inputs)
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)


    def res_net_block(input_data, filters, conv_size):
        x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, input_data])
        x = layers.Activation('relu')(x)
        return x

    # model itself
    x = keras.layers.Conv2D(filters=64, kernel_size=5, input_shape=(28, 28, 1), padding='same', activation="relu")(inputs)
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(50, activation='relu')(x)

    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = model.fit(np.array(X_train), y_train, epochs=20, validation_data=[X_test, y_test])

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
