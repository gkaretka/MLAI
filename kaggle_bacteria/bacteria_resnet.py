import keras
import matplotlib
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tensorflow.keras import layers


train = True
save = True
use_cnn = True
show_data = True
model_name = "res_net_bacteria_model_0.h5"
model_dir = "resnet_models/"

if train:
    data = pd.read_csv("train.csv", index_col="row_id")

    # labels
    label = ["target"]
    y = np.array(data[label].values).ravel()
    y_unique = np.unique(y)
    y_unique_cnt = len(y_unique)

    # data
    X = data.iloc[:, 0:len(data.columns) - 1].values
    X = scale(X)

    # turn into toon
    """for i in range(len(X)):
        _temp = np.array(X[i])
        exit(1)
    """

    # reshape and fix
    def my_reshape(par):
        return par.reshape(len(X), 13, 22).astype("float64")


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


    def res_net_block(input_data, filters, conv_size):
        x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, input_data])
        x = layers.Activation('relu')(x)
        return x

    inputs = keras.Input(shape=(13, 22, 1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = model.fit(X_train, y_train, epochs=1, batch_size=800, validation_data=[X_test, y_test])

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    matplotlib.pyplot.show()

    if save:
        model.save(model_dir + model_name)

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

    with open("resnet_output.csv", "w") as of:
        of.write("row_id,target\n")
        for i in range(len(results)):
            of.write(str(row_id[i][0]) + "," + str(results[i]) + "\n")

    print("Done!")
