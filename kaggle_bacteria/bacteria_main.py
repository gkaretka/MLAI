import keras
import sklearn.model_selection
from sklearn import preprocessing
import numpy as np
import pandas as pd
from keras import layers
import matplotlib
from sklearn.preprocessing import scale


train = False
model_name = "bacteria_model_6.h5"

if train:
    data = pd.read_csv("train.csv")

    # labels
    label = ["target"]
    y = np.array(data[label].values).ravel()
    y_unique = np.unique(y)
    y_unique_cnt = len(y_unique)

    # data
    X = scale(data.iloc[:, 1:len(data.columns)-1].values)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(y_unique_cnt, activation="softmax"))

    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = model.fit(np.array(X_train), y_train, epochs=30, validation_data=[X_test, y_test])

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    matplotlib.pyplot.show()

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
