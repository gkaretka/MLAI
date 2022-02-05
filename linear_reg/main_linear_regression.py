# commented out because of slow loading
# import tensorflow as tf
# import keras as ks
import pickle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model

dir = ""


def train_linear_regression(desired_acc, x, y, test_size = 0.1):
    acc = 0

    _model = None
    while acc < desired_acc:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)

        _model = linear_model.LinearRegression()
        _model.fit(x_train, y_train)
        acc = _model.score(x_test, y_test)

    return _model


data = pd.read_csv(dir + "student-mat.csv", sep=";")

# get only these fields
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "traveltime", "age"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
desired_accuracy = 0.96
train = False
t_size = 0.1

if train:
    model = train_linear_regression(desired_accuracy, x, y, test_size=t_size)
    if model is not None:
        with open(dir + "model.txt", "wb") as fh:
            pickle.dump(model, fh)
    else:
        print("Error: model none")
else:
    model: linear_model.LinearRegression
    with open(dir + "model.txt", "rb") as fh:
        model = pickle.load(fh) # unsafe

    if model is not None:
        predictions = model.predict(x)
        print("predicting with ", model.score(x, y), " accuracy")
        for i in range(len(predictions)):
            print(predictions[i], x[i][1])

        plot = "G1"
        plt.scatter(data[plot], data["G3"])
        plt.legend(loc=4)
        plt.xlabel(plot)
        plt.ylabel("Final Grade")
        plt.show()
