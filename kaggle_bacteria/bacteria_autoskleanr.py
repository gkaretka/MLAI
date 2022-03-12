import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import autosklearn.classification

print(sklearn.__version__)

cls = autosklearn.classification.AutoSklearnClassifier()

data_test = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2022/test.csv", index_col="row_id")
data = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2022/train.csv", index_col="row_id")

# labels
label = ["target"]
y = np.array(data[label].values).ravel()
y_unique = np.unique(y)
y_unique_cnt = len(y_unique)

# data
X = scale(data.iloc[:, 0:len(data.columns) - 1].values)
X_test = scale(data_test.iloc[:, 0:len(data_test.columns)].values)

print(X.head())
print(X_test.head())

cls.fit(X, y)
predictions = cls.predict(X_test)
