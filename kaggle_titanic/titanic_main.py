import pandas as pd
import numpy as np
import sklearn.model_selection
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, ExtraTreeRegressor
from tensorflow import keras

""" Dataset description
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
"""

data = pd.read_csv("train.csv", index_col="PassengerId")

features = ["Pclass", "Sex", "SibSp", "Parch"]
targets = ["Survived"]

missing_features_cnt = pd.DataFrame(data=data[features].isnull().sum(axis=1), columns=["missing_features"])

# data and labels
X = pd.get_dummies(data[features])
X = pd.concat([X, missing_features_cnt], axis=1)

print(X.head())

y = data[targets]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# model = KNeighborsClassifier(n_neighbors=10) # 0.75
# model = SVC(C=1) # 0.67
# model = SVC(C=5) # 0.78
# model = SVC(C=10) # 0.86
# model = SVC(C=30) # 0.86
# model = SVC(C=100) # 0.84
# model = DecisionTreeClassifier() # 0.80
# model = ExtraTreeClassifier(max_leaf_nodes=3000) # 0.75
# model = ExtraTreeRegressor() # 0.26
# model = LogisticRegression() # 0.81
# model = LinearRegression()

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

""" ExtraTreesClasifier
model = ExtraTreesClassifier(
    n_estimators=300,
    n_jobs=-1,
    verbose=0,
    random_state=1
)
"""

""" NN
model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="sigmoid"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", metrics=["accuracy"], loss="categorical_crossentropy")
model.fit(X_train, y_train, epochs=10, validation_data=[X_test, y_test])
"""

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(acc)

""" PREDICTING """

data_test = pd.read_csv("test.csv", index_col="PassengerId")
missing_features_cnt = pd.DataFrame(data=data_test[features].isnull().sum(axis=1), columns=["missing_features"])

# data and labels
test_X = pd.get_dummies(data_test[features])
test_X = pd.concat([test_X, missing_features_cnt], axis=1)

predictions = model.predict(test_X)

output = pd.DataFrame({'PassengerId': data_test.index, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Done!")
