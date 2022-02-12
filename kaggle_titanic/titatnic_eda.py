import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv", index_col="PassengerId")

features = ["Fare"]
targets = ["Survived"]

missing_features = data.isnull().sum(axis=1)

X = data[features]
y = data[targets]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[features], missing_features, y)

ax.set_xlabel('Fare')
ax.set_ylabel('Missing features')
ax.set_zlabel('Survived')

plt.show()
