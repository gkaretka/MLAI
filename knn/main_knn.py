import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier

dir = ""


def get_best_run(x_train, x_test, y_train, y_test, max_distance=51):
    acc = 0
    k = 0
    for i in range(1, max_distance, 2):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)

        _acc = model.score(x_test, y_test)
        if _acc > acc:
            acc = _acc
            k = i

        predicted = model.predict(x_test)

        names = ["unacc", "acc", "good", "vgood"]
        """for i in range(len(predicted)):
            print("Predicted:", names[predicted[i]], ", original: ", names[y_test[i]])
        """
    print("Best run:", k, " accuracy: ", acc)


data = pd.read_csv(dir + "car.data")
print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(data["buying"])
maint = le.fit_transform(data["maint"])
doors = le.fit_transform(data["doors"])
persons = le.fit_transform(data["persons"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
cls = le.fit_transform(data["class"])

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)


for i in range(100):
    x_tn, x_ts, y_tr, y_ts = model_selection.train_test_split(x, y, test_size=0.1)
    get_best_run(x_tn, x_ts, y_tr, y_ts)
