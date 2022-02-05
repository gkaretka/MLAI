from sklearn import datasets
from sklearn import model_selection
from sklearn import svm

data = datasets.load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5)

print("\n\n\nlinear")
model = svm.SVC(kernel="linear", C=1)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="linear", C=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="linear", C=3)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="linear", C=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

print("\n\n\nrbf")
model = svm.SVC(kernel="rbf", C=1)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="rbf", C=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="rbf", C=3)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="rbf", C=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

print("\n\n\nsigmoid")
model = svm.SVC(kernel="sigmoid", C=1)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="sigmoid", C=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="sigmoid", C=3)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="sigmoid", C=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

print("\n\n\npoly, deg 2")
model = svm.SVC(kernel="poly", C=1, degree=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="poly", C=2, degree=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="poly", C=3, degree=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="poly", C=4, degree=2)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

print("\n\n\npoly, deg 5")
model = svm.SVC(kernel="poly", C=1, degree=5)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="poly", C=2, degree=5)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="poly", C=3, degree=5)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

model = svm.SVC(kernel="poly", C=4, degree=5)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)
