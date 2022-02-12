import keras
from scipy.stats import mode
from sklearn.ensemble import ExtraTreesClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import remove_if_duplicates_weights
import numpy as np
import pandas as pd

data_test = pd.read_csv("test.csv", index_col="row_id")

data = pd.read_csv("train.csv", index_col="row_id")
data, weights = remove_if_duplicates_weights(data)

# Encoding categorical features
le = LabelEncoder()

TARGET = "target"
features = data.columns[data.columns != TARGET]

X = data[features]
y = pd.DataFrame(le.fit_transform(data[TARGET]), columns=[TARGET])
n_speciments = len(np.unique(y))
print("Number of different speciments: ", n_speciments)

N_SPLITS = 2
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)
y_pred_list, y_proba_list, scores = [], [], []

for fold, (train_id, valid_id) in enumerate(tqdm(folds.split(X, y), total=N_SPLITS)):
    print("####### Fold: ", fold)

    # Splitting
    X_train, y_train, sample_weight_train = X.iloc[train_id],  np.array(y.iloc[train_id]).ravel(), weights[train_id]
    X_valid, y_valid, sample_weight_valid = X.iloc[valid_id],  np.array(y.iloc[valid_id]).ravel(), weights[valid_id]

    # Model
    """
    model = ExtraTreesClassifier(
        n_estimators=300,
        n_jobs=-1,
        verbose=0,
        random_state=1
    )
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(n_speciments, activation="softmax"))
    model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

    # Training
    model.fit(X_train, y_train, sample_weight=sample_weight_train, epochs=10)

    # Validation
    valid_pred = model.predict(X_valid)
    valid_res = []
    for i in range(len(valid_pred)):
        valid_res.append(np.argmax(valid_pred[i]))

    valid_score = accuracy_score(y_valid, valid_res, sample_weight=sample_weight_valid)
    print(f"Accuracy score: {valid_score:5f}\n")
    scores.append(valid_score)

    # Prediction for submission
    y_pred_list.append(model.predict(data_test))
    y_proba_list.append(model.predict(data_test))

score = np.array(scores).mean()
print(f"Mean accuracy score: {score:6f}")

# Majority vote
y_pred = mode(y_pred_list).mode[0]
_y_pred = []
for i in range(len(y_pred)):
    _y_pred.append(np.argmax(y_pred[i]))
y_pred = le.inverse_transform(_y_pred)

target_distrib = pd.DataFrame({
    "count": data.target.value_counts(),
    "share": data[TARGET].value_counts() / data.shape[0] * 100
})

target_distrib["pred_count"] = pd.Series(y_pred, index=data_test.index).value_counts()
target_distrib["pred_share"] = target_distrib["pred_count"] / len(data_test) * 100
target_distrib.sort_index()

print(target_distrib["pred_share"].values)

tun_str = input("Insert tuning values (10): ")
tun_arr = tun_str.split(" ")
for i in range(len(tun_arr)):
    tun_arr[i] = float(tun_arr[i])

y_proba = sum(y_proba_list) / len(y_proba_list)
y_proba += np.array(tun_arr)
y_pred_tuned = le.inverse_transform(np.argmax(y_proba, axis=1))
pd.Series(y_pred_tuned, index=data_test.index).value_counts().sort_index() / len(data_test) * 100

submission = pd.read_csv("output.csv")
submission[TARGET] = y_pred_tuned
submission.to_csv("submission.csv", index=False)
