import sklearn.preprocessing as preprocessing
from sklearn import datasets
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


def bench_k_means(estimator, name, _data):
    estimator.fit(_data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(_data, estimator.labels_,
                                      metric='euclidean')))


digits = datasets.load_digits()

data = preprocessing.scale(digits.data)
y = digits.target

k = len(np.unique(y))
print("K: ", k)

samples, features = data.shape

bench_k_means(KMeans(init='k-means++', n_clusters=k, n_init=10),
              name="k-means++", _data=data)

for _ in range(100):
    bench_k_means(KMeans(init='random', n_clusters=k, n_init=10),
                  name="random", _data=data)