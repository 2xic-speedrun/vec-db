from libvec_db import PyKmeans
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from helper import plot


def make_classification(n_samples):
    X, y = [], []

    for i in range(n_samples):
        point_x = np.random.randint(1, 99)
        point_y = np.random.randint(1, 99)
        point_y = np.sqrt(point_y) if point_x < 50 else point_y
        X.append((point_x, point_y))
        y.append(int(point_y < 50 and point_x < 50))

    return np.asarray(X), np.asarray(y)

def get_centroid():
    point_x = np.random.randint(1, 99)
    point_y = np.random.randint(1, 99)

    return [point_x, point_y]

classes = 2
n_features = 2
kmeans = PyKmeans(classes, n_features)
n_samples = 1_00


(X, y) = make_classification(
    n_samples=n_samples
)

for i in X:
    kmeans.add(i)


mean_vector = X.mean(axis=0) / 2
mean_vector_times_2 = mean_vector * 2
kmeans.add_centroid(mean_vector)
kmeans.add_centroid(mean_vector_times_2)

#kmeans.add_centroid(get_centroid())
#kmeans.add_centroid(get_centroid())
kmeans.fit(1_000)

plot(
    kmeans=kmeans,
    X=X,
    y=y,
    classes=classes,
    use_sklearn=False,
    filename="simple.png"
)
