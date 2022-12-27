from libvec_db import PyKmeans
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from helper import plot

classes = 2
n_features = 10
kmeans = PyKmeans(classes, n_features)
n_samples = 1_00


(X, y) = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_classes=classes,
    n_clusters_per_class=1,
    n_informative=1
)

for i in X:
    kmeans.add(i)

mean_vector = -X.mean(axis=0) / 2
mean_vector_times_2 = mean_vector * 2
kmeans.add_centroid(mean_vector)
kmeans.add_centroid(mean_vector_times_2)

kmeans.fit(1_000)

plot(
    kmeans=kmeans,
    X=X,
    y=y,
    classes=classes,
    use_sklearn=True,
    filename="sklearn.png"
)
