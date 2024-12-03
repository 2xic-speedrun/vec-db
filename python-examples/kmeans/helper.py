from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def plot(kmeans, X, y, classes, use_sklearn, filename=None):
    colors = ["y", "g"]
    centroids = [
        kmeans.get_centroid(i) for i in range(classes)
    ]

    n_samples = X.shape[0]

    X = np.concatenate((X, np.asarray(centroids)), axis=0)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(
        X
    ) if use_sklearn else X

    for i in range(n_samples):
        plt.scatter(*X_embedded[i], color=colors[y[i]])

    for i in range(n_samples, n_samples + len(centroids)):
        plt.scatter(*X_embedded[i], color="red")

    if not use_sklearn:
        plt.xlim(0, 100)
        plt.ylim(0, 100)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
