from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def plot(query, found, X, y, filename):
    colors = ["y", "g"]

    n_samples = X.shape[0]

    assert len(found) > 1
    assert len(query) > 0

    X = np.concatenate((X, np.asarray(query).reshape((1, -1)), np.asarray(found)), axis=0)
    X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=12).fit_transform(
        X
    )

    for i in range(n_samples):
        plt.scatter(*X[i], color=colors[y[i]])
    
    for i in range(n_samples, n_samples + len(query)):
        plt.scatter(*X[i], color="red")
        print(X[i])

    for i in range(n_samples + len(query), n_samples + len(found) + len(query)):
        if np.allclose(X[i], X[n_samples + len(query)]):
            continue
        print(X[i])
        plt.scatter(*X[i], color="blue")

    plt.savefig(filename)
