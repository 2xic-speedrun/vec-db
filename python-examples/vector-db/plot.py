from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def find_index(a, b):
    for index, v in enumerate(a):
        if v == b:
            return index
    return -1

def plot(query_vec, close_vecs, X, y, filename):
    colors = ["y", "g"]

    n_samples = X.shape[0]

    assert len(close_vecs) >= 1
    assert len(query_vec) > 0

    X_org = X.copy().tolist()
    X = np.concatenate((X, np.asarray(query_vec).reshape((1, -1)), np.asarray(close_vecs)), axis=0)
    X = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

    for i in range(n_samples):
        plt.scatter(*X[i], color=colors[y[i]])
    
    index = find_index(X_org, query_vec[0].tolist())
    assert index != -1
    plt.scatter(*X[index], color="red")

    for i in range(len(close_vecs)):
        index = find_index(X_org, close_vecs[i])
        assert index != -1
        plt.scatter(*X[index], color="blue")

    plt.savefig(filename)
