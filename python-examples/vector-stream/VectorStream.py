from libvec_db import PyDatabase
from sklearn import datasets
from plot import plot

# We have many samples, and will stream them over time!
n_features = 100
n_samples = 30
classes = 2


(X, y) = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_classes=classes,
    n_clusters_per_class=1,
    n_informative=15,
    shuffle=True
)

database = PyDatabase()

# Insert all the entries
for i in X:
    database.insert(i)

# Results
# TODO: add test to make the output is actually close to the rest.
print(len(database.query(X[0], 3)))
print(len(database.centroids()))

plot(
    [X[0]],
    database.query(X[0], 3),
    X,
    y,
    "similar.png"
)
