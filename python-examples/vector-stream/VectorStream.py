from libvec_db import PyDatabase
from sklearn import datasets

n_features = 100
# We have many samples, and will stream them over time!
n_samples = 10#_000
classes = 10


(X, y) = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_classes=classes,
    n_clusters_per_class=1,
    n_informative=5,
    shuffle=True
)

database = PyDatabase()
database.load()

for i in X:
    database.save(i)
database.dump()
