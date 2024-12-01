from libvec_db import PyDatabase
from sklearn import datasets

# We have many samples, and will stream them over time!
n_features = 100
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

# Insert all the entries
for i in X:
    database.insert(i)

# Store the entries
database.dump()

# Now we want to do a lookup. 
database = PyDatabase()
database.load()
print(database.query(X[0]))
