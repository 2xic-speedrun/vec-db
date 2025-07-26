import pytest
import time
import numpy as np
from sklearn.datasets import make_blobs
from libvec_db import PyDatabase


# cluster1 = np.random.normal([0, 0], 1, (50, 2))
# cluster2 = np.random.normal([10, 10], 1, (50, 2))

db = PyDatabase(vector_size=128)

for i in range(2048):
    x = np.random.rand((128)).tolist()
    db.insert(x)
