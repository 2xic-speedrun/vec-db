import pytest
import time
import numpy as np
from sklearn.datasets import make_blobs
from libvec_db import PyDatabase


def test_similarity_quality():
    cluster1 = np.random.normal([0, 0], 1, (50, 2))
    cluster2 = np.random.normal([10, 10], 1, (50, 2))

    db = PyDatabase.with_kmeans_backend(vector_size=2)

    # Insert both clusters
    for vector in np.vstack([cluster1, cluster2]):
        db.insert(vector.tolist())

    # Query with point from cluster1
    query_point = cluster1[0]
    results = db.query(query_point.tolist(), 3)

    # Should always return results when DB has data
    assert len(results) > 0, "Should return results when database has data"
    assert len(results) <= 3, "Should not return more than requested"

    # Calculate average distance to results
    result_vectors = np.array(results)
    distances = [np.linalg.norm(query_point - result) for result in result_vectors]
    avg_distance = np.mean(distances)

    # Results should be closer than the distance between clusters (~14)
    assert avg_distance < 7, f"Results too far from query point: {avg_distance}"


@pytest.mark.parametrize("size", [25, 100, 500])
def test_performance_scaling(size):
    """Test that performance doesn't degrade too badly with size."""
    insert_times = []
    query_times = []

    centers = np.random.RandomState(42).rand(5, size)
    X, _ = make_blobs(n_samples=size, n_features=size, centers=centers, random_state=42)
    db = PyDatabase.with_kmeans_backend(vector_size=size)

    # Time insertions
    start = time.time()
    for vector in X:
        db.insert(vector.tolist())
    insert_time = time.time() - start
    insert_times.append(insert_time)

    # Time query
    start = time.time()
    results = db.query(X[0].tolist(), 5)
    query_time = time.time() - start
    query_times.append(query_time)
    assert len(results) >= 0

    # Basic performance expectations (adjust as you improve the DB)
    assert all(t < 5 for t in insert_times), "Insert times too slow"
    assert all(t < 5 for t in query_times), "Query times too slow"


def test_empty_database():
    """Test behavior with empty database."""
    db = PyDatabase.with_kmeans_backend(vector_size=10)

    query = [1.0] * 10
    results = db.query(query, 5)

    # Should handle empty DB gracefully
    assert isinstance(results, list)
    assert len(results) == 0


if __name__ == "__main__":
    test_similarity_quality()
    test_performance_scaling()
    test_empty_database()
    print("All tests passed!")
