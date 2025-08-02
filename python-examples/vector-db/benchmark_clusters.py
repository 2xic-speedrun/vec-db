#!/usr/bin/env python3
"""
Benchmark script to evaluate vector database insert speed with multiple clusters.
Tests performance when inserting 10,000 items across 1,000 clusters (10 items per cluster).
"""

import time
import numpy as np
from libvec_db import PyDatabase


def generate_clustered_data(n_clusters=1000, items_per_cluster=10, vector_size=128):
    """Generate clustered data for benchmarking."""
    cluster_centers = np.random.rand(n_clusters, vector_size) * 100

    all_vectors = []

    for cluster_id in range(n_clusters):
        cluster_data = np.random.normal(
            loc=cluster_centers[cluster_id],
            scale=1.0,
            size=(items_per_cluster, vector_size),
        )
        all_vectors.extend(cluster_data)

    return np.array(all_vectors)


def benchmark_insert_performance():
    """Benchmark insert performance for clustered data."""
    n_clusters = 1000
    items_per_cluster = 10
    vector_size = 128
    total_items = n_clusters * items_per_cluster

    print(f"Benchmarking {total_items} items across {n_clusters} clusters...")

    vectors = generate_clustered_data(n_clusters, items_per_cluster, vector_size)
    db = PyDatabase(vector_size=vector_size)

    insert_times = []
    overall_start = time.time()

    for index, vector in enumerate(vectors):
        insert_start = time.time()
        db.insert(vector.tolist())
        insert_time = time.time() - insert_start
        insert_times.append(insert_time)
        if index % 100 == 0:
            print(f"Centroids count: {len(db.centroids())}, index: {index}")

    overall_time = time.time() - overall_start

    avg_insert_time = np.mean(insert_times)
    p90_insert_time = np.percentile(insert_times, 90)
    p99_insert_time = np.percentile(insert_times, 99)
    throughput = total_items / overall_time

    print(f"Total time: {overall_time:.3f}s")
    print(f"Throughput: {throughput:.1f} items/second")
    print(f"Average insert: {avg_insert_time * 1000:.3f}ms")
    print(f"P90 insert: {p90_insert_time * 1000:.3f}ms")
    print(f"P99 insert: {p99_insert_time * 1000:.3f}ms")


if __name__ == "__main__":
    benchmark_insert_performance()
