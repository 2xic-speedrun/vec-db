use crate::math::vector::Vector;
use anyhow::Result;
use std::{cmp::min, collections::HashMap};

pub struct Kmeans {
    centroids: Vec<Vector>,
    inertia_distance_centroids: Vec<f64>,
    dataset: Vec<Vector>,
    vector_length: usize,
    fit_index: i64,
}

struct ClusterStats {
    avg_cluster_variance: f64,
    max_cluster_variance: f64,
    largest_cluster_size: usize,
}

impl Kmeans {
    pub fn new(vector_length: usize) -> Kmeans {
        Kmeans {
            dataset: Vec::new(),
            centroids: vec![Vector::rand(vector_length), Vector::rand(vector_length)],
            inertia_distance_centroids: Vec::new(),
            vector_length,
            fit_index: 0,
        }
    }

    pub fn add_centroid(&mut self, vector: Vector) {
        self.centroids.push(vector);
    }

    pub fn add_datapoint(&mut self, vector: Vector) {
        self.dataset.push(vector);
    }

    pub fn centroids(&self) -> &Vec<Vector> {
        &self.centroids
    }

    pub fn dataset(&self) -> &Vec<Vector> {
        &self.dataset
    }

    pub fn fit(&mut self, iterations: u64) -> Result<()> {
        let mut previous_centroids = self.centroids.clone();
        const CONVERGENCE_THRESHOLD: f64 = 1e-8;

        for _ in 0..iterations {
            self.forward()?;

            let mut total_movement = 0.0;
            for (new_centroid, old_centroid) in self.centroids.iter().zip(&previous_centroids) {
                total_movement += new_centroid.l2_distance(old_centroid)?;
            }

            if total_movement < CONVERGENCE_THRESHOLD {
                break;
            }

            previous_centroids = self.centroids.clone();

            if self.fit_index > 0 && self.should_add_centroid()? {
                self.add_centroid(Vector::rand(self.vector_length));
            }
            self.fit_index += 1
        }
        Ok(())
    }

    pub fn results(&self) -> Result<Vec<(&Vector, i8)>> {
        let mut results: Vec<(&Vector, i8)> = Vec::new();
        for i in self.dataset.iter() {
            let closest = self.find_closest_centroid(i)? as i8;
            results.push((i, closest));
        }
        Ok(results)
    }

    fn find_closest_centroid(&self, data_point: &Vector) -> Result<usize> {
        let mut best_score = f64::INFINITY;
        let mut best_index: usize = 0;

        for (index, centroid) in self.centroids.iter().enumerate() {
            let distance = centroid.l2_distance_squared(data_point).map_err(|e| {
                anyhow::anyhow!("Failed to calculate distance to centroid {}: {}", index, e)
            })?;

            if distance < best_score {
                best_score = distance;
                best_index = index;
            }
        }

        Ok(best_index)
    }

    pub fn find_closest_data_points(
        &self,
        data_point: &Vector,
        n: usize,
    ) -> anyhow::Result<Vec<Vec<f64>>> {
        let mut results: Vec<Vec<f64>> = Vec::new();

        let clustered_data_pints = self.get_centroids_data_point()?;
        let centroid = self.find_closest_centroid(data_point)?;
        let items = clustered_data_pints.get(&centroid);
        if let Some(items) = items {
            let max_index = min(n, items.len());

            for vector in items.iter().take(max_index) {
                if !vector.equal(data_point) {
                    results.push(vector.raw().clone());
                }
            }
        }

        Ok(results)
    }

    fn forward(&mut self) -> Result<()> {
        let mut assignments = Vec::with_capacity(self.dataset.len());
        let mut distances = Vec::with_capacity(self.dataset.len());

        for data_point in &self.dataset {
            let (closest_idx, closest_dist) =
                self.find_closest_centroid_with_distance(data_point)?;
            assignments.push(closest_idx);
            distances.push(closest_dist);
        }

        let mut new_centroids = vec![Vector::empty(self.vector_length); self.centroids.len()];
        let mut cluster_counts = vec![0; self.centroids.len()];
        let mut new_inertia_distance = vec![0.0; self.centroids.len()];

        for ((data_point, &cluster_id), &distance) in
            self.dataset.iter().zip(&assignments).zip(&distances)
        {
            new_centroids[cluster_id].add_inplace(data_point)?;
            cluster_counts[cluster_id] += 1;
            new_inertia_distance[cluster_id] += distance;
        }

        for (i, &count) in cluster_counts.iter().enumerate() {
            if count > 0 {
                new_centroids[i] = new_centroids[i].div_constant(count as f64);
            }
        }

        self.centroids = new_centroids;
        self.inertia_distance_centroids = new_inertia_distance;
        Ok(())
    }

    fn find_closest_centroid_with_distance(&self, data_point: &Vector) -> Result<(usize, f64)> {
        let mut best_score = f64::INFINITY;
        let mut best_index: usize = 0;

        for (index, centroid) in self.centroids.iter().enumerate() {
            let distance = centroid.l2_distance_squared(data_point)?;
            if distance < best_score {
                best_score = distance;
                best_index = index;
            }
        }

        Ok((best_index, best_score))
    }

    fn get_centroids_data_point(&self) -> Result<HashMap<usize, Vec<&Vector>>> {
        let mut clustered_data_pints: HashMap<usize, Vec<&Vector>> = HashMap::new();

        for data_point in self.dataset.iter() {
            let best_index = self.find_closest_centroid(data_point)?;

            match clustered_data_pints.get_mut(&best_index) {
                Some(value) => {
                    value.push(data_point);
                }
                None => {
                    let vec: Vec<&Vector> = vec![data_point];
                    clustered_data_pints.insert(best_index, vec);
                }
            };
        }

        Ok(clustered_data_pints)
    }

    fn should_add_centroid(&self) -> Result<bool> {
        if self.dataset.len() < 10 {
            return Ok(false);
        }

        let min_points_per_centroid = 6;
        if self.dataset.len() < self.centroids.len() * min_points_per_centroid {
            return Ok(false);
        }

        let cluster_stats = self.analyze_cluster_dispersion()?;
        let cluster_sizes = self.get_cluster_sizes()?;

        let min_cluster_size = 3;
        let has_tiny_clusters = cluster_sizes.iter().any(|&size| size < min_cluster_size);
        if has_tiny_clusters {
            return Ok(false); // Don't add more centroids if we have tiny clusters
        }

        let variance_ratio = if cluster_stats.avg_cluster_variance > 0.0 {
            cluster_stats.max_cluster_variance / cluster_stats.avg_cluster_variance
        } else {
            1.0
        };

        let expected_cluster_size = self.dataset.len() / self.centroids.len();
        let size_ratio = cluster_stats.largest_cluster_size as f64 / expected_cluster_size as f64;

        let has_significant_size_imbalance = size_ratio > 1.5;
        let has_high_absolute_variance = cluster_stats.max_cluster_variance > 100.0;
        let has_variance_imbalance = variance_ratio > 1.3;

        let should_add =
            has_significant_size_imbalance || has_high_absolute_variance || has_variance_imbalance;

        Ok(should_add)
    }

    fn get_cluster_sizes(&self) -> Result<Vec<usize>> {
        let clustered_data = self.get_centroids_data_point()?;
        Ok(clustered_data.values().map(|points| points.len()).collect())
    }

    fn analyze_cluster_dispersion(&self) -> Result<ClusterStats> {
        let clustered_data = self.get_centroids_data_point()?;
        let mut cluster_variances = Vec::new();
        let mut cluster_sizes = Vec::new();

        for (centroid_idx, points) in clustered_data.iter() {
            if points.is_empty() {
                continue;
            }

            let centroid = &self.centroids[*centroid_idx];
            let mut variance = 0.0;

            for point in points {
                variance += centroid.l2_distance_squared(point)?;
            }

            variance /= points.len() as f64;
            cluster_variances.push(variance);
            cluster_sizes.push(points.len());
        }

        let avg_variance = if cluster_variances.is_empty() {
            0.0
        } else {
            cluster_variances.iter().sum::<f64>() / cluster_variances.len() as f64
        };

        let max_variance = cluster_variances.iter().fold(0.0, |a: f64, &b| a.max(b));
        let max_size = cluster_sizes.iter().fold(0, |a, &b| a.max(b));

        Ok(ClusterStats {
            avg_cluster_variance: avg_variance,
            max_cluster_variance: max_variance,
            largest_cluster_size: max_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vector::Vector;

    #[test]
    fn test_centroid_addition_with_dispersed_clusters() -> Result<()> {
        let mut kmeans = Kmeans::new(2);

        for i in 0..20 {
            let noise = (i as f64 - 10.0) * 2.0; // High variance: -20 to 18
            kmeans.add_datapoint(Vector::new(vec![noise, noise]));
        }

        for i in 0..20 {
            let noise = (i as f64 - 10.0) * 2.0;
            kmeans.add_datapoint(Vector::new(vec![100.0 + noise, 100.0 + noise]));
        }

        let initial_centroids = kmeans.centroids().len();

        kmeans.fit(100)?;

        let final_centroids = kmeans.centroids().len();

        assert!(
            final_centroids > initial_centroids,
            "Should add centroids for dispersed clusters (initial: {initial_centroids}, final: {final_centroids})"
        );

        Ok(())
    }

    #[test]
    fn test_centroid_addition_with_large_clusters() -> Result<()> {
        let mut kmeans = Kmeans::new(3);

        for i in 0..50 {
            let x = (i % 10) as f64 - 5.0;
            let y = ((i / 10) % 5) as f64 - 2.5;
            let z = (i / 50) as f64;
            kmeans.add_datapoint(Vector::new(vec![x, y, z]));
        }

        for i in 0..5 {
            let offset = i as f64 - 2.0;
            kmeans.add_datapoint(Vector::new(vec![
                50.0 + offset,
                50.0 + offset,
                50.0 + offset,
            ]));
        }

        let initial_centroids = kmeans.centroids().len();

        kmeans.fit(150)?;

        let final_centroids = kmeans.centroids().len();

        assert!(
            final_centroids > initial_centroids,
            "Should add centroids for unbalanced cluster sizes (initial: {initial_centroids}, final: {final_centroids})"
        );

        Ok(())
    }

    #[test]
    fn test_centroid_addition_with_natural_clusters() -> Result<()> {
        let mut kmeans = Kmeans::new(4);

        let cluster_centers = [
            vec![0.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];

        for (cluster_id, center) in cluster_centers.iter().enumerate() {
            for i in 0..10 {
                let noise_x = ((i * 7 + cluster_id * 3) % 20) as f64 / 10.0 - 1.0; // ±1.0 noise
                let noise_y = ((i * 11 + cluster_id * 5) % 20) as f64 / 10.0 - 1.0;
                let noise_z = ((i * 13 + cluster_id * 7) % 20) as f64 / 10.0 - 1.0;
                let noise_w = ((i * 17 + cluster_id * 11) % 20) as f64 / 10.0 - 1.0;

                kmeans.add_datapoint(Vector::new(vec![
                    center[0] + noise_x,
                    center[1] + noise_y,
                    center[2] + noise_z,
                    center[3] + noise_w,
                ]));
            }
        }

        let initial_centroids = kmeans.centroids().len();

        kmeans.fit(200)?;

        let final_centroids = kmeans.centroids().len();

        assert!(
            final_centroids > initial_centroids,
            "Should discover natural clusters (initial: {initial_centroids}, final: {final_centroids})"
        );

        assert!(
            final_centroids < 25,
            "Should not create excessive centroids: {final_centroids}"
        );

        Ok(())
    }

    #[test]
    fn test_no_centroid_addition_for_good_clusters() -> Result<()> {
        let mut kmeans = Kmeans::new(2);

        for i in 0..15 {
            let small_noise = (i as f64 - 7.5) * 0.2; // Very small variance
            kmeans.add_datapoint(Vector::new(vec![small_noise, small_noise]));
        }

        for i in 0..15 {
            let small_noise = (i as f64 - 7.5) * 0.2;
            kmeans.add_datapoint(Vector::new(vec![20.0 + small_noise, 20.0 + small_noise]));
        }

        let initial_centroids = kmeans.centroids().len();

        kmeans.fit(100)?;

        let final_centroids = kmeans.centroids().len();

        assert_eq!(
            final_centroids, initial_centroids,
            "Should not add centroids for well-clustered data (initial: {initial_centroids}, final: {final_centroids})"
        );

        Ok(())
    }

    #[test]
    fn test_centroid_addition_timing() -> Result<()> {
        let mut kmeans = Kmeans::new(2);
        let mut centroid_changes = Vec::new();

        for batch in 0..10 {
            for i in 0..10 {
                let spread = batch as f64 * 20.0 + (i as f64 - 5.0) * 5.0;
                kmeans.add_datapoint(Vector::new(vec![spread, spread * 0.5]));
            }

            let centroids_before = kmeans.centroids().len();
            kmeans.fit(25)?; // Fit after each batch
            let centroids_after = kmeans.centroids().len();

            if centroids_after > centroids_before {
                centroid_changes.push((batch, centroids_before, centroids_after));
            }
        }

        assert!(
            !centroid_changes.is_empty(),
            "Should add centroids as data becomes dispersed"
        );

        Ok(())
    }
}
