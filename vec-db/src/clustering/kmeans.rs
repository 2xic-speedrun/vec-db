use crate::math::vector::Vector;
use std::{cmp::min, collections::HashMap};

pub struct Kmeans {
    centroids: Vec<Vector>,
    inertia_distance_centroids: Vec<f64>,
    dataset: Vec<Vector>,
    vector_length: usize,
    fit_index: i64,
}

impl Kmeans {
    pub fn new(vector_length: usize) -> Kmeans {
        Kmeans {
            dataset: Vec::new(),
            centroids: Vec::new(),
            inertia_distance_centroids: Vec::new(),
            vector_length,
            fit_index: 0,
        }
    }

    pub fn add_centroid(&mut self, vector: Vector) {
        self.centroids.push(vector);
    }

    pub fn add_datapoint(&mut self, vector: Vector) {
        if self.centroids.is_empty() {
            self.centroids.push(Vector::rand(self.vector_length));
            self.centroids.push(Vector::rand(self.vector_length));
        }
        self.dataset.push(vector);
    }

    pub fn centroids(&self) -> &Vec<Vector> {
        &self.centroids
    }

    pub fn dataset(&self) -> &Vec<Vector> {
        &self.dataset
    }

    pub fn fit(&mut self, iterations: i64) {
        let mut previous_run = self.inertia_distance_centroids.clone();
        let mut previous_centroids = self.centroids.clone();
        const CONVERGENCE_THRESHOLD: f64 = 1e-6;

        for _ in 0..iterations {
            self.forward();

            // Check for convergence
            let mut total_movement = 0.0;
            for (new_centroid, old_centroid) in self.centroids.iter().zip(&previous_centroids) {
                total_movement += new_centroid.l2_distance(old_centroid).unwrap_or(0.0);
            }

            if total_movement < CONVERGENCE_THRESHOLD {
                break;
            }

            previous_centroids = self.centroids.clone();

            // Check if more centroids need to be added
            if self.fit_index > 0 && self.fit_index % (75 * (self.centroids().len() as i64)) == 0 {
                let previous_average_inertia = Vector::new(previous_run.clone()).abs().sum_d1()
                    / (self.centroids().len() as f64);
                let new_average_inertia = Vector::new(self.inertia_distance_centroids.clone())
                    .abs()
                    .sum_d1()
                    / (self.centroids().len() as f64);

                let delta =
                    (new_average_inertia - previous_average_inertia) / previous_average_inertia;
                if delta > 0.2 {
                    self.add_centroid(Vector::rand(self.vector_length));
                    previous_run = self.inertia_distance_centroids.clone();
                }
            }
            self.fit_index += 1
        }
    }

    pub fn results(&self) -> Vec<(&Vector, i8)> {
        let mut results: Vec<(&Vector, i8)> = Vec::new();
        for i in self.dataset.iter() {
            results.push((i, (self.find_closest_centroid(i)) as i8));
        }
        results
    }

    fn find_closest_centroid(&self, data_point: &Vector) -> usize {
        let mut best_score = f64::INFINITY;
        let mut best_index: usize = 0;
        for (index, centroid) in self.centroids.iter().enumerate() {
            let distance = centroid.l2_distance_squared(data_point).unwrap();

            if distance < best_score {
                best_score = distance;
                best_index = index;
            }
        }

        best_index
    }

    pub fn find_closest_data_points(&self, data_point: &Vector, n: usize) -> Vec<Vec<f64>> {
        let mut results: Vec<Vec<f64>> = Vec::new();

        let clustered_data_pints = self.get_centroids_data_point();
        let centroid = self.find_closest_centroid(data_point);
        let items = clustered_data_pints.get(&centroid);
        if let Some(items) = items {
            let max_index = min(n, items.len());

            for vector in items.iter().take(max_index) {
                if !vector.equal(data_point) {
                    results.push(vector.raw().clone());
                }
            }
        }

        results
    }

    fn forward(&mut self) {
        let mut assignments = Vec::with_capacity(self.dataset.len());

        for data_point in &self.dataset {
            assignments.push(self.find_closest_centroid(data_point));
        }

        let mut new_centroids = vec![Vector::empty(self.vector_length); self.centroids.len()];
        let mut cluster_counts = vec![0; self.centroids.len()];
        let mut new_inertia_distance = vec![0.0; self.centroids.len()];

        for (data_point, &cluster_id) in self.dataset.iter().zip(&assignments) {
            new_centroids[cluster_id].add_inplace(data_point).unwrap();
            cluster_counts[cluster_id] += 1;
        }

        for (i, &count) in cluster_counts.iter().enumerate() {
            if count > 0 {
                new_centroids[i] = new_centroids[i].div_constant(count as f64);
                new_inertia_distance[i] = new_centroids[i].sum_d1();
            }
        }

        self.centroids = new_centroids;
        self.inertia_distance_centroids = new_inertia_distance;
    }

    fn get_centroids_data_point(&self) -> HashMap<usize, Vec<&Vector>> {
        let mut clustered_data_pints: HashMap<usize, Vec<&Vector>> = HashMap::new();

        for data_point in self.dataset.iter() {
            let best_index = self.find_closest_centroid(data_point);

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

        clustered_data_pints
    }
}
