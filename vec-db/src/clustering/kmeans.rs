use crate::math::vector::Vector;
use rand::Rng;
use std::collections::HashMap;

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

    // TODO: Replace this method with some rand method available on vector
    // TODO: Need this to change depending on how many items are in the database
    pub fn add_centroid(&mut self, vector: Vector) {
        self.centroids.push(vector);
    }

    pub fn add_datapoint(&mut self, vector: Vector) {
        if self.centroids.is_empty() {
            self.centroids.push(Vector::new(self.get_random_vec()));
            self.centroids.push(Vector::new(self.get_random_vec()));
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

        for _ in 0..iterations {
            self.forward();

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
                    self.add_centroid(Vector::new(self.get_random_vec()));
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
            let distance = centroid.l2_distance(data_point).unwrap();

            if distance < best_score {
                best_score = distance;
                best_index = index;
            }
        }

        best_index
    }

    pub fn find_closest_data_points(&self, data_point: &Vector, n: usize) -> Vec<Vec<f64>> {
        let mut results: Vec<Vec<f64>> = Vec::new();

        let clustered_data_pints = self.get_centroids_data_point().clone();
        let centroid = self.find_closest_centroid(data_point);
        let items = &clustered_data_pints.clone()[&centroid.clone()];

        for index in 0..n {
            if index < items.len() {
                let vector = items[index];
                if !vector.equal(data_point.clone()) {
                    results.push(vector.raw().clone());
                }
            }
        }

        results
    }

    fn forward(&mut self) {
        // TODO: How does one efficiently set a centroid location ?
        //      Currently we make the user do it
        let clustered_data_pints = self.get_centroids_data_point().clone();
        let mut new_centorids = Vec::with_capacity(self.centroids.len() + 1);
        let mut new_inertia_distance: Vec<f64> = Vec::with_capacity(self.centroids.len() + 1);
        for _ in 0..self.centroids.len() {
            new_centorids.push(Vector::new(self.get_zero_vec().clone()));
            new_inertia_distance.push(0.0);
        }

        for (key, vectors) in clustered_data_pints.clone().into_iter() {
            let clustered_size = vectors.len();
            // TODO: Should be possible to initialize a zero vector based on another vec
            let zero_vec = self.get_zero_vec().clone();
            let mut delta_vector: Vector = Vector::new(zero_vec);
            for vector in vectors {
                delta_vector = delta_vector.add(vector).unwrap();
            }
            delta_vector = delta_vector.div_constant(clustered_size as f64).clone();

            new_centorids[key] = delta_vector.clone();
            new_inertia_distance[key] += delta_vector.clone().sum_d1();
        }

        self.centroids = new_centorids;
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

    pub fn get_zero_vec(&self) -> Vec<f64> {
        let zero_vec = vec![0.0;self.vector_length];
        zero_vec
    }

    pub fn get_random_vec(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut zero_vec = Vec::new();
        for _ in 0..self.vector_length {
            zero_vec.push(rng.gen());
        }
        zero_vec
    }
}
