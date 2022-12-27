use crate::vector::vector::Vector;
use core::f64::INFINITY;
use std::collections::HashMap;

pub struct Kmeans {
    centroids: Vec<Vector>,
    dataset: Vec<Vector>,
    centroids_count: usize,
    length: usize
}

impl Kmeans  {
    pub fn new(centroids_count: usize, length: usize) -> Kmeans {
        return Kmeans{
            dataset: Vec::new(),
            centroids: Vec::new(),
            centroids_count: centroids_count,
            length: length,
        }
    }

    // TODO: Replace this method with some rand method available on vector
    pub fn add_centroid(&mut self, vector: Vector){
        if self.centroids.len() < self.centroids_count {
            self.centroids.push(vector);
        }
    }

    pub fn add(&mut self, vector: Vector) {
        self.dataset.push(vector);
    }

    pub fn centroids(&self) -> &Vec<Vector> {
        return &self.centroids;
    }

    pub fn fit(&mut self, iterations: i64) {
        for _ in 0..iterations {
            self.forward();
        }
    }

    fn forward(&mut self) {
        // TODO: How does one efficiently set a centroid location ?
        //      Currently we make the user do it

        let mut clustered_data_pints:HashMap<usize, Vec<&Vector>> = HashMap::new();

        for data_point in self.dataset.iter() {
         //   let mut best_centroid: Option<&Vector> = None;
            let mut best_score = INFINITY;
            let mut best_index: usize = 0;
            for (index, centroid) in self.centroids.iter().enumerate() {
                let distance = centroid.l2_distance(data_point).unwrap();

                if distance < best_score {
                    best_score = distance;
                    best_index = index;
                }
            }

            match clustered_data_pints.get_mut(&best_index) {
                Some(value) => {
                    value.push(data_point);
                },
                None => {
                    let mut vec = Vec::new();
                    vec.push(data_point);
                    clustered_data_pints.insert(
                        best_index,
                        vec
                    );
                }
            };
        }

        for (key, vectors) in clustered_data_pints.into_iter() {
            let clustered_len = vectors.len();
            // TODO: Should be possible to initialize a zero vector based on another vec
            let mut zero_vec = Vec::new();
            for _ in 0..self.length {
                zero_vec.push(0.0);
            }
            
            if 0 < clustered_len {
                let mut delta_vector: Vector = Vector::new(zero_vec);
                for vector in vectors {
                    delta_vector = delta_vector.add(vector).unwrap();
                }
    
                delta_vector = delta_vector.mul_constant((1.0/(clustered_len as f64)));
        
                self.centroids[key] = delta_vector;
            }
        }
    }
}
