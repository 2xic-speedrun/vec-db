use crate::clustering::kmeans::Kmeans;
use crate::math::vector::Vector;
use pyo3::prelude::*;

#[pyclass]
pub struct PyVector {
    pub vector: Vector,
}

#[pymethods]
impl PyVector {
    #[new]
    fn new(value: Vec<f64>) -> PyVector {
        PyVector {
            vector: Vector::new(value),
        }
    }
}

#[pyclass]
pub struct PyKmeans {
    kmeans: Kmeans,
}

#[pymethods]
impl PyKmeans {
    #[new]
    fn new(shape: usize) -> PyKmeans {
        PyKmeans {
            kmeans: Kmeans::new(shape),
        }
    }

    fn add_centroid(&mut self, vec: Vec<f64>) {
        self.kmeans.add_centroid(Vector::new(vec));
    }

    fn get_centroid(&mut self, loc: usize) -> Vec<f64> {
        let vec = self.kmeans.centroids().get(loc).unwrap().raw();
        let mut new_vec: Vec<f64> = Vec::new();
        for i in 0..vec.len() {
            new_vec.push(*vec.get(i).unwrap());
        }
        new_vec
    }

    fn add_datapoint(&mut self, vec: Vec<f64>) {
        self.kmeans.add_datapoint(Vector::new(vec));
    }

    fn fit(&mut self, iterations: i64) {
        self.kmeans.fit(iterations);
    }
}
