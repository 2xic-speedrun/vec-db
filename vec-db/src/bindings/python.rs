use crate::backends::kmeans::Kmeans;
use crate::math::vector::Vector;
use anyhow::Result;
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
            kmeans: Kmeans::new(shape, None),
        }
    }

    fn add_centroid(&mut self, vec: Vec<f64>) {
        self.kmeans.add_centroid(Vector::new(vec));
    }

    fn get_centroid(&mut self, loc: usize) -> PyResult<Vec<f64>> {
        self.get_centroid_impl(loc)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn add_datapoint(&mut self, vec: Vec<f64>) {
        self.kmeans.add_datapoint(Vector::new(vec));
    }

    fn fit(&mut self, iterations: u64) -> PyResult<()> {
        self.kmeans
            .fit(iterations)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

impl PyKmeans {
    fn get_centroid_impl(&self, loc: usize) -> Result<Vec<f64>> {
        let centroids = self.kmeans.centroids();
        let vec = centroids
            .get(loc)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Centroid index {} out of bounds (have {})",
                    loc,
                    centroids.len()
                )
            })?
            .as_ref();

        Ok(vec.to_vec())
    }
}
