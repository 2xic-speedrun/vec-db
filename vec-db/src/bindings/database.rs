use crate::{
    backends::{kmeans::KmeansDb, BackendEnum},
    math::vector::Vector,
};
use pyo3::{prelude::*, types::PyType};

#[pyclass]
pub struct PyDatabase {
    database: BackendEnum,
}

#[pymethods]
impl PyDatabase {
    #[classmethod]
    fn with_kmeans_backend(_cls: &PyType, vector_size: usize) -> PyResult<Self> {
        Ok(PyDatabase {
            database: BackendEnum::Kmenas(KmeansDb::new(vector_size)),
        })
    }

    fn insert(&mut self, vec: Vec<f64>) -> PyResult<()> {
        let results = match &mut self.database {
            BackendEnum::Kmenas(backend) => backend.insert(Vector::new(vec)),
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn query(&mut self, vec: Vec<f64>, n: usize) -> PyResult<Vec<Vec<f64>>> {
        let results = match &mut self.database {
            BackendEnum::Kmenas(backend) => backend.query(Vector::new(vec), n),
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
