use crate::{
    backends::{kmeans::KmeansDb, lsh_hash::LshDB, min_hash::MinHashDb, Backends},
    math::vector::Vector,
};
use pyo3::{prelude::*, types::PyType};

#[pyclass]
pub struct PyDatabase {
    database: Backends,
}

#[pymethods]
impl PyDatabase {
    #[classmethod]
    fn with_kmeans_backend(_cls: &PyType, vector_size: usize) -> PyResult<Self> {
        Ok(PyDatabase {
            database: Backends::Kmenas(KmeansDb::new(vector_size)),
        })
    }

    #[classmethod]
    fn with_min_hash_backends(
        _cls: &PyType,
        num_hashes: u64,
        num_bands: u64,
        similarity_threshold: f64,
    ) -> PyResult<Self> {
        Ok(PyDatabase {
            database: Backends::MinHash(MinHashDb::new(
                num_hashes,
                num_bands,
                similarity_threshold,
            )),
        })
    }

    #[classmethod]
    fn with_lsh_backends(
        _cls: &PyType,
        num_hashes: usize,
        num_bands: usize,
        vector_size: usize,
        similarity_threshold: f64,
    ) -> PyResult<Self> {
        Ok(PyDatabase {
            database: Backends::LSH(LshDB::new(
                num_hashes,
                num_bands,
                vector_size,
                similarity_threshold,
            )),
        })
    }

    fn insert(&mut self, vec: Vec<f64>) -> PyResult<()> {
        let results = match &mut self.database {
            Backends::Kmenas(backend) => backend.insert(Vector::new(vec)),
            Backends::MinHash(backend) => backend.insert(vec),
            Backends::LSH(lsh_db) => lsh_db.insert(vec),
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn query(&mut self, vec: Vec<f64>, n: usize) -> PyResult<Vec<Vec<f64>>> {
        let results = match &mut self.database {
            Backends::Kmenas(backend) => backend.query(Vector::new(vec), n),
            Backends::MinHash(backend) => backend.query(vec, n),
            Backends::LSH(lsh_db) => lsh_db.query(vec, n),
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
