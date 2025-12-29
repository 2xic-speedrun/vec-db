use crate::{
    backends::{hnsw::HnswDB, kmeans::KmeansDb, lsh_hash::LshDB, min_hash::MinHashDb, Backends},
    math::vector::Vector,
    storage::bucket::{Metadata, RocksDbBucket},
};
use pyo3::{prelude::*, types::PyType};
use std::collections::HashMap;

type QueryResultWithMetadata = (Vec<f64>, HashMap<String, String>, f64);

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
    fn with_persistent_min_hash_rocksdb(
        _cls: &PyType,
        num_hashes: u64,
        num_bands: u64,
        similarity_threshold: f64,
        db_path: String,
    ) -> PyResult<Self> {
        Ok(PyDatabase {
            database: Backends::MinHashRocksDB(MinHashDb::persistent(
                num_hashes,
                num_bands,
                similarity_threshold,
                db_path,
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
        let lsh_db = LshDB::new(num_hashes, num_bands, vector_size, similarity_threshold)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyDatabase {
            database: Backends::LSH(lsh_db),
        })
    }

    #[classmethod]
    fn with_lsh_rocksdb_backends(
        _cls: &PyType,
        num_hashes: usize,
        num_bands: usize,
        vector_size: usize,
        similarity_threshold: f64,
    ) -> PyResult<Self> {
        let lsh_db = LshDB::new(num_hashes, num_bands, vector_size, similarity_threshold)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyDatabase {
            database: Backends::LSHRocksDB(lsh_db),
        })
    }

    #[classmethod]
    fn with_persistent_lsh_rocksdb(
        _cls: &PyType,
        num_hashes: usize,
        num_bands: usize,
        vector_size: usize,
        similarity_threshold: f64,
        db_path: String,
    ) -> PyResult<Self> {
        let lsh_db = LshDB::<RocksDbBucket>::persistent(
            num_hashes,
            num_bands,
            vector_size,
            similarity_threshold,
            db_path,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyDatabase {
            database: Backends::LSHRocksDB(lsh_db),
        })
    }

    #[classmethod]
    fn with_hnsw(_cls: &PyType, similarity_threshold: f64) -> PyResult<Self> {
        Ok(PyDatabase {
            database: Backends::HNSW(HnswDB::new(256, similarity_threshold)),
        })
    }

    fn insert(&mut self, vec: Vec<f64>) -> PyResult<()> {
        let results = match &mut self.database {
            Backends::Kmenas(backend) => backend.insert(Vector::new(vec)),
            Backends::MinHash(backend) => backend.insert(vec),
            Backends::MinHashRocksDB(backend) => backend.insert(vec),
            Backends::LSH(lsh_db) => lsh_db.insert(vec),
            Backends::HNSW(hnsw_db) => hnsw_db.insert(Vector::new(vec)),
            Backends::LSHRocksDB(lsh_db) => lsh_db.insert(vec),
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn insert_with_metadata(
        &mut self,
        vec: Vec<f64>,
        metadata: HashMap<String, String>,
    ) -> PyResult<()> {
        let metadata: Metadata = metadata.into_iter().collect();
        let results = match &mut self.database {
            Backends::LSH(lsh_db) => lsh_db.insert_with_metadata(vec, metadata),
            Backends::LSHRocksDB(lsh_db) => lsh_db.insert_with_metadata(vec, metadata),
            Backends::MinHash(db) => db.insert_with_metadata(vec, metadata),
            Backends::MinHashRocksDB(db) => db.insert_with_metadata(vec, metadata),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "insert_with_metadata only supported for LSH/MinHash backends",
                ))
            }
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn query(&mut self, vec: Vec<f64>, n: usize) -> PyResult<Vec<Vec<f64>>> {
        let results = match &mut self.database {
            Backends::Kmenas(backend) => backend.query(Vector::new(vec), n),
            Backends::MinHash(backend) => backend.query(vec, n),
            Backends::MinHashRocksDB(backend) => backend.query(vec, n),
            Backends::LSH(lsh_db) => lsh_db.query(vec, n),
            Backends::HNSW(hnsw_db) => hnsw_db.query(Vector::new(vec), n),
            Backends::LSHRocksDB(lsh_db) => lsh_db.query(vec, n),
        };
        results.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn query_with_metadata(
        &self,
        vec: Vec<f64>,
        n: usize,
    ) -> PyResult<Vec<QueryResultWithMetadata>> {
        let results = match &self.database {
            Backends::LSH(lsh_db) => lsh_db.query_with_metadata(vec, n),
            Backends::LSHRocksDB(lsh_db) => lsh_db.query_with_metadata(vec, n),
            Backends::MinHash(db) => db.query_with_metadata(vec, n),
            Backends::MinHashRocksDB(db) => db.query_with_metadata(vec, n),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "query_with_metadata only supported for LSH/MinHash backends",
                ))
            }
        };
        results
            .map(|r| r.into_iter().map(Into::into).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn compact(&self) -> PyResult<()> {
        match &self.database {
            Backends::LSHRocksDB(lsh_db) => {
                lsh_db.compact();
                Ok(())
            }
            Backends::MinHashRocksDB(db) => {
                db.compact();
                Ok(())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "compact only supported for RocksDB backends",
            )),
        }
    }

    fn query_dissimilar_with_metadata(
        &self,
        vec: Vec<f64>,
        n: usize,
    ) -> PyResult<Vec<QueryResultWithMetadata>> {
        let results = match &self.database {
            Backends::LSH(lsh_db) => lsh_db.query_dissimilar_with_metadata(vec, n),
            Backends::LSHRocksDB(lsh_db) => lsh_db.query_dissimilar_with_metadata(vec, n),
            Backends::MinHash(mh_db) => mh_db.query_dissimilar_with_metadata(vec, n),
            Backends::MinHashRocksDB(mh_db) => mh_db.query_dissimilar_with_metadata(vec, n),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "query_dissimilar_with_metadata not supported for this backend",
                ))
            }
        };
        results
            .map(|r| r.into_iter().map(Into::into).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
