use crate::db::simple_database::SimpleDatabase;
use crate::math::vector::Vector;
use pyo3::prelude::*;

#[pyclass]
pub struct PyDatabase {
    database: SimpleDatabase,
}

#[pymethods]
impl PyDatabase {
    #[new]
    fn new(vector_size: usize) -> PyDatabase {
        PyDatabase {
            database: SimpleDatabase::new(vector_size),
        }
    }

    fn insert(&mut self, vec: Vec<f64>) {
        self.database.insert(Vector::new(vec));
    }

    fn centroids(&mut self) -> Vec<Vec<f64>> {
        self.database.centrodis()
    }

    fn query(&mut self, vec: Vec<f64>, n: usize) -> Vec<Vec<f64>> {
        self.database.query(Vector::new(vec), n)
    }
}
