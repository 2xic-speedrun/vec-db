use pyo3::prelude::*;
use crate::vector::vector::Vector;
use crate::fileformat::simple_database::SimpleDatabase;


#[pyclass]
pub struct PyDatabase {
    database: SimpleDatabase
}

#[pymethods]
impl PyDatabase {
    #[new]
    fn new() -> PyDatabase {
        return PyDatabase {
            database: SimpleDatabase::new()
        }
    }

    fn insert(&mut self, vec: Vec<f64>) {
        self.database.insert(Vector::new(vec));
    }

    fn query(&mut self, vec: Vec<f64>) -> Vec<Vec<f64>> {
        self.database.query(Vector::new(vec))
    }

    fn load(&mut self) {
        self.database.load();
    }

    fn dump(&mut self){
        self.database.dump();
    }
}
