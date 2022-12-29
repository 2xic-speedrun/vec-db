use pyo3::prelude::*;
use crate::vector::vector::Vector;
use crate::kmeans::kmeans::Kmeans;
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

    fn save(&mut self, vec: Vec<f64>) {
        self.database.save(Vector::new(vec));
    }

    fn load(&mut self) {
        self.database.load();
    }

    fn dump(&mut self){
        self.database.dump();
    }
}
