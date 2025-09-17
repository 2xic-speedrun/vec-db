use crate::backends::kmeans::KmeansDb;

pub mod kmeans;

pub trait Backend: Send {
    fn query(self, vec: Vec<f64>, n: usize) -> anyhow::Result<()>;

    fn insert(self, vec: Vec<f64>) -> Vec<Vec<f64>>;
}

pub enum BackendEnum {
    Kmenas(KmeansDb),
}
