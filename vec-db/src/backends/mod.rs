use crate::backends::{kmeans::KmeansDb, min_hash::MinHashDb};

pub mod kmeans;
pub mod min_hash;

pub enum Backends {
    Kmenas(KmeansDb),
    MinHash(MinHashDb),
}
