use crate::backends::{kmeans::KmeansDb, lsh_hash::LshDB, min_hash::MinHashDb};

pub mod kmeans;
pub mod lsh_hash;
pub mod min_hash;

pub enum Backends {
    Kmenas(KmeansDb),
    MinHash(MinHashDb),
    LSH(LshDB),
}
