use crate::backends::{
    hnsw::HnswDB,
    kmeans::KmeansDb,
    lsh_hash::{InMemoryBucket, LshDB, RocksDbBucket},
    min_hash::MinHashDb,
};

pub mod hnsw;
pub mod kmeans;
pub mod lsh_hash;
pub mod min_hash;

pub enum Backends {
    Kmenas(KmeansDb),
    MinHash(MinHashDb),
    LSH(LshDB<InMemoryBucket>),
    LSHRocksDB(LshDB<RocksDbBucket>),
    HNSW(HnswDB),
}
