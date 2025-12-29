use crate::{
    backends::{hnsw::HnswDB, kmeans::KmeansDb, lsh_hash::LshDB, min_hash::MinHashDb},
    storage::bucket::{InMemoryBucket, RocksDbBucket},
};

pub mod hnsw;
pub mod kmeans;
pub mod lsh_hash;
pub mod min_hash;

pub enum Backends {
    Kmenas(KmeansDb),
    MinHash(MinHashDb<InMemoryBucket>),
    MinHashRocksDB(MinHashDb<RocksDbBucket>),
    LSH(LshDB<InMemoryBucket>),
    LSHRocksDB(LshDB<RocksDbBucket>),
    HNSW(HnswDB),
}
