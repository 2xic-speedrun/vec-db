use super::rocksdb::RocksDB;
use crate::math::vector::Vector;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
};

pub type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone)]
pub struct VectorWithMetadata {
    pub vector: Vector,
    pub metadata: Metadata,
}

pub struct ResultsWithMetadata {
    pub results: Vec<f64>,
    pub metadata: Metadata,
    pub similarity: f64,
}

impl From<ResultsWithMetadata> for (Vec<f64>, HashMap<String, String>, f64) {
    fn from(r: ResultsWithMetadata) -> Self {
        (r.results, r.metadata.into_iter().collect(), r.similarity)
    }
}

#[derive(Debug, Clone)]
pub enum PersistenceMode {
    Temporary,
    Persistent(String),
}

pub fn hash_vector(vector: &Vector) -> u64 {
    let mut hasher = DefaultHasher::new();
    vector.hash(&mut hasher);
    hasher.finish()
}

fn bytes_to_floats(data: &[u8]) -> Option<Vec<f64>> {
    (data.len() % 8 == 0).then(|| {
        data.chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    })
}

fn deserialize_metadata(bytes: Option<Vec<u8>>) -> Metadata {
    bytes
        .and_then(|b| bincode::deserialize(&b).ok())
        .unwrap_or_default()
}

pub trait BucketStorage: Send + Sync {
    fn new() -> Self;
    fn new_with_persistence(mode: PersistenceMode) -> Self;
    fn insert(&mut self, keys: &[&str], value: Vector) -> anyhow::Result<()>;
    fn insert_with_metadata(
        &mut self,
        keys: &[&str],
        value: Vector,
        metadata: Metadata,
    ) -> anyhow::Result<()>;
    fn get(&self, key: &str) -> Option<HashSet<Vector>>;
    fn get_with_metadata(&self, key: &str) -> Option<Vec<VectorWithMetadata>>;
    fn get_hashes(&self, key: &str) -> Vec<u64>;
    fn load_vectors(&self, hashes: &[u64]) -> Vec<(u64, Vector)>;
    fn load_metadata(&self, hashes: &[u64]) -> Vec<Metadata>;
    fn load_vectors_with_metadata(&self, hashes: &[u64]) -> Vec<VectorWithMetadata>;
    fn random_sample_excluding_buckets(
        &self,
        exclude_buckets: &HashSet<String>,
        n: usize,
    ) -> Vec<VectorWithMetadata>;
}

#[derive(Default)]
pub struct InMemoryBucket {
    storage: HashMap<String, HashSet<Vector>>,
    metadata_store: HashMap<u64, Metadata>,
}

impl BucketStorage for InMemoryBucket {
    fn new() -> Self {
        Self::default()
    }

    fn new_with_persistence(_mode: PersistenceMode) -> Self {
        Self::new()
    }

    fn insert(&mut self, keys: &[&str], value: Vector) -> anyhow::Result<()> {
        for key in keys {
            self.storage
                .entry(key.to_string())
                .or_default()
                .insert(value.clone());
        }
        Ok(())
    }

    fn insert_with_metadata(
        &mut self,
        keys: &[&str],
        value: Vector,
        metadata: Metadata,
    ) -> anyhow::Result<()> {
        self.metadata_store.insert(hash_vector(&value), metadata);
        self.insert(keys, value)
    }

    fn get(&self, key: &str) -> Option<HashSet<Vector>> {
        self.storage.get(key).cloned()
    }

    fn get_with_metadata(&self, key: &str) -> Option<Vec<VectorWithMetadata>> {
        self.storage.get(key).map(|vectors| {
            vectors
                .iter()
                .map(|v| VectorWithMetadata {
                    metadata: self
                        .metadata_store
                        .get(&hash_vector(v))
                        .cloned()
                        .unwrap_or_default(),
                    vector: v.clone(),
                })
                .collect()
        })
    }

    fn get_hashes(&self, key: &str) -> Vec<u64> {
        self.storage
            .get(key)
            .map(|vectors| vectors.iter().map(hash_vector).collect())
            .unwrap_or_default()
    }

    fn load_vectors(&self, hashes: &[u64]) -> Vec<(u64, Vector)> {
        hashes
            .iter()
            .filter_map(|&h| {
                self.storage
                    .values()
                    .flatten()
                    .find(|v| hash_vector(v) == h)
                    .map(|v| (h, v.clone()))
            })
            .collect()
    }

    fn load_metadata(&self, hashes: &[u64]) -> Vec<Metadata> {
        hashes
            .iter()
            .map(|h| self.metadata_store.get(h).cloned().unwrap_or_default())
            .collect()
    }

    fn load_vectors_with_metadata(&self, hashes: &[u64]) -> Vec<VectorWithMetadata> {
        hashes
            .iter()
            .filter_map(|&h| {
                self.storage
                    .values()
                    .flatten()
                    .find(|v| hash_vector(v) == h)
                    .map(|v| VectorWithMetadata {
                        vector: v.clone(),
                        metadata: self.metadata_store.get(&h).cloned().unwrap_or_default(),
                    })
            })
            .collect()
    }

    fn random_sample_excluding_buckets(
        &self,
        exclude_buckets: &HashSet<String>,
        n: usize,
    ) -> Vec<VectorWithMetadata> {
        use rand::prelude::*;

        let mut seen = HashSet::new();
        let mut results: Vec<_> = self
            .storage
            .iter()
            .filter(|(key, _)| !exclude_buckets.contains(*key))
            .flat_map(|(_, vectors)| vectors)
            .filter_map(|v| {
                let h = hash_vector(v);
                seen.insert(h).then(|| VectorWithMetadata {
                    vector: v.clone(),
                    metadata: self.metadata_store.get(&h).cloned().unwrap_or_default(),
                })
            })
            .collect();

        results.shuffle(&mut rand::rng());
        results.truncate(n);
        results
    }
}

pub struct RocksDbBucket {
    storage: RocksDB,
    db_path: String,
    persistence_mode: PersistenceMode,
}

impl RocksDbBucket {
    fn collect_hashes_from_prefix(&self, prefix: &str) -> Vec<u64> {
        self.storage
            .prefix_iterator(prefix)
            .filter_map(|(_, bytes)| bytes.as_ref().try_into().ok().map(u64::from_be_bytes))
            .collect()
    }

    fn parse_vectors(&self, keys: &[String]) -> impl Iterator<Item = Vector> + '_ {
        self.storage
            .multi_get(keys)
            .into_iter()
            .filter_map(|data| bytes_to_floats(&data?).map(Vector::new))
    }

    pub fn compact(&self) {
        self.storage.compact();
    }

    pub fn random_sample_with_metadata(&self, n: usize) -> Vec<VectorWithMetadata> {
        let hashes: Vec<u64> = self
            .storage
            .sample_keys("v:", n)
            .iter()
            .filter_map(|k| k.strip_prefix("v:")?.parse().ok())
            .collect();
        self.load_vectors_with_metadata(&hashes)
    }
}

impl BucketStorage for RocksDbBucket {
    fn new() -> Self {
        Self::new_with_persistence(PersistenceMode::Temporary)
    }

    fn new_with_persistence(mode: PersistenceMode) -> Self {
        let (db_path, persistence_mode) = match mode {
            PersistenceMode::Temporary => (
                format!("/tmp/bucket_db_{}", std::process::id()),
                PersistenceMode::Temporary,
            ),
            PersistenceMode::Persistent(ref path) => {
                (path.clone(), PersistenceMode::Persistent(path.clone()))
            }
        };

        Self {
            storage: RocksDB::new(&db_path).expect("Failed to create RocksDB instance"),
            db_path,
            persistence_mode,
        }
    }

    fn insert(&mut self, keys: &[&str], value: Vector) -> anyhow::Result<()> {
        let h = hash_vector(&value);
        let vector_key = format!("v:{h}");
        let vector_bytes: Vec<u8> = value
            .as_ref()
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        let hash_bytes = h.to_be_bytes();
        let bucket_keys: Vec<String> = keys.iter().map(|k| format!("b:{k}:{h}")).collect();

        self.storage.write_batch(|batch| {
            batch.put(&vector_key, &vector_bytes);
            for bucket_key in &bucket_keys {
                batch.put(bucket_key, hash_bytes);
            }
        })
    }

    fn insert_with_metadata(
        &mut self,
        keys: &[&str],
        value: Vector,
        metadata: Metadata,
    ) -> anyhow::Result<()> {
        let h = hash_vector(&value);
        let vector_key = format!("v:{h}");
        let metadata_key = format!("m:{h}");
        let vector_bytes: Vec<u8> = value
            .as_ref()
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        let metadata_bytes = bincode::serialize(&metadata).expect("Failed to serialize metadata");
        let hash_bytes = h.to_be_bytes();
        let bucket_keys: Vec<String> = keys.iter().map(|k| format!("b:{k}:{h}")).collect();

        self.storage.write_batch(|batch| {
            batch.put(&vector_key, &vector_bytes);
            batch.put(&metadata_key, &metadata_bytes);
            for bucket_key in &bucket_keys {
                batch.put(bucket_key, hash_bytes);
            }
        })
    }

    fn get(&self, key: &str) -> Option<HashSet<Vector>> {
        let hashes = self.collect_hashes_from_prefix(&format!("b:{key}:"));
        if hashes.is_empty() {
            return None;
        }

        let keys: Vec<_> = hashes.iter().map(|h| format!("v:{h}")).collect();
        let vectors: HashSet<_> = self.parse_vectors(&keys).collect();
        (!vectors.is_empty()).then_some(vectors)
    }

    fn get_with_metadata(&self, key: &str) -> Option<Vec<VectorWithMetadata>> {
        let hashes = self.collect_hashes_from_prefix(&format!("b:{key}:"));
        if hashes.is_empty() {
            return None;
        }

        let vector_keys: Vec<_> = hashes.iter().map(|h| format!("v:{h}")).collect();
        let metadata_keys: Vec<_> = hashes.iter().map(|h| format!("m:{h}")).collect();

        let results: Vec<_> = self
            .storage
            .multi_get(&vector_keys)
            .into_iter()
            .zip(self.storage.multi_get(&metadata_keys))
            .filter_map(|(vec_data, meta_data)| {
                Some(VectorWithMetadata {
                    vector: Vector::new(bytes_to_floats(&vec_data?)?),
                    metadata: deserialize_metadata(meta_data),
                })
            })
            .collect();

        (!results.is_empty()).then_some(results)
    }

    fn get_hashes(&self, key: &str) -> Vec<u64> {
        self.collect_hashes_from_prefix(&format!("b:{key}:"))
    }

    fn load_vectors(&self, hashes: &[u64]) -> Vec<(u64, Vector)> {
        if hashes.is_empty() {
            return Vec::new();
        }

        const BATCH_SIZE: usize = 2000;
        hashes
            .par_chunks(BATCH_SIZE)
            .flat_map(|batch| {
                let keys: Vec<_> = batch.iter().map(|h| format!("v:{h}")).collect();
                batch
                    .iter()
                    .zip(self.storage.multi_get(&keys))
                    .filter_map(|(&h, data)| Some((h, Vector::new(bytes_to_floats(&data?)?))))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn load_metadata(&self, hashes: &[u64]) -> Vec<Metadata> {
        if hashes.is_empty() {
            return Vec::new();
        }

        let keys: Vec<_> = hashes.iter().map(|h| format!("m:{h}")).collect();
        self.storage
            .multi_get(&keys)
            .into_iter()
            .map(deserialize_metadata)
            .collect()
    }

    fn load_vectors_with_metadata(&self, hashes: &[u64]) -> Vec<VectorWithMetadata> {
        if hashes.is_empty() {
            return Vec::new();
        }

        const BATCH_SIZE: usize = 1000;
        hashes
            .par_chunks(BATCH_SIZE)
            .flat_map(|batch| {
                let vector_keys: Vec<_> = batch.iter().map(|h| format!("v:{h}")).collect();
                let metadata_keys: Vec<_> = batch.iter().map(|h| format!("m:{h}")).collect();

                self.storage
                    .multi_get(&vector_keys)
                    .into_iter()
                    .zip(self.storage.multi_get(&metadata_keys))
                    .filter_map(|(vec_data, meta_data)| {
                        Some(VectorWithMetadata {
                            vector: Vector::new(bytes_to_floats(&vec_data?)?),
                            metadata: deserialize_metadata(meta_data),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn random_sample_excluding_buckets(
        &self,
        exclude_buckets: &HashSet<String>,
        n: usize,
    ) -> Vec<VectorWithMetadata> {
        let mut hashes: Vec<u64> = self
            .storage
            .sample_keys("b:", n * 5)
            .iter()
            .filter_map(|key| {
                let rest = key.strip_prefix("b:")?;
                let last_colon = rest.rfind(':')?;
                let bucket_key = &rest[..last_colon];
                (!exclude_buckets.contains(bucket_key))
                    .then(|| rest[last_colon + 1..].parse().ok())?
            })
            .collect();

        hashes.sort_unstable();
        hashes.dedup();
        hashes.truncate(n);

        self.load_vectors_with_metadata(&hashes)
    }
}

impl Drop for RocksDbBucket {
    fn drop(&mut self) {
        if let PersistenceMode::Temporary = self.persistence_mode {
            if let Err(e) = std::fs::remove_dir_all(&self.db_path) {
                eprintln!(
                    "Warning: Failed to clean up temporary RocksDB directory {}: {}",
                    self.db_path, e
                );
            }
        }
    }
}
