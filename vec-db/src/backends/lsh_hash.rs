use crate::{math::vector::Vector, storage::rocksdb::RocksDB};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
};

pub type Metadata = BTreeMap<String, String>;

pub struct VectorLSH {
    num_hashes: usize,
    num_bands: usize,
    rows_per_band: usize,
    dimension: usize,
    random_vectors: Vec<Vector>,
}

impl VectorLSH {
    pub fn new(num_hashes: usize, num_bands: usize, dimension: usize) -> anyhow::Result<Self> {
        if num_hashes % num_bands != 0 {
            return Err(anyhow::anyhow!(
                "num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands})"
            ));
        }
        let rows_per_band = num_hashes / num_bands;
        let rng_seed = 12345u64;
        let rng = Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(rng_seed)));

        let mut random_vectors = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            let vec = Vector::rand(dimension, &rng).norm();
            random_vectors.push(vec);
        }

        Ok(VectorLSH {
            num_hashes,
            num_bands,
            rows_per_band,
            dimension,
            random_vectors,
        })
    }

    fn hash_vector(&self, vector: &[f64]) -> anyhow::Result<Vec<u32>> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let mut signature = Vec::with_capacity(self.num_hashes);

        let input_vector = Vector::new(vector.to_vec());
        for random_vec in &self.random_vectors {
            let dot_product = random_vec.l1_dot(&input_vector)?;
            signature.push(if dot_product >= 0.0 { 1 } else { 0 });
        }

        Ok(signature)
    }

    fn get_bucket_keys(&self, vector: &[f64]) -> anyhow::Result<Vec<String>> {
        let signature = self.hash_vector(vector)?;
        let mut bucket_keys = Vec::with_capacity(self.num_bands);

        for band in 0..self.num_bands {
            let start_idx = band * self.rows_per_band;
            let end_idx = start_idx + self.rows_per_band;
            let band_signature = &signature[start_idx..end_idx];

            let mut key = String::with_capacity(self.rows_per_band + 10);
            key.push_str(&format!("b{band}_"));
            for &bit in band_signature {
                key.push(if bit == 1 { '1' } else { '0' });
            }
            bucket_keys.push(key);
        }

        Ok(bucket_keys)
    }
}

#[derive(Debug, Clone)]
pub struct VectorWithMetadata {
    pub vector: Vector,
    pub metadata: Metadata,
}

pub trait Storage {
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

fn hash_vector(vector: &Vector) -> u64 {
    let mut hasher = DefaultHasher::new();
    vector.hash(&mut hasher);
    hasher.finish()
}

#[derive(Default)]
pub struct InMemoryBucket {
    storage: HashMap<String, HashSet<Vector>>,
    metadata_store: HashMap<u64, Metadata>,
}

impl InMemoryBucket {
    pub fn new() -> Self {
        InMemoryBucket {
            storage: HashMap::new(),
            metadata_store: HashMap::new(),
        }
    }
}

impl Storage for InMemoryBucket {
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
        let vector_hash = hash_vector(&value);
        self.metadata_store.insert(vector_hash, metadata);
        self.insert(keys, value)
    }

    fn get(&self, key: &str) -> Option<HashSet<Vector>> {
        self.storage.get(key).cloned()
    }

    fn get_with_metadata(&self, key: &str) -> Option<Vec<VectorWithMetadata>> {
        self.storage.get(key).map(|vectors| {
            vectors
                .iter()
                .map(|v| {
                    let vector_hash = hash_vector(v);
                    let metadata = self
                        .metadata_store
                        .get(&vector_hash)
                        .cloned()
                        .unwrap_or_default();
                    VectorWithMetadata {
                        vector: v.clone(),
                        metadata,
                    }
                })
                .collect()
        })
    }

    fn new() -> Self {
        InMemoryBucket::new()
    }

    fn new_with_persistence(_mode: PersistenceMode) -> Self {
        InMemoryBucket::new()
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
            .filter_map(|h| {
                self.storage
                    .values()
                    .flatten()
                    .find(|v| hash_vector(v) == *h)
                    .map(|v| (*h, v.clone()))
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
            .filter_map(|h| {
                self.storage
                    .values()
                    .flatten()
                    .find(|v| hash_vector(v) == *h)
                    .map(|v| VectorWithMetadata {
                        vector: v.clone(),
                        metadata: self.metadata_store.get(h).cloned().unwrap_or_default(),
                    })
            })
            .collect()
    }

    fn random_sample_excluding_buckets(
        &self,
        exclude_buckets: &HashSet<String>,
        n: usize,
    ) -> Vec<VectorWithMetadata> {
        let mut rng = rand::rng();
        let mut seen_hashes = HashSet::new();
        let mut results = Vec::new();

        for (bucket_key, vectors) in &self.storage {
            if exclude_buckets.contains(bucket_key) {
                continue;
            }
            for v in vectors {
                let h = hash_vector(v);
                if seen_hashes.insert(h) {
                    results.push(VectorWithMetadata {
                        vector: v.clone(),
                        metadata: self.metadata_store.get(&h).cloned().unwrap_or_default(),
                    });
                }
            }
        }

        results.shuffle(&mut rng);
        results.truncate(n);
        results
    }
}

#[derive(Debug, Clone)]
pub enum PersistenceMode {
    Temporary,
    Persistent(String),
}

pub struct RocksDbBucket {
    storage: RocksDB,
    db_path: String,
    persistence_mode: PersistenceMode,
}

impl Storage for RocksDbBucket {
    fn insert(&mut self, keys: &[&str], value: Vector) -> anyhow::Result<()> {
        let vector_hash = hash_vector(&value);
        let vector_key = format!("v:{vector_hash}");

        self.storage.write_batch(|batch| {
            let vector_bytes = value
                .as_ref()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>();
            batch.put(&vector_key, vector_bytes);

            for key in keys {
                let bucket_entry_key = format!("b:{key}:{vector_hash}");
                batch.put(bucket_entry_key, vector_hash.to_be_bytes());
            }
        })?;

        Ok(())
    }

    fn insert_with_metadata(
        &mut self,
        keys: &[&str],
        value: Vector,
        metadata: Metadata,
    ) -> anyhow::Result<()> {
        let vector_hash = hash_vector(&value);
        let vector_key = format!("v:{vector_hash}");
        let metadata_key = format!("m:{vector_hash}");

        self.storage.write_batch(|batch| {
            let vector_bytes = value
                .as_ref()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>();
            batch.put(&vector_key, vector_bytes);

            let metadata_bytes =
                bincode::serialize(&metadata).expect("Failed to serialize metadata");
            batch.put(&metadata_key, metadata_bytes);

            for key in keys {
                let bucket_entry_key = format!("b:{key}:{vector_hash}");
                batch.put(bucket_entry_key, vector_hash.to_be_bytes());
            }
        })?;

        Ok(())
    }

    fn get(&self, key: &str) -> Option<HashSet<Vector>> {
        let prefix = format!("b:{key}:");

        let hashes: Vec<u64> = self
            .storage
            .prefix_iterator(&prefix)
            .filter_map(|(_, hash_bytes)| {
                hash_bytes.as_ref().try_into().ok().map(u64::from_be_bytes)
            })
            .collect();

        if hashes.is_empty() {
            return None;
        }

        let vector_keys: Vec<String> = hashes.iter().map(|h| format!("v:{h}")).collect();
        let vector_data = self.storage.multi_get(&vector_keys);

        let mut vectors = HashSet::new();
        for data in vector_data.into_iter().flatten() {
            if data.len() % 8 == 0 {
                let floats: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                vectors.insert(Vector::new(floats));
            }
        }

        (!vectors.is_empty()).then_some(vectors)
    }

    fn get_with_metadata(&self, key: &str) -> Option<Vec<VectorWithMetadata>> {
        let prefix = format!("b:{key}:");

        let hashes: Vec<u64> = self
            .storage
            .prefix_iterator(&prefix)
            .filter_map(|(_, hash_bytes)| {
                hash_bytes
                    .as_ref()
                    .try_into()
                    .ok()
                    .map(u64::from_be_bytes)
            })
            .collect();

        if hashes.is_empty() {
            return None;
        }

        let vector_keys: Vec<String> = hashes.iter().map(|h| format!("v:{h}")).collect();
        let metadata_keys: Vec<String> = hashes.iter().map(|h| format!("m:{h}")).collect();

        let vectors = self.storage.multi_get(&vector_keys);
        let metadatas = self.storage.multi_get(&metadata_keys);

        let mut results = Vec::new();
        for (vec_data, meta_data) in vectors.into_iter().zip(metadatas.into_iter()) {
            if let Some(data) = vec_data {
                if data.len() % 8 != 0 {
                    continue;
                }
                let floats: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                let metadata = meta_data
                    .and_then(|bytes| bincode::deserialize(&bytes).ok())
                    .unwrap_or_default();

                results.push(VectorWithMetadata {
                    vector: Vector::new(floats),
                    metadata,
                });
            }
        }

        (!results.is_empty()).then_some(results)
    }

    fn new() -> Self {
        Self::new_with_persistence(PersistenceMode::Temporary)
    }

    fn new_with_persistence(mode: PersistenceMode) -> Self {
        let (db_path, persistence_mode) = match mode {
            PersistenceMode::Temporary => (
                format!("/tmp/lsh_db_{}", std::process::id()),
                PersistenceMode::Temporary,
            ),
            PersistenceMode::Persistent(path) => (path.clone(), PersistenceMode::Persistent(path)),
        };

        RocksDbBucket {
            storage: RocksDB::new(&db_path).expect("Failed to create RocksDB instance"),
            db_path,
            persistence_mode,
        }
    }

    fn get_hashes(&self, key: &str) -> Vec<u64> {
        let prefix = format!("b:{key}:");
        self.storage
            .prefix_iterator(&prefix)
            .filter_map(|(_, hash_bytes)| {
                hash_bytes.as_ref().try_into().ok().map(u64::from_be_bytes)
            })
            .collect()
    }

    fn load_vectors(&self, hashes: &[u64]) -> Vec<(u64, Vector)> {
        if hashes.is_empty() {
            return Vec::new();
        }

        const BATCH_SIZE: usize = 2000;

        hashes
            .par_chunks(BATCH_SIZE)
            .flat_map(|batch| {
                let vector_keys: Vec<String> = batch.iter().map(|h| format!("v:{h}")).collect();
                let vectors = self.storage.multi_get(&vector_keys);

                batch
                    .iter()
                    .zip(vectors)
                    .filter_map(|(hash, vec_data)| {
                        let data = vec_data?;
                        if data.len() % 8 != 0 {
                            return None;
                        }
                        let floats: Vec<f64> = data
                            .chunks_exact(8)
                            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                            .collect();
                        Some((*hash, Vector::new(floats)))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn load_metadata(&self, hashes: &[u64]) -> Vec<Metadata> {
        if hashes.is_empty() {
            return Vec::new();
        }

        let metadata_keys: Vec<String> = hashes.iter().map(|h| format!("m:{h}")).collect();
        let metadatas = self.storage.multi_get(&metadata_keys);

        metadatas
            .into_iter()
            .map(|meta_data| {
                meta_data
                    .and_then(|bytes| bincode::deserialize(&bytes).ok())
                    .unwrap_or_default()
            })
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
                let vector_keys: Vec<String> = batch.iter().map(|h| format!("v:{h}")).collect();
                let metadata_keys: Vec<String> = batch.iter().map(|h| format!("m:{h}")).collect();

                let vectors = self.storage.multi_get(&vector_keys);
                let metadatas = self.storage.multi_get(&metadata_keys);

                vectors
                    .into_iter()
                    .zip(metadatas)
                    .filter_map(|(vec_data, meta_data)| {
                        let data = vec_data?;
                        if data.len() % 8 != 0 {
                            return None;
                        }
                        let floats: Vec<f64> = data
                            .chunks_exact(8)
                            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                            .collect();
                        let metadata = meta_data
                            .and_then(|bytes| bincode::deserialize(&bytes).ok())
                            .unwrap_or_default();
                        Some(VectorWithMetadata {
                            vector: Vector::new(floats),
                            metadata,
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
        let sampled_keys = self.storage.sample_keys("b:", n * 5);

        let mut hashes = Vec::new();
        for key in sampled_keys {
            if let Some(rest) = key.strip_prefix("b:") {
                if let Some(last_colon) = rest.rfind(':') {
                    let bucket_key = &rest[..last_colon];
                    if !exclude_buckets.contains(bucket_key) {
                        if let Ok(hash) = rest[last_colon + 1..].parse::<u64>() {
                            hashes.push(hash);
                        }
                    }
                }
            }
        }

        hashes.sort();
        hashes.dedup();
        hashes.truncate(n);

        self.load_vectors_with_metadata(&hashes)
    }
}

impl RocksDbBucket {
    pub fn compact(&self) {
        self.storage.compact();
    }

    pub fn random_sample_with_metadata(&self, n: usize) -> Vec<VectorWithMetadata> {
        let keys = self.storage.sample_keys("v:", n);
        let hashes: Vec<u64> = keys
            .iter()
            .filter_map(|k| k.strip_prefix("v:"))
            .filter_map(|h| h.parse().ok())
            .collect();
        self.load_vectors_with_metadata(&hashes)
    }
}

impl Drop for RocksDbBucket {
    fn drop(&mut self) {
        match self.persistence_mode {
            PersistenceMode::Temporary => {
                if let Err(e) = std::fs::remove_dir_all(&self.db_path) {
                    eprintln!(
                        "Warning: Failed to clean up temporary RocksDB directory {}: {}",
                        self.db_path, e
                    );
                }
            }
            PersistenceMode::Persistent(_) => {
                // Don't delete persistent databases
            }
        }
    }
}

pub struct Results {
    results: Vec<f64>,
    similarity: f64,
}

pub struct ResultsWithMetadata {
    pub results: Vec<f64>,
    pub metadata: Metadata,
    pub similarity: f64,
}

pub struct LshDB<T = InMemoryBucket> {
    lsh: VectorLSH,
    similarity_threshold: f64,
    buckets: T,
}

impl LshDB<RocksDbBucket> {
    pub fn persistent(
        num_hashes: usize,
        num_bands: usize,
        dimension: usize,
        similarity_threshold: f64,
        db_path: String,
    ) -> anyhow::Result<Self> {
        Self::new_with_persistence(
            num_hashes,
            num_bands,
            dimension,
            similarity_threshold,
            PersistenceMode::Persistent(db_path),
        )
    }

    pub fn compact(&self) {
        self.buckets.compact();
    }
}

impl<T: Storage + Sync> LshDB<T> {
    pub fn new(
        num_hashes: usize,
        num_bands: usize,
        dimension: usize,
        similarity_threshold: f64,
    ) -> anyhow::Result<Self> {
        Ok(LshDB {
            lsh: VectorLSH::new(num_hashes, num_bands, dimension)?,
            similarity_threshold,
            buckets: T::new(),
        })
    }

    pub fn new_with_persistence(
        num_hashes: usize,
        num_bands: usize,
        dimension: usize,
        similarity_threshold: f64,
        persistence_mode: PersistenceMode,
    ) -> anyhow::Result<Self> {
        Ok(LshDB {
            lsh: VectorLSH::new(num_hashes, num_bands, dimension)?,
            similarity_threshold,
            buckets: T::new_with_persistence(persistence_mode),
        })
    }

    pub fn insert(&mut self, vec: Vec<f64>) -> anyhow::Result<()> {
        if vec.len() != self.lsh.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.lsh.dimension,
                vec.len()
            ));
        }

        let buckets = self.lsh.get_bucket_keys(&vec)?;
        let bucket_refs: Vec<&str> = buckets.iter().map(|s| s.as_str()).collect();
        self.buckets
            .insert(&bucket_refs, Vector::new(vec.clone()))?;
        Ok(())
    }

    pub fn insert_with_metadata(
        &mut self,
        vec: Vec<f64>,
        metadata: Metadata,
    ) -> anyhow::Result<()> {
        if vec.len() != self.lsh.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.lsh.dimension,
                vec.len()
            ));
        }

        let buckets = self.lsh.get_bucket_keys(&vec)?;
        let bucket_refs: Vec<&str> = buckets.iter().map(|s| s.as_str()).collect();
        self.buckets
            .insert_with_metadata(&bucket_refs, Vector::new(vec.clone()), metadata)?;
        Ok(())
    }

    pub fn query(&self, vec: Vec<f64>, n: usize) -> anyhow::Result<Vec<Vec<f64>>> {
        if vec.len() != self.lsh.dimension {
            return Err(anyhow::anyhow!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.lsh.dimension,
                vec.len()
            ));
        }

        let bucket_keys = self.lsh.get_bucket_keys(&vec)?;
        let bucket_results: Vec<HashSet<Vector>> = bucket_keys
            .par_iter()
            .filter_map(|key| self.buckets.get(key))
            .collect();

        let mut candidates = HashSet::new();
        for bucket_entries in bucket_results {
            candidates.extend(bucket_entries.into_iter());
        }

        let q_vec = Vector::new(vec);
        let threshold = self.similarity_threshold;

        let mut results: Vec<Results> = candidates
            .into_par_iter()
            .filter_map(|candidate| {
                if candidate.equal(&q_vec) {
                    return None;
                }
                let similarity = candidate.cosine_similarity(&q_vec).ok()?;
                if similarity >= threshold {
                    Some(Results {
                        results: candidate.into(),
                        similarity,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(n);

        Ok(results.into_iter().map(|r| r.results).collect())
    }

    pub fn query_with_metadata(
        &self,
        vec: Vec<f64>,
        n: usize,
    ) -> anyhow::Result<Vec<ResultsWithMetadata>> {
        if vec.len() != self.lsh.dimension {
            return Err(anyhow::anyhow!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.lsh.dimension,
                vec.len()
            ));
        }

        let bucket_keys = self.lsh.get_bucket_keys(&vec)?;
        let bucket_results: Vec<HashSet<Vector>> = bucket_keys
            .par_iter()
            .filter_map(|key| self.buckets.get(key))
            .collect();

        let mut candidates = HashSet::new();
        for bucket_entries in bucket_results {
            candidates.extend(bucket_entries.into_iter());
        }

        let q_vec = Vector::new(vec);
        let threshold = self.similarity_threshold;

        let mut top_results: Vec<(u64, Vec<f64>, f64)> = candidates
            .into_par_iter()
            .filter_map(|candidate| {
                if candidate.equal(&q_vec) {
                    return None;
                }
                let similarity = candidate.cosine_similarity(&q_vec).ok()?;
                if similarity >= threshold {
                    let h = hash_vector(&candidate);
                    Some((h, candidate.into(), similarity))
                } else {
                    None
                }
            })
            .collect();

        top_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        top_results.truncate(n);

        let top_hashes: Vec<u64> = top_results.iter().map(|(h, _, _)| *h).collect();
        let metadatas = self.buckets.load_metadata(&top_hashes);

        let results: Vec<ResultsWithMetadata> = top_results
            .into_iter()
            .zip(metadatas)
            .map(|((_, results, similarity), metadata)| ResultsWithMetadata {
                results,
                metadata,
                similarity,
            })
            .collect();

        Ok(results)
    }

    pub fn query_dissimilar_with_metadata(
        &self,
        vec: Vec<f64>,
        n: usize,
    ) -> anyhow::Result<Vec<ResultsWithMetadata>> {
        if vec.len() != self.lsh.dimension {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        let original_bucket_keys: HashSet<String> =
            self.lsh.get_bucket_keys(&vec)?.into_iter().collect();

        let samples = self
            .buckets
            .random_sample_excluding_buckets(&original_bucket_keys, n * 10);

        let original = Vector::new(vec);

        let mut results: Vec<ResultsWithMetadata> = samples
            .into_iter()
            .filter_map(|entry| {
                let similarity = entry.vector.cosine_similarity(&original).ok()?;
                Some(ResultsWithMetadata {
                    results: entry.vector.into(),
                    metadata: entry.metadata,
                    similarity,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(n);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db() -> LshDB {
        LshDB::new(64, 16, 5, 0.8).expect("Failed to create test database")
    }

    #[test]
    fn test_database_insert_and_find() {
        let mut db = create_test_db();

        let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vec2 = vec![1.1, 2.1, 3.0, 4.0, 5.0];
        let vec3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let vec4 = vec![-1.0, -2.0, -3.0, -4.0, -5.0];

        db.insert(vec1.clone()).expect("Should insert");
        db.insert(vec2.clone()).expect("Should insert");
        db.insert(vec3.clone()).expect("Should insert");
        db.insert(vec4.clone()).expect("Should insert");

        let results = db.query(vec1.clone(), 5).expect("Should get results");
        assert_eq!(results.len(), 2);

        let similarity = Vector::new(vec1)
            .cosine_similarity(&Vector::new(results[0].clone()))
            .expect("Bad Cosine");
        assert!(similarity > 0.8);

        let results = db.query(vec4.clone(), 5).expect("Should get results");
        assert_eq!(results.len(), 0);
    }
}
