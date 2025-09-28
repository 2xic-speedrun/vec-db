use crate::{math::vector::Vector, storage::rocksdb::RocksDB};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
};

pub struct VectorLSH {
    num_hashes: usize,
    num_bands: usize,
    rows_per_band: usize,
    dimension: usize,
    random_vectors: Vec<Vector>,
}

impl VectorLSH {
    pub fn new(num_hashes: usize, num_bands: usize, dimension: usize) -> Self {
        assert!(
            num_hashes % num_bands == 0,
            "num_hashes must be divisible by num_bands"
        );
        let rows_per_band = num_hashes / num_bands;
        let rng_seed = 12345u64;
        let rng = Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(rng_seed)));

        let mut random_vectors = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            let vec = Vector::rand(dimension, &rng).norm();
            random_vectors.push(vec);
        }

        VectorLSH {
            num_hashes,
            num_bands,
            rows_per_band,
            dimension,
            random_vectors,
        }
    }

    fn hash_vector(&self, vector: &[f64]) -> anyhow::Result<Vec<u32>> {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");

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

pub trait Storage {
    fn new() -> Self;
    fn insert(&mut self, key: String, value: Vector) -> anyhow::Result<()>;
    fn get(&self, key: &str) -> Option<HashSet<Vector>>;
}

#[derive(Default)]
pub struct InMemoryBucket {
    storage: HashMap<String, HashSet<Vector>>,
}

impl InMemoryBucket {
    pub fn new() -> Self {
        InMemoryBucket {
            storage: HashMap::new(),
        }
    }
}

impl Storage for InMemoryBucket {
    fn insert(&mut self, key: String, value: Vector) -> anyhow::Result<()> {
        self.storage.entry(key.clone()).or_default().insert(value);
        Ok(())
    }

    fn get(&self, key: &str) -> Option<HashSet<Vector>> {
        self.storage.get(key).cloned()
    }

    fn new() -> Self {
        InMemoryBucket::new()
    }
}

pub struct RocksDbBucket {
    storage: RocksDB,
}

fn hash_vector(vector: &Vector) -> u64 {
    let mut hasher = DefaultHasher::new();
    vector.hash(&mut hasher);
    hasher.finish()
}

impl Storage for RocksDbBucket {
    fn insert(&mut self, key: String, value: Vector) -> anyhow::Result<()> {
        let hash = hash_vector(&value);
        let composite_key = format!("{key}:{hash}");

        let data = bincode::serialize(&value).unwrap();
        self.storage.put(composite_key, data).unwrap();
        Ok(())
    }

    fn get(&self, key: &str) -> Option<HashSet<Vector>> {
        self.storage.get(key).ok()
    }

    fn new() -> Self {
        RocksDbBucket {
            storage: RocksDB::new("lsh_db").unwrap(),
        }
    }
}

pub struct Results {
    results: Vec<f64>,
    similarity: f64,
}

pub struct LshDB<T = InMemoryBucket> {
    lsh: VectorLSH,
    similarity_threshold: f64,
    buckets: T,
}

impl<T: Storage> LshDB<T> {
    pub fn new(
        num_hashes: usize,
        num_bands: usize,
        dimension: usize,
        similarity_threshold: f64,
    ) -> Self {
        LshDB {
            lsh: VectorLSH::new(num_hashes, num_bands, dimension),
            similarity_threshold,
            buckets: T::new(),
        }
    }

    pub fn insert(&mut self, vec: Vec<f64>) -> anyhow::Result<()> {
        if vec.len() != self.lsh.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.lsh.dimension,
                vec.len()
            ));
        }

        for bucket_key in self.lsh.get_bucket_keys(&vec)? {
            self.buckets.insert(bucket_key, Vector::new(vec.clone()))?;
        }
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

        let mut candidates = HashSet::new();
        for bucket_key in self.lsh.get_bucket_keys(&vec)? {
            if let Some(bucket_entries) = self.buckets.get(&bucket_key) {
                candidates.extend(bucket_entries.iter().cloned());
            }
        }

        let mut results = Vec::new();
        let query_vector = Vector::new(vec.clone());
        let q_vec = Vector::new(vec);

        for candidate in candidates {
            if candidate.equal(&query_vector) {
                continue;
            }

            let candidate_vec = candidate;
            let similarity = candidate_vec.cosine_similarity(&q_vec)?;

            if similarity >= self.similarity_threshold {
                results.push(Results {
                    results: candidate_vec.as_vec(),
                    similarity,
                });
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(n);

        Ok(results.into_iter().map(|r| r.results).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db() -> LshDB {
        LshDB::new(64, 16, 5, 0.8)
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
