use crate::{
    math::vector::Vector,
    storage::bucket::{hash_vector, BucketStorage},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::{
    collections::HashSet,
    fmt::Write,
    sync::{Arc, Mutex},
};

pub use crate::storage::bucket::{
    InMemoryBucket, Metadata, PersistenceMode, ResultsWithMetadata, RocksDbBucket,
};

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

        for random_vec in &self.random_vectors {
            let dot_product = random_vec.l1_dot(vector)?;
            signature.push(if dot_product >= 0.0 { 1 } else { 0 });
        }

        Ok(signature)
    }

    fn get_bucket_keys(&self, vector: &[f64]) -> anyhow::Result<Vec<String>> {
        let signature = self.hash_vector(vector)?;
        let mut bucket_keys = Vec::with_capacity(self.num_bands);

        for (band, chunk) in signature.chunks(self.rows_per_band).enumerate() {
            let mut key = String::with_capacity(self.rows_per_band + 10);
            write!(key, "b{band}_").expect("write to String cannot fail");
            for &bit in chunk {
                key.push(if bit == 1 { '1' } else { '0' });
            }
            bucket_keys.push(key);
        }

        Ok(bucket_keys)
    }
}

struct Results {
    results: Vec<f64>,
    similarity: f64,
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

    pub fn read_only(
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
            PersistenceMode::ReadOnly(db_path),
        )
    }

    pub fn compact(&self) {
        self.buckets.compact();
    }
}

impl<T: BucketStorage> LshDB<T> {
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
            buckets: T::new_with_persistence(persistence_mode)?,
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
        self.buckets.insert(&bucket_refs, Vector::new(vec))?;
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
            .insert_with_metadata(&bucket_refs, Vector::new(vec), metadata)?;
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
        if candidates.len() > 100000 {
            eprintln!(
                "[LSH WARNING] {} candidates found - consider mean-centering vectors or using different encoding",
                candidates.len()
            );
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
        if candidates.len() > 100000 {
            eprintln!(
                "[LSH WARNING] {} candidates found - consider mean-centering vectors or using different encoding",
                candidates.len()
            );
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
