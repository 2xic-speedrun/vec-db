use crate::{
    math::vector::Vector,
    storage::bucket::{
        hash_vector, BucketStorage, InMemoryBucket, Metadata, PersistenceMode, ResultsWithMetadata,
        RocksDbBucket,
    },
};
use std::collections::HashSet;

pub struct MinHash {
    num_hashes: u64,
    rows_per_band: u64,
}

impl MinHash {
    pub fn new(num_hashes: u64, num_bands: u64) -> Self {
        assert!(
            num_hashes % num_bands == 0,
            "num_hashes must be divisible by num_bands",
        );
        MinHash {
            num_hashes,
            rows_per_band: num_hashes / num_bands,
        }
    }

    fn min_hash(&self, elements: &HashSet<u64>) -> Vec<u64> {
        if elements.is_empty() {
            return vec![0; self.num_hashes as usize];
        }

        let mut signature = vec![u64::MAX; self.num_hashes as usize];
        for &element in elements {
            for (i, sig) in signature.iter_mut().enumerate() {
                let seed = (i as u64).wrapping_mul(0x517cc1b727220a95);
                let hash_val = element.wrapping_add(seed).wrapping_mul(0x9e3779b97f4a7c15);
                *sig = hash_val.min(*sig);
            }
        }

        signature
    }

    fn signature(&self, elements: &HashSet<u64>) -> Vec<u64> {
        self.min_hash(elements)
    }

    fn get_bucket_keys(&self, elements: &HashSet<u64>) -> Vec<String> {
        use std::fmt::Write;
        let signature = self.signature(elements);

        signature
            .chunks(self.rows_per_band as usize)
            .enumerate()
            .map(|(band, chunk)| {
                let mut key = String::with_capacity(8 + chunk.len() * 17);
                write!(key, "m{band}_").expect("write to String cannot fail");
                for &val in chunk {
                    write!(key, "{val:016x}").expect("write to String cannot fail");
                }
                key
            })
            .collect()
    }

    fn jaccard_similarity(&self, sig1: &[u64], sig2: &[u64]) -> f64 {
        let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
        matches as f64 / sig1.len() as f64
    }
}

fn vec_to_set(vec: &[f64]) -> HashSet<u64> {
    vec.iter().map(|&x| x as u64).collect()
}

struct Results {
    results: Vec<f64>,
    similarity: f64,
}

pub struct MinHashDb<T = InMemoryBucket> {
    min_hash: MinHash,
    similarity_threshold: f64,
    buckets: T,
}

impl MinHashDb<RocksDbBucket> {
    pub fn persistent(
        num_hashes: u64,
        num_bands: u64,
        similarity_threshold: f64,
        db_path: String,
    ) -> anyhow::Result<Self> {
        Self::new_with_persistence(
            num_hashes,
            num_bands,
            similarity_threshold,
            PersistenceMode::Persistent(db_path),
        )
    }

    pub fn read_only(
        num_hashes: u64,
        num_bands: u64,
        similarity_threshold: f64,
        db_path: String,
    ) -> anyhow::Result<Self> {
        Self::new_with_persistence(
            num_hashes,
            num_bands,
            similarity_threshold,
            PersistenceMode::ReadOnly(db_path),
        )
    }

    pub fn compact(&self) {
        self.buckets.compact();
    }
}

impl<T: BucketStorage> MinHashDb<T> {
    pub fn new(num_hashes: u64, num_bands: u64, similarity_threshold: f64) -> Self {
        MinHashDb {
            min_hash: MinHash::new(num_hashes, num_bands),
            buckets: T::new(),
            similarity_threshold,
        }
    }

    pub fn new_with_persistence(
        num_hashes: u64,
        num_bands: u64,
        similarity_threshold: f64,
        persistence_mode: PersistenceMode,
    ) -> anyhow::Result<Self> {
        Ok(MinHashDb {
            min_hash: MinHash::new(num_hashes, num_bands),
            buckets: T::new_with_persistence(persistence_mode)?,
            similarity_threshold,
        })
    }

    pub fn insert(&mut self, elements: Vec<f64>) -> anyhow::Result<()> {
        let set = vec_to_set(&elements);
        let bucket_keys = self.min_hash.get_bucket_keys(&set);
        let bucket_refs: Vec<&str> = bucket_keys.iter().map(|s| s.as_str()).collect();
        self.buckets.insert(&bucket_refs, Vector::new(elements))?;
        Ok(())
    }

    pub fn insert_with_metadata(
        &mut self,
        elements: Vec<f64>,
        metadata: Metadata,
    ) -> anyhow::Result<()> {
        let set = vec_to_set(&elements);
        let bucket_keys = self.min_hash.get_bucket_keys(&set);
        let bucket_refs: Vec<&str> = bucket_keys.iter().map(|s| s.as_str()).collect();
        self.buckets
            .insert_with_metadata(&bucket_refs, Vector::new(elements), metadata)?;
        Ok(())
    }

    pub fn query(&self, elements: Vec<f64>, n: usize) -> anyhow::Result<Vec<Vec<f64>>> {
        let query_set = vec_to_set(&elements);
        let mut candidates = HashSet::new();
        for bucket_key in self.min_hash.get_bucket_keys(&query_set) {
            if let Some(bucket_entries) = self.buckets.get(&bucket_key) {
                candidates.extend(bucket_entries.into_iter());
            }
        }

        let mut results = Vec::new();
        let query_signature = self.min_hash.signature(&query_set);
        let query = Vector::new(elements);
        for candidate in candidates {
            if candidate.equal(&query) {
                continue;
            }
            let vec_data: Vec<f64> = candidate.into();
            let candidate_set = vec_to_set(&vec_data);
            let candidate_signature = self.min_hash.signature(&candidate_set);
            let similarity = self
                .min_hash
                .jaccard_similarity(&query_signature, &candidate_signature);

            if similarity >= self.similarity_threshold {
                results.push(Results {
                    results: vec_data,
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

        Ok(results.into_iter().map(|f| f.results).collect())
    }

    pub fn query_with_metadata(
        &self,
        vec: Vec<f64>,
        n: usize,
    ) -> anyhow::Result<Vec<ResultsWithMetadata>> {
        let query_set = vec_to_set(&vec);
        let mut candidates = HashSet::new();
        for bucket_key in self.min_hash.get_bucket_keys(&query_set) {
            if let Some(bucket_entries) = self.buckets.get(&bucket_key) {
                candidates.extend(bucket_entries.into_iter());
            }
        }

        let query_signature = self.min_hash.signature(&query_set);
        let query = Vector::new(vec);
        let mut results = Vec::new();

        for candidate in candidates {
            if candidate.equal(&query) {
                continue;
            }
            let hash = hash_vector(&candidate);
            let vec_data: Vec<f64> = candidate.into();
            let candidate_set = vec_to_set(&vec_data);
            let candidate_signature = self.min_hash.signature(&candidate_set);
            let similarity = self
                .min_hash
                .jaccard_similarity(&query_signature, &candidate_signature);

            if similarity >= self.similarity_threshold {
                let metadata = self.buckets.load_metadata(&[hash]);
                results.push(ResultsWithMetadata {
                    results: vec_data,
                    metadata: metadata.into_iter().next().unwrap_or_default(),
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

        Ok(results)
    }

    pub fn query_dissimilar_with_metadata(
        &self,
        vec: Vec<f64>,
        n: usize,
    ) -> anyhow::Result<Vec<ResultsWithMetadata>> {
        let query_set = vec_to_set(&vec);
        let original_bucket_keys: HashSet<String> = self
            .min_hash
            .get_bucket_keys(&query_set)
            .into_iter()
            .collect();

        let samples = self
            .buckets
            .random_sample_excluding_buckets(&original_bucket_keys, n * 10);

        let query_signature = self.min_hash.signature(&query_set);

        let mut results: Vec<ResultsWithMetadata> = samples
            .into_iter()
            .map(|entry| {
                let vec_data: Vec<f64> = entry.vector.into();
                let candidate_set = vec_to_set(&vec_data);
                let candidate_signature = self.min_hash.signature(&candidate_set);
                let similarity = self
                    .min_hash
                    .jaccard_similarity(&query_signature, &candidate_signature);
                ResultsWithMetadata {
                    results: vec_data,
                    metadata: entry.metadata,
                    similarity,
                }
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
    use crate::backends::min_hash::MinHashDb;
    use crate::math::vector::Vector;

    fn create_test_db() -> MinHashDb {
        MinHashDb::new(64, 16, 0.6)
    }

    #[test]
    fn test_database_insert_and_find() {
        let mut db = create_test_db();

        let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vec2 = vec![1.1, 2.1, 3.0, 4.0, 5.0];
        let vec3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        db.insert(vec1.clone()).expect("Should insert");
        db.insert(vec2.clone()).expect("Should insert");
        db.insert(vec3.clone()).expect("Should insert");

        let results = db.query(vec1.clone(), 5).expect("Should get results");
        assert!(Vector::new(results[0].clone()).equal(&Vector::new(vec2)));
        assert_eq!(results.len(), 1);

        let results = db.query(vec3.clone(), 5).expect("Should get results");
        assert_eq!(results.len(), 0);
    }
}
