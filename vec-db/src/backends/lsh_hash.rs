use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use crate::math::vector::Vector;

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

pub struct Results {
    results: Vec<f64>,
    similarity: f64,
}

pub struct LshDB {
    lsh: VectorLSH,
    similarity_threshold: f64,
    buckets: HashMap<String, HashSet<Vector>>,
}

impl LshDB {
    pub fn new(
        num_hashes: usize,
        num_bands: usize,
        dimension: usize,
        similarity_threshold: f64,
    ) -> Self {
        LshDB {
            lsh: VectorLSH::new(num_hashes, num_bands, dimension),
            similarity_threshold,
            buckets: HashMap::new(),
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
            self.buckets
                .entry(bucket_key)
                .or_default()
                .insert(Vector::new(vec.clone()));
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
            // Use actual cosine similarity instead of LSH estimate for final ranking
            let similarity = candidate_vec.cosine_similarity(&q_vec)?;

            if similarity >= self.similarity_threshold {
                results.push(Results {
                    results: candidate_vec.as_vec(),
                    similarity,
                });
            }
        }

        // Sort by actual similarity (descending)
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
        LshDB::new(64, 16, 5, 0.8) // Much more reasonable threshold!
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

    #[test]
    fn test_cosine_similarity() {
        let vec1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let vec2 = Vector::new(vec![0.0, 1.0, 0.0]);
        let vec3 = Vector::new(vec![1.0, 1.0, 0.0]);
        let vec4 = Vector::new(vec![2.0, 0.0, 0.0]);

        assert!(((vec1.cosine_similarity(&vec2)).expect("Bad cosine") - 0.0).abs() < 1e-10);
        assert!(((vec1.cosine_similarity(&vec4)).expect("Bad cosine") - 1.0).abs() < 1e-10);
        assert!(
            ((vec1.cosine_similarity(&vec3)).expect("Bad cosine") - (1.0 / 2.0_f64.sqrt())).abs()
                < 1e-10
        );
    }
}
