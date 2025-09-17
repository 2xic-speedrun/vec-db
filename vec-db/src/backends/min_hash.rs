use std::collections::{HashMap, HashSet};

use crate::math::vector::Vector;

pub struct MinHash {
    num_hashes: u64,
    num_bands: u64,
    rows_per_band: u64,
}

impl MinHash {
    pub fn new(num_hashes: u64, num_bands: u64) -> Self {
        assert!(
            num_hashes % num_bands == 0,
            "Num hash must be divisible by num_bands",
        );
        MinHash {
            num_bands,
            num_hashes,
            rows_per_band: num_hashes / num_bands,
        }
    }

    // TODO: does this just break the algorithm?
    fn float64_to_bytes(&self, values: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn min_hash(&self, shingles: &HashSet<Vec<u8>>) -> Vec<u32> {
        if shingles.is_empty() {
            return vec![0; self.num_hashes as usize];
        }
        let seeds: Vec<u32> = (0..self.num_hashes)
            .map(|i| (i as u32).wrapping_mul(2654435761))
            .collect();
        // TODO: optimize this with a map
        let mut signature = vec![u32::MAX; self.num_hashes as usize];
        for shingle in shingles {
            let mut shingle_bytes = [0u8; 4];
            let copy_len = std::cmp::min(4, shingle.len());
            shingle_bytes[..copy_len].copy_from_slice(&shingle[..copy_len]);
            let shingle_int = u32::from_be_bytes(shingle_bytes);

            for (i, &seed) in seeds.iter().enumerate() {
                let hash_val = (shingle_int ^ seed).wrapping_mul(2654435761);
                signature[i] = std::cmp::min(signature[i], hash_val);
            }
        }

        signature
            .into_iter()
            .map(|s| if s == u32::MAX { 0 } else { s })
            .collect()
    }

    fn get_byte_shingles(&self, data: &[u8], shingle_size: usize) -> HashSet<Vec<u8>> {
        let mut set = HashSet::new();
        if data.len() < shingle_size {
            set.insert(data.to_vec());
            return set;
        }

        let step = std::cmp::max(1, shingle_size / 4);

        for i in (0..=data.len().saturating_sub(shingle_size)).step_by(step) {
            set.insert(data[i..i + shingle_size].to_vec());
        }

        set
    }

    fn signature(&self, entry: &Vec<f64>) -> Vec<u32> {
        let bytes = self.float64_to_bytes(&entry);
        let shingles = self.get_byte_shingles(&bytes, 8);
        self.min_hash(&shingles)
    }

    fn get_bucket_key(&self, entry: Vec<f64>) -> Vec<String> {
        let signature = self.signature(&entry);

        //        let mut buckets: Vec<String> = Vec::with_capacity(self.num_bands as usize);
        let mut buckets = vec![String::new(); self.num_bands as usize];
        for band in 0..self.num_bands {
            let start_idx = (band as usize) * (self.rows_per_band as usize);
            let end_idx = start_idx + (self.rows_per_band as usize);
            let band_signature = signature[start_idx..end_idx].to_vec();
            let bucket_key = format!("minhash_{band}_{band_signature:?}");
            buckets[band as usize] = bucket_key;
        }
        buckets
    }

    fn jaccard_similarity(&self, sig1: &[u32], sig2: &[u32]) -> f64 {
        let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
        matches as f64 / sig1.len() as f64
    }
}

pub struct Results {
    results: Vec<f64>,
    similarity: f64,
}

pub struct MinHashDb {
    min_hash: MinHash,
    similarity_threshold: f64,

    buckets: HashMap<String, HashSet<Vector>>,
}

impl MinHashDb {
    pub fn new(num_hashes: u64, num_bands: u64, similarity_threshold: f64) -> Self {
        MinHashDb {
            min_hash: MinHash::new(num_hashes, num_bands),
            buckets: HashMap::new(),
            similarity_threshold,
        }
    }

    pub fn insert(&mut self, vec: Vec<f64>) -> anyhow::Result<()> {
        for bucket_key in self.min_hash.get_bucket_key(vec.clone()) {
            self.buckets
                .entry(bucket_key)
                .or_default()
                .insert(Vector::new(vec.clone()));
        }
        Ok(())
    }

    pub fn query(&self, vec: Vec<f64>, n: usize) -> anyhow::Result<Vec<Vec<f64>>> {
        let mut candidates = HashSet::new();
        for bucket_key in self.min_hash.get_bucket_key(vec.clone()) {
            if let Some(bucket_entries) = self.buckets.get(&bucket_key) {
                candidates.extend(bucket_entries.iter().cloned());
            }
        }

        let mut results = Vec::new();
        let query_signature = self.min_hash.signature(&vec);
        let query = Vector::new(vec);
        for candidate_id in candidates {
            if candidate_id.equal(&query) {
                continue;
            }
            let vec = candidate_id.as_vec();
            let candidate_signature = self.min_hash.signature(&vec);
            let similarity = self
                .min_hash
                .jaccard_similarity(&query_signature, &candidate_signature);

            if similarity >= self.similarity_threshold {
                results.push(Results {
                    results: vec,
                    similarity,
                });
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(n);

        Ok(results.iter().map(|f| f.results.clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        backends::min_hash::{MinHash, MinHashDb},
        math::vector::Vector,
    };

    fn create_test_db() -> MinHashDb {
        MinHashDb {
            min_hash: MinHash::new(64, 16),
            similarity_threshold: 0.6,
            buckets: HashMap::new(),
        }
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
