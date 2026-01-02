use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::math::vector::Vector;
use crate::storage::bucket::PersistenceMode;
use crate::storage::rocksdb::RocksDB;

fn hash_vector(vector: &Vector) -> u64 {
    let mut hasher = DefaultHasher::new();
    vector.hash(&mut hasher);
    hasher.finish()
}

pub type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    distance: f64,
    node_id: usize,
}

pub struct ResultWithMetadata {
    pub vector: Vec<f64>,
    pub metadata: Option<Metadata>,
    pub similarity: f64,
}

impl From<ResultWithMetadata> for (Vec<f64>, Option<HashMap<String, String>>, f64) {
    fn from(r: ResultWithMetadata) -> Self {
        (
            r.vector,
            r.metadata.map(|m| m.into_iter().collect()),
            r.similarity,
        )
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match other.distance.partial_cmp(&self.distance) {
            Some(ordering) => ordering,
            None => std::cmp::Ordering::Equal,
        }
    }
}

pub trait HnswStorage: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn find(&self, vector: &Vector) -> Option<usize>;
    fn load_vector(&self, node_id: usize) -> Option<Vector>;
    fn load_neighbors(&self, node_id: usize) -> Vec<usize>;
    fn load_metadata(&self, node_id: usize) -> Option<Metadata>;
    fn set_metadata(&mut self, node_id: usize, metadata: Metadata);
    fn insert(
        &mut self,
        vector: &Vector,
        neighbors: &[usize],
        metadata: Option<Metadata>,
        backlinks: &[(usize, Vec<usize>)],
    ) -> anyhow::Result<usize>;
}

#[derive(Debug, Clone)]
struct Node {
    vector: Vector,
    neighbors: Vec<usize>,
}

#[derive(Default)]
pub struct InMemoryHnswStorage {
    nodes: Vec<Node>,
    metadata: Vec<Metadata>,
    hash_to_id: HashMap<u64, usize>,
}

impl HnswStorage for InMemoryHnswStorage {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn find(&self, vector: &Vector) -> Option<usize> {
        let hash = hash_vector(vector);
        self.hash_to_id.get(&hash).copied()
    }

    fn load_vector(&self, node_id: usize) -> Option<Vector> {
        self.nodes.get(node_id).map(|n| n.vector.clone())
    }

    fn load_neighbors(&self, node_id: usize) -> Vec<usize> {
        self.nodes
            .get(node_id)
            .map(|n| n.neighbors.clone())
            .unwrap_or_default()
    }

    fn load_metadata(&self, node_id: usize) -> Option<Metadata> {
        self.metadata.get(node_id).cloned()
    }

    fn set_metadata(&mut self, node_id: usize, metadata: Metadata) {
        if node_id < self.metadata.len() {
            self.metadata[node_id] = metadata;
        }
    }

    fn insert(
        &mut self,
        vector: &Vector,
        neighbors: &[usize],
        metadata: Option<Metadata>,
        backlinks: &[(usize, Vec<usize>)],
    ) -> anyhow::Result<usize> {
        let node_id = self.nodes.len();
        let hash = hash_vector(vector);

        for (neighbor_id, new_neighbors) in backlinks {
            if let Some(node) = self.nodes.get_mut(*neighbor_id) {
                node.neighbors = new_neighbors.clone();
            }
        }

        self.hash_to_id.insert(hash, node_id);
        self.nodes.push(Node {
            vector: vector.clone(),
            neighbors: neighbors.to_vec(),
        });
        self.metadata.push(metadata.unwrap_or_default());

        Ok(node_id)
    }
}

fn bytes_to_floats(data: &[u8]) -> Option<Vec<f64>> {
    (data.len() % 8 == 0).then(|| {
        data.chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    })
}

fn floats_to_bytes(floats: &[f64]) -> Vec<u8> {
    floats.iter().flat_map(|&x| x.to_le_bytes()).collect()
}

pub struct RocksDbHnswStorage {
    storage: RocksDB,
    node_count: usize,
    persistence_mode: PersistenceMode,
    db_path: String,
}

impl RocksDbHnswStorage {
    pub fn persistent(db_path: String) -> anyhow::Result<Self> {
        Self::new_with_persistence(PersistenceMode::Persistent(db_path))
    }

    pub fn read_only(db_path: String) -> anyhow::Result<Self> {
        Self::new_with_persistence(PersistenceMode::ReadOnly(db_path))
    }

    fn new_with_persistence(mode: PersistenceMode) -> anyhow::Result<Self> {
        let (db_path, read_only) = match &mode {
            PersistenceMode::Temporary => (format!("/tmp/hnsw_db_{}", std::process::id()), false),
            PersistenceMode::Persistent(path) => (path.clone(), false),
            PersistenceMode::ReadOnly(path) => (path.clone(), true),
        };

        let storage = if read_only {
            RocksDB::open_read_only(&db_path)?
        } else {
            RocksDB::new(&db_path)?
        };

        let node_count = storage
            .get("node_count")
            .ok()
            .flatten()
            .and_then(|b| b.try_into().ok().map(usize::from_le_bytes))
            .unwrap_or(0);

        Ok(Self {
            storage,
            node_count,
            persistence_mode: mode,
            db_path,
        })
    }

    pub fn compact(&self) {
        self.storage.compact();
    }
}

impl HnswStorage for RocksDbHnswStorage {
    fn len(&self) -> usize {
        self.node_count
    }

    fn find(&self, vector: &Vector) -> Option<usize> {
        let hash = hash_vector(vector);
        let key = format!("h:{hash}");
        self.storage
            .get(&key)
            .ok()
            .flatten()
            .and_then(|b| b.try_into().ok().map(usize::from_le_bytes))
    }

    fn load_vector(&self, node_id: usize) -> Option<Vector> {
        let key = format!("v:{node_id}");
        self.storage
            .get(&key)
            .ok()
            .flatten()
            .and_then(|b| bytes_to_floats(&b))
            .map(Vector::new)
    }

    fn load_neighbors(&self, node_id: usize) -> Vec<usize> {
        let key = format!("n:{node_id}");
        self.storage
            .get(&key)
            .ok()
            .flatten()
            .and_then(|b| bincode::deserialize(&b).ok())
            .unwrap_or_default()
    }

    fn load_metadata(&self, node_id: usize) -> Option<Metadata> {
        let key = format!("m:{node_id}");
        self.storage
            .get(&key)
            .ok()
            .flatten()
            .and_then(|b| bincode::deserialize(&b).ok())
    }

    fn set_metadata(&mut self, node_id: usize, metadata: Metadata) {
        let key = format!("m:{node_id}");
        let bytes = bincode::serialize(&metadata).expect("Failed to serialize metadata");
        let _ = self.storage.put(&key, &bytes);
    }

    fn insert(
        &mut self,
        vector: &Vector,
        neighbors: &[usize],
        metadata: Option<Metadata>,
        backlinks: &[(usize, Vec<usize>)],
    ) -> anyhow::Result<usize> {
        let node_id = self.node_count;
        let hash = hash_vector(vector);

        let hash_key = format!("h:{hash}");
        let vector_key = format!("v:{node_id}");
        let neighbors_key = format!("n:{node_id}");
        let vector_bytes = floats_to_bytes(vector.as_ref());
        let neighbors_bytes = bincode::serialize(neighbors)?;
        let metadata_bytes = metadata.map(|m| bincode::serialize(&m)).transpose()?;

        self.storage.write_batch(|batch| {
            for (neighbor_id, new_neighbors) in backlinks {
                let key = format!("n:{neighbor_id}");
                let bytes = bincode::serialize(new_neighbors).expect("serialize");
                batch.put(&key, &bytes);
            }
            batch.put(&hash_key, node_id.to_le_bytes());
            batch.put(&vector_key, &vector_bytes);
            batch.put(&neighbors_key, &neighbors_bytes);
            if let Some(ref bytes) = metadata_bytes {
                batch.put(format!("m:{node_id}"), bytes);
            }
            batch.put("node_count", (node_id + 1).to_le_bytes());
        })?;

        self.node_count += 1;
        Ok(node_id)
    }
}

impl Drop for RocksDbHnswStorage {
    fn drop(&mut self) {
        if let PersistenceMode::Temporary = self.persistence_mode {
            let _ = std::fs::remove_dir_all(&self.db_path);
        }
    }
}

pub struct HnswDB<T: HnswStorage = InMemoryHnswStorage> {
    storage: T,
    max_connections: usize,
    similarity_threshold: f64,
}

impl HnswDB<InMemoryHnswStorage> {
    pub fn new(max_connections: usize, similarity_threshold: f64) -> Self {
        Self {
            storage: InMemoryHnswStorage::default(),
            max_connections,
            similarity_threshold,
        }
    }
}

impl HnswDB<RocksDbHnswStorage> {
    pub fn persistent(
        max_connections: usize,
        similarity_threshold: f64,
        db_path: String,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            storage: RocksDbHnswStorage::persistent(db_path)?,
            max_connections,
            similarity_threshold,
        })
    }

    pub fn read_only(
        max_connections: usize,
        similarity_threshold: f64,
        db_path: String,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            storage: RocksDbHnswStorage::read_only(db_path)?,
            max_connections,
            similarity_threshold,
        })
    }

    pub fn compact(&self) {
        self.storage.compact();
    }
}

impl<T: HnswStorage> HnswDB<T> {
    fn do_insert(&mut self, vector: Vector, metadata: Option<Metadata>) -> anyhow::Result<()> {
        if let Some(existing_id) = self.storage.find(&vector) {
            if let Some(m) = metadata {
                self.storage.set_metadata(existing_id, m);
            }
            return Ok(());
        }

        let node_id = self.storage.len();
        let neighbors: Vec<usize> = self
            .search(&vector, self.max_connections, None)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let backlinks: Vec<_> = neighbors
            .iter()
            .filter_map(|&neighbor_id| {
                let mut existing = self.storage.load_neighbors(neighbor_id);
                if existing.contains(&node_id) {
                    return None;
                }
                existing.push(node_id);
                if existing.len() > self.max_connections {
                    existing.remove(0);
                }
                Some((neighbor_id, existing))
            })
            .collect();

        self.storage
            .insert(&vector, &neighbors, metadata, &backlinks)?;
        Ok(())
    }

    pub fn insert(&mut self, vector: Vector) -> anyhow::Result<()> {
        self.do_insert(vector, None)
    }

    pub fn insert_with_metadata(
        &mut self,
        vector: Vector,
        metadata: Metadata,
    ) -> anyhow::Result<()> {
        self.do_insert(vector, Some(metadata))
    }

    pub fn query(&self, query: Vector, k: usize) -> anyhow::Result<Vec<Vec<f64>>> {
        let results = self.search(&query, k, Some(self.similarity_threshold))?;
        Ok(results.into_iter().map(|(_, v)| v.into()).collect())
    }

    pub fn query_with_metadata(
        &self,
        query: Vector,
        k: usize,
    ) -> anyhow::Result<Vec<ResultWithMetadata>> {
        let results = self.search(&query, k, Some(self.similarity_threshold))?;

        Ok(results
            .into_iter()
            .map(|(node_id, vector)| {
                let metadata = self.storage.load_metadata(node_id);
                let similarity = vector.cosine_similarity(&query).unwrap_or(0.0);
                ResultWithMetadata {
                    vector: vector.into(),
                    metadata,
                    similarity,
                }
            })
            .collect())
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<(Vec<f64>, Option<Metadata>)> {
        let vector = self.storage.load_vector(index)?;
        let metadata = self.storage.load_metadata(index);
        Some((vector.into(), metadata))
    }

    fn search(
        &self,
        query: &Vector,
        k: usize,
        similarity_threshold: Option<f64>,
    ) -> anyhow::Result<Vec<(usize, Vector)>> {
        let Some(start_vector) = self.storage.load_vector(0) else {
            return Ok(Vec::new());
        };

        let node_count = self.storage.len();
        let mut visited = vec![false; node_count];
        visited[0] = true;

        let mut candidates = BinaryHeap::from([Candidate {
            distance: query.l2_distance(&start_vector)?,
            node_id: 0,
        }]);

        let mut results = Vec::new();

        while let Some(current) = candidates.pop() {
            let Some(vector) = self.storage.load_vector(current.node_id) else {
                continue;
            };

            let dominated = similarity_threshold
                .map(|t| vector.cosine_similarity(query).map(|s| s >= t))
                .transpose()?
                .unwrap_or(true);

            if dominated && !query.equal(&vector) {
                results.push((current.node_id, vector, current.distance));
                if results.len() >= k {
                    break;
                }
            }

            candidates.extend(
                self.storage
                    .load_neighbors(current.node_id)
                    .into_iter()
                    .filter(|&id| id < node_count && !std::mem::replace(&mut visited[id], true))
                    .filter_map(|id| self.storage.load_vector(id).map(|v| (id, v)))
                    .map(|(id, v)| {
                        query.l2_distance(&v).map(|d| Candidate {
                            distance: d,
                            node_id: id,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }

        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().map(|(id, v, _)| (id, v)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db() -> HnswDB {
        HnswDB::new(64, 0.8)
    }

    #[test]
    fn test_database_insert_and_find() {
        let mut db = create_test_db();

        let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vec2 = vec![1.1, 2.1, 3.0, 4.0, 5.0];
        let vec3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let vec4 = vec![-1.0, -2.0, -3.0, -4.0, -5.0];

        db.insert(vec1.clone().into()).expect("Should insert");
        db.insert(vec2.into()).expect("Should insert");
        db.insert(vec3.into()).expect("Should insert");
        db.insert(vec4.clone().into()).expect("Should insert");

        let results = db
            .query(vec1.clone().into(), 5)
            .expect("Should get results");
        assert_eq!(results.len(), 2);

        let similarity = Vector::new(vec1)
            .cosine_similarity(&Vector::new(results[0].clone()))
            .expect("Bad Cosine");
        assert!(similarity > 0.8);

        let results = db
            .query(vec4.clone().into(), 5)
            .expect("Should get results");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_same_vector_produces_same_id() {
        let mut db = create_test_db();
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        db.insert(vec.clone().into()).unwrap();
        db.insert(vec.clone().into()).unwrap();
        db.insert(vec.into()).unwrap();

        assert_eq!(db.len(), 1);
    }
}
