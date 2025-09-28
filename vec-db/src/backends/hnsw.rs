use std::collections::BinaryHeap;

use anyhow::Ok;

use crate::math::vector::Vector;

#[derive(Debug, Clone)]
struct Node {
    vector: Vector,
    neighbors: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    distance: f64,
    node_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct VectorCandidate<'a> {
    candidate: Candidate,
    vector: &'a Vector,
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

pub struct HnswDB {
    nodes: Vec<Node>,
    max_connections: usize,
    similarity_threshold: f64,
}

impl HnswDB {
    pub fn new(max_connections: usize, similarity_threshold: f64) -> Self {
        Self {
            nodes: Vec::new(),
            max_connections,
            similarity_threshold,
        }
    }

    pub fn insert(&mut self, vector: Vector) -> anyhow::Result<()> {
        let node_id = self.nodes.len();

        let closest_results = self.search(&vector, self.max_connections * 2, None)?;
        let candidates: Vec<Candidate> = closest_results
            .iter()
            .map(|f| f.candidate.clone())
            .collect();
        let neighbors: Vec<usize> = candidates
            .into_iter()
            .take(self.max_connections)
            .map(|c| c.node_id)
            .collect();

        let new_node = Node { vector, neighbors };
        self.nodes.push(new_node);

        // TODO: pruning?
        // This also isn't fully correct with the algorithm
        let num_to_connect = self.nodes.len().saturating_sub(1).min(self.max_connections);
        for i in 0..num_to_connect {
            self.nodes[node_id].neighbors.push(i);
            self.nodes[i].neighbors.push(node_id);
        }

        Ok(())
    }

    pub fn query(&self, query: Vector, k: usize) -> anyhow::Result<Vec<Vec<f64>>> {
        let results = self.search(&query, k, Some(self.similarity_threshold))?;
        let vector_results = results.iter().map(|f| f.vector.clone().as_vec()).collect();

        Ok(vector_results)
    }

    fn search(
        &self,
        query: &Vector,
        k: usize,
        similarity_threshold: Option<f64>,
    ) -> anyhow::Result<Vec<VectorCandidate>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let start_node = 0;
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates = BinaryHeap::new();

        let start_dist = query.l2_distance(&self.nodes[start_node].vector)?;
        candidates.push(Candidate {
            distance: start_dist,
            node_id: start_node,
        });
        visited[start_node] = true;

        let mut results = Vec::new();

        while let Some(current) = candidates.pop() {
            let candidate = &self.nodes[current.node_id].vector;
            let is_okay = match similarity_threshold {
                Some(value) => value <= candidate.cosine_similarity(query)?,
                None => true,
            };
            if is_okay && !query.equal(candidate) {
                results.push(VectorCandidate {
                    candidate: Candidate {
                        node_id: current.node_id,
                        distance: current.distance,
                    },
                    vector: candidate,
                });
            }

            if results.len() >= k {
                break;
            }

            for &neighbor_id in &self.nodes[current.node_id].neighbors {
                if !visited[neighbor_id] {
                    visited[neighbor_id] = true;
                    let dist = query.l2_distance(&self.nodes[neighbor_id].vector)?;
                    candidates.push(Candidate {
                        distance: dist,
                        node_id: neighbor_id,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            a.candidate
                .distance
                .partial_cmp(&b.candidate.distance)
                .unwrap()
        });

        Ok(results)
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
}
