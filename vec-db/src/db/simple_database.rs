use crate::clustering::kmeans::Kmeans;
use crate::math::vector::Vector;

pub struct SimpleDatabase {
    kmeans: Kmeans,
}

impl Default for SimpleDatabase {
    fn default() -> Self {
        Self::new(100)
    }
}

impl SimpleDatabase {
    pub fn new(vector_size: usize) -> SimpleDatabase {
        SimpleDatabase {
            kmeans: Kmeans::new(vector_size),
        }
    }

    pub fn insert(&mut self, vector: Vector) -> anyhow::Result<()> {
        let len = vector.len();
        self.kmeans.add_datapoint(vector);
        if self.kmeans.centroids().is_empty() {
            self.kmeans.add_centroid(Vector::rand(len));
        }

        let dataset_size = self.kmeans.dataset().len();
        let num_centroids = self.kmeans.centroids().len();

        // Fit based on data-to-centroid ratio, not arbitrary constants
        let points_per_centroid = dataset_size / num_centroids;
        let should_fit = dataset_size % (points_per_centroid * 2).max(10) == 0;

        if should_fit {
            let iterations = (num_centroids / 20).clamp(1, 5);
            self.kmeans.fit(iterations as u64)?;
        }

        Ok(())
    }

    pub fn query(&mut self, vector: Vector, n: usize) -> anyhow::Result<Vec<Vec<f64>>> {
        // 1. Find the closest centroid.
        // 2. Find the closest vector inside that group ?
        // ^ this might be good enough for v0.
        self.kmeans.find_closest_data_points(&vector, n)
    }

    pub fn centrodis(&mut self) -> Vec<Vec<f64>> {
        let mut raw = Vec::new();
        for i in self.kmeans.centroids().iter() {
            raw.push(i.raw().clone());
        }
        raw
    }
}

#[cfg(test)]
mod tests {

    use crate::{db::simple_database::SimpleDatabase, math::vector::Vector};

    #[test]
    pub fn test_empty_query() -> anyhow::Result<()> {
        let mut db = SimpleDatabase::new(100);
        let results = db.query(Vector::empty(100), 5)?;
        assert_eq!(results.len(), 0);
        Ok(())
    }

    #[test]
    pub fn test_centroid_growth() -> anyhow::Result<()> {
        let mut db = SimpleDatabase::new(64);
        let mut centroid_counts = Vec::new();

        for i in 1..=500 {
            let vector = generate_random_vector(64, (i / 50) as f64 * 20.0);
            db.insert(Vector::new(vector))?;

            if i % 50 == 0 {
                let count = db.kmeans.centroids().len();
                centroid_counts.push(count);
            }
        }

        assert!(centroid_counts.len() >= 5);
        assert!(centroid_counts[0] >= 2);
        assert!(centroid_counts.last().expect("items expected") > &centroid_counts[0]);
        Ok(())
    }

    #[test]
    pub fn test_adaptive_fitting_schedule() -> anyhow::Result<()> {
        let mut db = SimpleDatabase::new(32);
        let mut fit_counts = Vec::new();

        for i in 1..=200 {
            let vector = generate_random_vector(32, i as f64);
            let centroids_before = db.kmeans.centroids().len();

            db.insert(Vector::new(vector))?;

            let centroids_after = db.kmeans.centroids().len();
            if centroids_before != centroids_after {
                fit_counts.push(i);
            }
        }

        Ok(())
    }

    fn generate_random_vector(size: usize, center_offset: f64) -> Vec<f64> {
        (0..size)
            .map(|i| {
                // Create more variance with larger random components
                let pseudo_random1 =
                    ((i * 17 + center_offset as usize * 23) % 1000) as f64 / 1000.0;
                let pseudo_random2 =
                    ((i * 37 + center_offset as usize * 47) % 1000) as f64 / 1000.0;
                let pseudo_random3 =
                    ((i * 53 + center_offset as usize * 71) % 1000) as f64 / 1000.0;

                // Combine multiple random sources for more variance
                let base_variance = (pseudo_random1 - 0.5) * center_offset * 0.5;
                let cluster_variance = (pseudo_random2 - 0.5) * 10.0;
                let noise = (pseudo_random3 - 0.5) * 5.0;

                center_offset + base_variance + cluster_variance + noise
            })
            .collect()
    }
}
