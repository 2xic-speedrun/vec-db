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

    pub fn insert(&mut self, vector: Vector) {
        /*
         * Save the vector into k-means ?
         * We provide 100 vector sampling -> for each centroid
         * K-means only add new cluster at 100 nodes in a centroid ?
         * -> Means we split the rest into a separate file
         */
        self.kmeans.add_datapoint(vector);
        if self.kmeans.centroids().is_empty() {
            self.kmeans
                .add_centroid(Vector::new(self.kmeans.get_random_vec()));
        }

        // Why is there no more centroids added ?
        self.kmeans.fit(30);
    }

    pub fn query(&mut self, vector: Vector, n: usize) -> Vec<Vec<f64>> {
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
    pub fn test_empty_query() {
        let mut db = SimpleDatabase::new(100);
        let results = db.query(Vector::empty(100), 5);
        assert_eq!(results.len(), 0);
    }
}
