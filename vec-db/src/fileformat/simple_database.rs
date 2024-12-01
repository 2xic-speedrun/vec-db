use crate::vector::vector::Vector;
use crate::kmeans::kmeans::Kmeans;

pub struct SimpleDatabase {
    kmeans: Kmeans,
}

impl SimpleDatabase {
    pub fn new(
    ) -> SimpleDatabase {
        return SimpleDatabase {
            kmeans: Kmeans::new(100),
        };
    }

    pub fn insert(&mut self, vector: Vector) {
        /*
        * Save the vector into k-means ?
        * We provide 100 vector sampling -> for each centroid
        * K-means only add new cluster at 100 nodes in a centroid ? 
        * -> Means we split the rest into a separate file
        */
        self.kmeans.add_datapoint(vector);
        if self.kmeans.centroids().len() == 0 {
            // Push random
            self.kmeans.add_centroid(
                Vector::new(self.kmeans.get_random_vec())
            );
        }

        // Why is there no more centroids added ? 
        self.kmeans.fit(30);
    }

    pub fn query(&mut self, vector: Vector, n:usize) -> Vec<Vec<f64>> {
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