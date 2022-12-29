use crate::fileformat::fileformat::FileFormat;
use crate::vector::vector::Vector;
use std::io::prelude::*;

pub struct Database<'a, W> {
    centroids: &'a mut FileFormat<'a, W>,
    vectors: Vec<&'a mut FileFormat<'a, W>>
}

impl<W:Write + Read+ Seek> Database<'_, W> {
    pub fn new<'a>(
        centroids: &'a mut FileFormat<'a, W>,
        vectors: Vec<&'a mut FileFormat<'a, W>>
    ) -> Database<'a, W> {
        return Database {
            centroids: centroids,
            vectors: vectors,
        };
    }

    /**
     * TODO:
     * Maybe it's easier for this class just to return the index, 
     * and have higher level logic deal with loading the vectors into memory.
     * 
     */
    pub fn query(
        &mut self,
        query: &Vector
    ) -> Vec<Vector> {
        let best_centroid_index = self.get_closest_centroid(query);
        return self.get_n_closest_centroids(
            query,
            best_centroid_index,
        );
    }   

    
    fn get_n_closest_centroids(&mut self, query: &Vector, best_centroid_index: usize) -> Vec<Vector> {
        let file_format = self.vectors.get_mut(best_centroid_index).unwrap();

        let mut vector_index_distance: Vec<(u8, f64)> = Vec::new(); 
        let mut results: Vec<Vector> = Vec::new();
        let centroid_size = self.centroids.get_centroids() - 1;

        for i in 0..centroid_size {
            let distance = query.l2_distance(&self.centroids.read_vector(i.into())).unwrap();
            
            vector_index_distance.push((i, distance));
        }
        vector_index_distance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for i in 1..5 {
            if vector_index_distance.len() <= i {
                break;
            }
            let index = vector_index_distance.get(i).unwrap().0;
            results.push(file_format.read_vector(index.into()))
        }

        return results;
    }

    fn get_closest_centroid(&mut self, query: &Vector) -> usize {
        let centroid_size = self.centroids.get_centroids();
        let mut best_distance = f64::INFINITY;
        let mut best_index = 0;
        let size = centroid_size - 1;
        for i in 0..size {
            let distance = query.l2_distance(&self.centroids.read_vector(i.into())).unwrap();

            if distance < best_distance {
                best_index = i;
                best_distance = distance;
            }
        }

        return best_index.into();
    }
}
