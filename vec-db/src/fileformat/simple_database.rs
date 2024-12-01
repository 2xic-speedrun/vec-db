use crate::fileformat::fileformat::FileFormat;
use crate::vector::vector::Vector;
use crate::kmeans::kmeans::Kmeans;
use std::fs::File;
use std::collections::HashMap;
use std::fs::OpenOptions;

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
        self.kmeans.add(vector);
        if self.kmeans.centroids().len() == 0{
            // Push random
            self.kmeans.add_centroid(
                Vector::new(self.kmeans.get_zero_vec())
            );
        }

        // Why is there no more centroids added ? 
        self.kmeans.fit(10);
    }

    pub fn query(&mut self, vector: Vector) -> Vec<Vec<f64>> {
        // 1. Find the closest centroid.
        // 2. Find the closest vector inside that group ? 
        // ^ this might be good enough for v0.
        self.kmeans.find_closest_data_points(&vector)
    }

    pub fn load(&mut self) {
        // load the files (with sampling)
        // -> 
        for i in -1..4 {
            let result = &self.get_file_name(i);
            let b = std::path::Path::new(result).exists();
            if b {
                let mut file = self.get_file_cluster(i);
                // load the file, and load into k-means
                let mut file_format = FileFormat::new(&mut file, 100);
                for vector in 0..file_format.get_centroids() {
                    if i == -1 {
                        self.kmeans.add_centroid(file_format.read_vector(vector as usize));
                    } else {
                        self.kmeans.add(file_format.read_vector(vector as usize));
                    }
                }
            }
        }
    }

    pub fn dump(&mut self) {
        let clustered_data_pints = self.get_clustered_data_point();
        self.dump_by_hashmap_file(clustered_data_pints);

        let mut file = self.get_file_cluster(-1);
        let mut file_format = FileFormat::new(&mut file, 100);
        for i in self.kmeans.centroids().iter() {
            file_format.add_vector(i);
        }
    }

    pub fn dump_by_hashmap_file(&self, results:  HashMap<i8, Vec<&Vector>> ) {
        for (key, values) in results.into_iter() {
            let mut file = self.get_file_cluster(key);
            let mut file_format = FileFormat::new(&mut file, 100);
            for i in values.iter() {
                file_format.add_vector(i);
            }
        }
    }

    fn get_clustered_data_point(&self) -> HashMap<i8, Vec<&Vector>> {
        let mut clustered_data_pints:HashMap<i8, Vec<&Vector>> = HashMap::new();

        let res = self.kmeans.results();
        for i in res.iter() {
            let class = i.1;
            let vector = i.0;

            match clustered_data_pints.get_mut(&class) {
                Some(value) => {
                    value.push(vector);
                },
                None => {
                    let mut vec = Vec::new();
                    vec.push(vector);
                    clustered_data_pints.insert(
                        class,
                        vec
                    );
                }
            };
        }
        return clustered_data_pints;
    }

    fn get_file_cluster(&self, index: i8) -> File {
        let result = &self.get_file_name(index);
        let b = std::path::Path::new(result).exists();
        if b && index != -1 {
            let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true).open(result).unwrap();
            return file;
        } else {
            let file = File::create(result).unwrap();
            return file;
        }
    }

    fn get_file_name(&self, index: i8) -> String {
        let s:String = (index).to_string();
        let result = ["kmeans_centroids_", &s, ".raw"].join("");
        return result;
    }
}