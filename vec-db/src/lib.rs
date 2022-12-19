pub mod vector;
pub mod kmeans;
pub mod fileformat;

#[cfg(test)]
mod tests {
    #[test]
    fn it_should_give_distance_between_vectors() {
        use crate::vector::vector::Vector;
        
        let mut vec_a:Vec<f64> = Vec::new();
        vec_a.push(1.0);
        vec_a.push(1.0);

        let mut vec_b:Vec<f64> = Vec::new();
        vec_b.push(1.0);
        vec_b.push(1.0);

        let mut vec_c:Vec<f64> = Vec::new();
        vec_c.push(0.0);
        vec_c.push(0.0);
        let a = Vector::new(vec_a);
        let b = Vector::new(vec_b);
        let c = Vector::new(vec_c);

        let subtracted_vector = a.subtract(b).unwrap();
        assert_eq!(subtracted_vector.equal(c), true);
    }

    #[test]
    fn it_should_report_l2_distance() {
        use crate::vector::vector::Vector;
        
        let mut vec_a:Vec<f64> = Vec::new();
        vec_a.push(1.0);
        vec_a.push(1.0);

        let mut vec_b:Vec<f64> = Vec::new();
        vec_b.push(1.0);
        vec_b.push(1.0);

        let mut vec_c:Vec<f64> = Vec::new();
        vec_c.push(0.0);
        vec_c.push(0.0);
        let a = Vector::new(vec_a);
        let b = Vector::new(vec_b);
        let c = Vector::new(vec_c);

        let subtracted_vector = a.l2_distance(&b).unwrap();
        assert_eq!(subtracted_vector, 0.0);

        let subtracted_vector = a.l2_distance(&c).unwrap();
        assert_eq!(subtracted_vector, (2.0_f64).sqrt());
    }

    #[test]
    fn it_should_be_able_to_fit_kmeans() {
        use crate::kmeans::kmeans::Kmeans;
        use crate::vector::vector::Vector;
        
        let mut vec_a:Vec<f64> = Vec::new();
        vec_a.push(1.0);
        vec_a.push(1.0);
        let a = Vector::new(vec_a);

        let mut kmeans = Kmeans::new(3);
        kmeans.add(a.mul_constant(-500.0));
        kmeans.add(a.mul_constant(-440.0));

        kmeans.add(a.mul_constant(150.0));

        let mut centroid_a:Vec<f64> = Vec::new();
        centroid_a.push(1.0);
        centroid_a.push(1.0);

        let mut base_centroid = Vector::new(centroid_a);
        /*
            TODO: Clean the reference borrowing here
        */
        kmeans.add_centroid(base_centroid.mul_constant(-300.0));
        kmeans.add_centroid(base_centroid.mul_constant(500.0));

        kmeans.fit(2);
        
        let centroids = kmeans.centroids();
        assert_eq!(centroids[0].equal(base_centroid.mul_constant(-3.0)), false);
        assert_eq!(centroids[1].equal(base_centroid.mul_constant(5.0)), false);
    }


    #[test]
    fn it_should_have_a_functionally_file_format() {
        use crate::fileformat::fileformat::FileFormat;
        use std::fs::File;
        use std::io::prelude::*;
        use std::io::Cursor;
        use crate::vector::vector::Vector;
        
        // Fake file!
        let mut buff: Cursor<Vec<u8>> = Cursor::new(vec![]);
        
        let mut file_format = FileFormat::new(&mut buff);


        let mut centroid_a:Vec<f64> = Vec::new();
        centroid_a.push(1.0);
        centroid_a.push(1.0);
        centroid_a.push(1.0);

        let mut vec_a = Vector::new(centroid_a);

        file_format.add_centroid(vec_a);
        
        assert_eq!(file_format.get_dimensions(), 3);
        assert_eq!(file_format.get_centroids(), 1);

        let mut centroid_a:Vec<f64> = Vec::new();
        centroid_a.push(1.0);
        centroid_a.push(1.0);
        centroid_a.push(1.0);
        let mut vec_b = Vector::new(centroid_a);

        file_format.add_centroid(vec_b);
        assert_eq!(file_format.get_dimensions(), 3);
        assert_eq!(file_format.get_centroids(), 2);

        assert_eq!(file_format.len(), 2);
    }
}
