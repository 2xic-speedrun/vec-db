pub mod bindings;
pub mod db;
pub mod clustering;
pub mod math;

#[cfg(test)]
mod tests {
    #[test]
    fn it_should_give_distance_between_vectors() {
        use crate::math::vector::Vector;

        let vec_a: Vec<f64> = vec![1.0, 1.0];
        let vec_b: Vec<f64> = vec![1.0, 1.0];
        let vec_c: Vec<f64> = vec![0.0, 0.0];

        let a = Vector::new(vec_a);
        let b = Vector::new(vec_b);
        let c = Vector::new(vec_c);

        let subtracted_vector = a.subtract(b).unwrap();
        assert!(subtracted_vector.equal(c));
    }

    #[test]
    fn it_should_report_l2_distance() {
        use crate::math::vector::Vector;

        let vec_a: Vec<f64> = vec![1.0, 1.0];
        let vec_b: Vec<f64> = vec![1.0, 1.0];
        let vec_c: Vec<f64> = vec![0.0, 0.0];

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
        use crate::clustering::kmeans::Kmeans;
        use crate::math::vector::Vector;

        let vec_a: Vec<f64> = vec![1.0, 1.0];
        let a = Vector::new(vec_a);

        let mut kmeans = Kmeans::new(2);
        kmeans.add_datapoint(a.mul_constant(-500.0));
        kmeans.add_datapoint(a.mul_constant(-440.0));

        kmeans.add_datapoint(a.mul_constant(150.0));

        let centroid_a: Vec<f64> = vec![1.0, 1.0];
        let base_centroid = Vector::new(centroid_a);
        /*
            TODO: Clean the reference borrowing here
        */
        kmeans.add_centroid(base_centroid.mul_constant(-300.0));
        kmeans.add_centroid(base_centroid.mul_constant(500.0));

        kmeans.fit(2);

        let centroids = kmeans.centroids();
        assert!(!centroids[0].equal(base_centroid.mul_constant(-3.0)));
        assert!(!centroids[1].equal(base_centroid.mul_constant(5.0)));
    }
}

use pyo3::prelude::*;

#[pymodule]
fn libvec_db(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    use crate::bindings::database::PyDatabase;
    use crate::bindings::python::PyKmeans;
    use crate::bindings::python::PyVector;

    m.add_class::<PyVector>()?;
    m.add_class::<PyKmeans>()?;
    m.add_class::<PyDatabase>()?;
    Ok(())
}
