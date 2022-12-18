
pub mod vector;

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

        let subtracted_vector = a.l2_distance(b).unwrap();
        assert_eq!(subtracted_vector, 0.0);

        let subtracted_vector = a.l2_distance(c).unwrap();
        assert_eq!(subtracted_vector, (2.0_f64).sqrt());
    }
}
