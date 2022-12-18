
pub struct Vector {
    vector: Vec<f64>,
}

impl Vector {
    pub fn new(vec: Vec<f64>) -> Vector {
        return Vector{
            vector: vec
        };
    }

    pub fn Copy(mut self) -> Vector {
        let mut cloned_vec: Vec<f64> = Vec::new();
        for i in self.vector.iter_mut(){
            cloned_vec.push(*i);
        }
        return Vector{
            vector: self.vector.clone(),
        }
    }

    pub fn l2_distance(&self, other: &Vector) -> Option<f64> {
        if self.len() != other.len() {
            return None;
        }

        let mut distance: f64 = 0.0;
        for i in 0..self.len() {
            let other_value = other.get(i);
            if let Some(other_value) = other_value{
                distance += (self.vector[i] - other_value).powf(2.0);
            }
        }

        return Some(distance.sqrt());        
    }

    pub fn subtract(&self, other: Vector) -> Option<Vector> {
        if self.len() != other.len() {
            return None;
        }

        let mut vec = self.vector.clone();

        for i in 0..self.len() {
            let other_value = other.get(i);
            if let Some(other_value) = other_value{
                vec[i] = vec[i] - other_value;
            }
        }

        return Some(Vector {
            vector: vec,
        });
    }

    pub fn add(&self, other: &Vector) -> Option<Vector> {
        if self.len() != other.len() {
            return None;
        }

        let mut vec = self.vector.clone();

        for i in 0..self.len() {
            let other_value = other.get(i);
            if let Some(other_value) = other_value{
                vec[i] = vec[i] + other_value;
            }
        }

        return Some(Vector {
            vector: vec,
        });
    }

    pub fn mul_constant(&self, constant: f64) -> Vector {
        let mut vec = self.vector.clone();

        for i in 0..self.len() {
            vec[i] = vec[i] * constant;
        }

        return Vector {
            vector: vec,
        };
    }

    pub fn equal(&self, other: Vector) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let vec = self.vector.clone();
        for i in 0..self.len() {
            let other_value = other.get(i);
            if let Some(other_value) = other_value{
                if vec[i] != other_value {
                    return false;
                }
            }
        }

        return true;
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        let value = self.vector.get(index);
        if let Some(value) = value {
            return Some(*value);
        }
        return None;
    }

    pub fn len(&self) -> usize {
        return self.vector.len();
    }

    pub fn println(&self) {
//        println!("We out here");
        for (index, i) in self.vector.iter().enumerate() {
            if index > 0 {
                print!(", ");
            }
            print!("{}", i);
        }
        print!("\n");
    }
}

