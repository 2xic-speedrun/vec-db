use std::fmt;

#[derive(Clone)]
pub struct Vector {
    vector: Vec<f64>,
}

impl Vector {
    pub fn new(vec: Vec<f64>) -> Vector {
        return Vector{
            vector: vec
        };
    }

    pub fn l2_distance(&self, other: &Vector) -> Option<f64> {
        if self.len() != other.len() {
            panic!("Wrong size");
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
            panic!("Wrong size");
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
            panic!("Wrong size");
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

    pub fn div_constant(&self, constant: f64) -> Vector {
        return self.mul_constant(1.0 / constant);
    }

    pub fn abs(&self) -> Vector {
        let mut vec = self.vector.clone();

        for i in 0..self.len() {
            vec[i] = vec[i].abs()
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

    pub fn raw(&self) -> &Vec<f64> {
        return &self.vector;
    }

    pub fn println(&self) {
        for (index, i) in self.vector.iter().enumerate() {
            if index > 0 {
                print!(", ");
            }
            print!("{}", i);
        }
        print!("\n");
    }

    pub fn sum_d1(&self) -> f64 {
        let mut value = 0.0;
        for (_, i) in self.vector.iter().enumerate() {
            value += i;
        }
        value
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = write!(f, "[");
        for (index, i) in self.vector.iter().enumerate() {
            if index > 0 {
                let _ = write!(f, ", ");
            }
            let _ = write!(f, "{}", i);
        }
        write!(f, "]")
    }
}

