use std::fmt;

#[derive(Clone)]
pub struct Vector {
    vector: Vec<f64>,
}

impl Vector {
    pub fn new(vec: Vec<f64>) -> Vector {
        Vector { vector: vec }
    }

    pub fn l2_distance(&self, other: &Vector) -> Option<f64> {
        if self.len() != other.len() {
            panic!("Wrong size");
        }

        let mut distance: f64 = 0.0;
        for i in 0..self.len() {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                distance += (self.vector[i] - other_value).powf(2.0);
            }
        }

        Some(distance.sqrt())
    }

    pub fn subtract(&self, other: Vector) -> Option<Vector> {
        if self.len() != other.len() {
            panic!("Wrong size");
        }

        let mut vec = self.vector.clone();
        for (i, _item) in vec.clone().iter().enumerate().take(self.len()) {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                vec[i] -= other_value;
            }
        }

        Some(Vector { vector: vec })
    }

    pub fn add(&self, other: &Vector) -> Option<Vector> {
        if self.len() != other.len() {
            panic!("Wrong size");
        }

        let mut vec = self.vector.clone();

        for (i, _item) in vec.clone().iter().enumerate().take(self.len()) {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                vec[i] += other_value;
            }
        }

        Some(Vector { vector: vec })
    }

    pub fn mul_constant(&self, constant: f64) -> Vector {
        let mut vec = self.vector.clone();

        for (i, _item) in vec.clone().iter().enumerate().take(self.len()) {
            vec[i] *= constant;
        }

        Vector { vector: vec }
    }

    pub fn div_constant(&self, constant: f64) -> Vector {
        self.mul_constant(1.0 / constant)
    }

    pub fn abs(&self) -> Vector {
        let mut vec = self.vector.clone();

        for (i, item) in vec.clone().iter().enumerate().take(self.len()) {
            vec[i] = item.abs()
        }

        Vector { vector: vec }
    }

    pub fn equal(&self, other: Vector) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let vec = self.vector.clone();
        for (i, item) in vec.iter().enumerate().take(self.len()) {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                if *item != other_value {
                    return false;
                }
            }
        }

        true
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        let value = self.vector.get(index);
        if let Some(value) = value {
            return Some(*value);
        }
        None
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vector.len() == 0
    }

    pub fn raw(&self) -> &Vec<f64> {
        &self.vector
    }

    pub fn println(&self) {
        for (index, i) in self.vector.iter().enumerate() {
            if index > 0 {
                print!(", ");
            }
            print!("{}", i);
        }
        println!();
    }

    pub fn sum_d1(&self) -> f64 {
        let mut value = 0.0;
        for i in self.vector.iter() {
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
