use std::fmt;

use anyhow::{Result, bail};
use rand::Rng;

#[derive(Clone)]
pub struct Vector {
    vector: Vec<f64>,
}

impl Vector {
    pub fn new(vec: Vec<f64>) -> Vector {
        Vector { vector: vec }
    }

    pub fn empty(size: usize) -> Vector {
        let mut vec: Vec<f64> = Vec::with_capacity(size);
        for idx in 0..size {
            vec.insert(idx, 0.0);
        }
        Vector::new(vec)
    }

    pub fn rand(size: usize) -> Vector {
        let mut rng = rand::thread_rng();
        let mut zero_vec = Vec::new();
        for _ in 0..size {
            zero_vec.push(rng.gen());
        }
        Vector::new(zero_vec)
    }

    pub fn l2_distance(&self, other: &Vector) -> Result<f64> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        let mut distance: f64 = 0.0;
        for i in 0..self.len() {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                distance += (self.vector[i] - other_value).powf(2.0);
            }
        }

        Ok(distance.sqrt())
    }

    pub fn subtract(&self, other: Vector) -> Result<Vector> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        let mut vec = self.vector.clone();
        for (i, _item) in vec.clone().iter().enumerate().take(self.len()) {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                vec[i] -= other_value;
            }
        }

        Ok(Vector { vector: vec })
    }

    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        let mut vec = self.vector.clone();

        for (i, _item) in vec.clone().iter().enumerate().take(self.len()) {
            let other_value = other.get(i);
            if let Some(other_value) = other_value {
                vec[i] += other_value;
            }
        }

        Ok(Vector { vector: vec })
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

    pub fn equal(&self, other: &Vector) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for (i, item) in self.vector.iter().enumerate().take(self.len()) {
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
            print!("{i}");
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
                write!(f, ", ")?;
            }
            write!(f, "{i}")?;
        }
        write!(f, "]")
    }
}
