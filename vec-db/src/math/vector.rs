use std::cell::Cell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, PartialEq)]
pub struct Vector {
    vector: Vec<f64>,
}

impl Hash for Vector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the length first
        self.vector.len().hash(state);

        // Hash each f64 as bytes
        for &value in &self.vector {
            value.to_bits().hash(state); // Convert f64 to u64 bits
        }
    }
}

impl Eq for Vector {}

thread_local! {
    static COUNTER: Cell<u64> = const { Cell::new(0) };
}

// TODO: support binary quantization
impl Vector {
    pub fn new(vec: Vec<f64>) -> Vector {
        Vector { vector: vec }
    }

    pub fn empty(size: usize) -> Vector {
        Vector::new(vec![0.0; size])
    }

    pub fn rand(size: usize, rng: &Arc<Mutex<ChaCha8Rng>>) -> Vector {
        let mut zero_vec = Vec::new();
        for _ in 0..size {
            let random_val = {
                let mut rng_guard = rng.lock().unwrap();
                rng_guard.random::<f64>()
            };
            zero_vec.push(random_val);
        }
        Vector::new(zero_vec)
    }

    pub fn l2_distance(&self, other: &Vector) -> Result<f64> {
        Ok(self.l2_distance_squared(other)?.sqrt())
    }

    pub fn l2_distance_squared(&self, other: &Vector) -> Result<f64> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        let mut distance: f64 = 0.0;
        for i in 0..self.len() {
            let diff = self.vector[i] - other.vector[i];
            distance += diff * diff;
        }

        Ok(distance)
    }

    pub fn norm(&mut self) -> Self {
        let norm: f64 = self.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
        self.clone()
    }

    pub fn l1_dot(&self, other: &Vector) -> Result<f64> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }
        let results = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(x, y)| x * y)
            .sum();
        Ok(results)
    }

    pub fn cosine_similarity(&self, b: &Vector) -> Result<f64> {
        let dot = self.l1_dot(b)?;
        let norm_a: f64 = self.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.vector.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot / (norm_a * norm_b))
        }
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

        let mut vec = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            vec.push(self.vector[i] + other.vector[i]);
        }

        Ok(Vector { vector: vec })
    }

    pub fn add_inplace(&mut self, other: &Vector) -> Result<()> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        for i in 0..self.len() {
            self.vector[i] += other.vector[i];
        }

        Ok(())
    }

    pub fn mul_constant(&self, constant: f64) -> Vector {
        let vec: Vec<f64> = self.vector.iter().map(|&x| x * constant).collect();
        Vector { vector: vec }
    }

    pub fn div_constant(&self, constant: f64) -> Vector {
        self.mul_constant(1.0 / constant)
    }

    pub fn abs(&self) -> Vector {
        let vec: Vec<f64> = self.vector.iter().map(|&x| x.abs()).collect();
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

    pub fn as_vec(self) -> Vec<f64> {
        self.vector.clone()
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
