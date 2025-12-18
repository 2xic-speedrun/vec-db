use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub trait VectorElement:
    Clone + Copy + Send + Sync + 'static + std::fmt::Display + PartialEq
{
    fn l2_distance_squared(a: &[Self], b: &[Self]) -> f64;
    fn dot_product(a: &[Self], b: &[Self]) -> f64;
    fn magnitude_squared(slice: &[Self]) -> f64;
    fn add(a: Self, b: Self) -> Self;
    fn subtract(a: Self, b: Self) -> Self;
    fn mul_constant(value: Self, constant: f64) -> Self;
    fn to_f64(value: Self) -> f64;
    fn zero() -> Self;
}

impl VectorElement for f64 {
    fn l2_distance_squared(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }

    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn magnitude_squared(slice: &[f64]) -> f64 {
        slice.iter().map(|&x| x * x).sum()
    }

    fn add(a: f64, b: f64) -> f64 {
        a + b
    }
    fn subtract(a: f64, b: f64) -> f64 {
        a - b
    }
    fn mul_constant(value: f64, constant: f64) -> f64 {
        value * constant
    }
    fn to_f64(value: f64) -> f64 {
        value
    }
    fn zero() -> f64 {
        0.0
    }
}

impl VectorElement for f32 {
    fn l2_distance_squared(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = (x - y) as f64;
                diff * diff
            })
            .sum()
    }

    fn dot_product(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x * y) as f64).sum()
    }

    fn magnitude_squared(slice: &[f32]) -> f64 {
        slice.iter().map(|&x| (x * x) as f64).sum()
    }

    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
    fn subtract(a: f32, b: f32) -> f32 {
        a - b
    }
    fn mul_constant(value: f32, constant: f64) -> f32 {
        value * (constant as f32)
    }
    fn to_f64(value: f32) -> f64 {
        value as f64
    }
    fn zero() -> f32 {
        0.0
    }
}

impl VectorElement for u8 {
    fn l2_distance_squared(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = (x as i16 - y as i16) as f64;
                diff * diff
            })
            .sum()
    }

    fn dot_product(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum()
    }

    fn magnitude_squared(slice: &[u8]) -> f64 {
        slice.iter().map(|&x| (x as f64) * (x as f64)).sum()
    }

    fn add(a: u8, b: u8) -> u8 {
        a.saturating_add(b)
    }
    fn subtract(a: u8, b: u8) -> u8 {
        a.saturating_sub(b)
    }
    fn mul_constant(value: u8, constant: f64) -> u8 {
        ((value as f64) * constant).clamp(0.0, 255.0) as u8
    }
    fn to_f64(value: u8) -> f64 {
        value as f64
    }
    fn zero() -> u8 {
        0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T: VectorElement = f64> {
    vector: Vec<T>,
}

impl<T: VectorElement> Hash for Vector<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vector.len().hash(state);
        for &value in &self.vector {
            T::to_f64(value).to_bits().hash(state);
        }
    }
}

impl<T: VectorElement> Eq for Vector<T> {}

impl<T: VectorElement> Serialize for Vector<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.vector.serialize(serializer)
    }
}

impl<'de, T: VectorElement> Deserialize<'de> for Vector<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vector = Vec::<T>::deserialize(deserializer)?;
        Ok(Vector::new(vector))
    }
}

impl<T: VectorElement> Vector<T> {
    pub fn new(vec: Vec<T>) -> Vector<T> {
        Vector { vector: vec }
    }

    pub fn empty(size: usize) -> Vector<T> {
        Vector::new(vec![T::zero(); size])
    }

    pub fn rand(size: usize, rng: &Arc<Mutex<ChaCha8Rng>>) -> Vector<T>
    where
        T: From<f64>,
    {
        let mut rng_guard = rng.lock().unwrap();
        let vector: Vec<T> = (0..size)
            .map(|_| T::from(rng_guard.random::<f64>()))
            .collect();
        Vector::new(vector)
    }

    pub fn l2_distance(&self, other: &Vector<T>) -> Result<f64> {
        Ok(self.l2_distance_squared(other)?.sqrt())
    }

    pub fn l2_distance_squared(&self, other: &Vector<T>) -> Result<f64> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        Ok(T::l2_distance_squared(&self.vector, &other.vector))
    }

    pub fn norm(mut self) -> Self {
        let norm: f64 = T::magnitude_squared(&self.vector).sqrt();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x = T::mul_constant(*x, 1.0 / norm);
            }
        }
        self
    }

    pub fn l1_dot(&self, other: &Vector<T>) -> Result<f64> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }
        Ok(T::dot_product(&self.vector, &other.vector))
    }

    pub fn cosine_similarity(&self, b: &Vector<T>) -> Result<f64> {
        let dot = self.l1_dot(b)?;
        let norm_a: f64 = T::magnitude_squared(&self.vector).sqrt();
        let norm_b: f64 = T::magnitude_squared(&b.vector).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot / (norm_a * norm_b))
        }
    }

    pub fn subtract(&self, other: Vector<T>) -> Result<Vector<T>> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        let mut results = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            results.push(T::subtract(self.vector[i], other.vector[i]));
        }

        Ok(Vector { vector: results })
    }

    pub fn add(&self, other: &Vector<T>) -> Result<Vector<T>> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        let mut vec = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            vec.push(T::add(self.vector[i], other.vector[i]));
        }

        Ok(Vector { vector: vec })
    }

    pub fn add_inplace(&mut self, other: &Vector<T>) -> Result<()> {
        if self.len() != other.len() {
            bail!(
                "Vector sizes does not match, {} != {}",
                self.len(),
                other.len()
            );
        }

        for i in 0..self.len() {
            self.vector[i] = T::add(self.vector[i], other.vector[i]);
        }

        Ok(())
    }

    pub fn mul_constant(&self, constant: f64) -> Vector<T> {
        let vec: Vec<T> = self
            .vector
            .iter()
            .map(|&x| T::mul_constant(x, constant))
            .collect();
        Vector { vector: vec }
    }

    pub fn div_constant(&self, constant: f64) -> Vector<T> {
        self.mul_constant(1.0 / constant)
    }

    pub fn abs(&self) -> Vector<T>
    where
        T: VectorElement,
    {
        let vec: Vec<T> = self
            .vector
            .iter()
            .map(|&x| {
                if T::to_f64(x) < 0.0 {
                    T::mul_constant(x, -1.0)
                } else {
                    x
                }
            })
            .collect();
        Vector { vector: vec }
    }

    pub fn equal(&self, other: &Vector<T>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for (i, &item) in self.vector.iter().enumerate().take(self.len()) {
            if let Some(&other_value) = other.vector.get(i) {
                if item != other_value {
                    return false;
                }
            }
        }

        true
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.vector.get(index).copied()
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vector.len() == 0
    }

    pub fn sum_d1(&self) -> f64 {
        self.vector.iter().map(|&x| T::to_f64(x)).sum()
    }
}

impl<T: VectorElement> From<Vector<T>> for Vec<T> {
    fn from(v: Vector<T>) -> Vec<T> {
        v.vector
    }
}

impl<T: VectorElement> AsRef<[T]> for Vector<T> {
    fn as_ref(&self) -> &[T] {
        &self.vector
    }
}

impl<T: VectorElement> fmt::Display for Vector<T>
where
    T: fmt::Display,
{
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

impl<T: VectorElement> From<Vec<T>> for Vector<T> {
    fn from(value: Vec<T>) -> Self {
        Vector::new(value)
    }
}

pub type VectorF64 = Vector<f64>;
pub type VectorF32 = Vector<f32>;
pub type VectorU8 = Vector<u8>;

#[cfg(test)]
mod test {
    use crate::math::vector::Vector;

    #[test]
    fn test_cosine_similarity() {
        let vec1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let vec2 = Vector::new(vec![0.0, 1.0, 0.0]);
        let vec3 = Vector::new(vec![1.0, 1.0, 0.0]);
        let vec4 = Vector::new(vec![2.0, 0.0, 0.0]);

        assert!(((vec1.cosine_similarity(&vec2)).expect("Bad cosine") - 0.0).abs() < 1e-10);
        assert!(((vec1.cosine_similarity(&vec4)).expect("Bad cosine") - 1.0).abs() < 1e-10);
        assert!(
            ((vec1.cosine_similarity(&vec3)).expect("Bad cosine") - (1.0 / 2.0_f64.sqrt())).abs()
                < 1e-10
        );
    }
}
