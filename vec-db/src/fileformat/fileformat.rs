use std::io::prelude::*;
use std::mem::size_of;
/**
 * 
 * Header section
 * -> Metadata (vector dimension size, centroid counts)
 * -> 
 * Data section
 *  -> 
 *  ->
 * Centroid section
 *  -> store the plain vector 
 *  -> 
 * -> 
 * Connections
 *  -> Vector idx -> Centroid idx ? 
 *  -> Or 
 * I think I might have to do some more reading before doing this on filesystem effective 
 *  filesystem layouts
 */

 /**
  * So I think in the first version we make a super simple version
    - Centroids are stored in a single file (parent file)
    - Vectors connected to a centroids stored in seperate files (child files)
  */

use crate::vector::vector::Vector;
use std::io::SeekFrom;

pub struct FileFormat<'a, W> {
    // n vector szie ? 
    writer:&'a mut W,
    vector_len: usize,
}

//const VECTOR_LEN: usize = 3;

impl<W:Write + Read+ Seek> FileFormat<'_, W> {
    pub fn new(
        writer: &mut W,
        vector_len: usize
    ) -> FileFormat<W> {
        return FileFormat {
            writer: writer,
            vector_len,
        };
    }

    pub fn read_header(self){
        // pass
    }

    pub fn get_dimensions(&mut self) -> u8{
        let _ = self.writer.seek(std::io::SeekFrom::Start(0));
        let mut buffer = [0; 1];
        let _ = self.writer.read_exact(&mut buffer);
        return buffer[0];
    }

    pub fn get_centroids(&mut self) -> u8{
        let _ = self.writer.seek(std::io::SeekFrom::Start(1));
        let mut buffer = [0; 1];
        let _ = self.writer.read_exact(&mut buffer);
        return buffer[0];
    }

    pub fn create_or_update_header(&mut self, centroid: &Vector) {
       if self.len() == 0 {
            let _ = self.writer.write(&[(centroid.len() as u8)]);
            let _ = self.writer.write(&[(1)]);
        } else {
            let centroids = self.get_centroids() + 1;
            let _ = self.writer.seek(std::io::SeekFrom::Start(1));
            let _ = self.writer.write(&[centroids]);
        }
    }

    pub fn add_vector(&mut self, vector: &Vector){
        assert_eq!(vector.len(), self.vector_len);
        self.create_or_update_header(&vector);
        self.len();
        let size = vector.len();
        for i in 0..size {
            let value = vector.get(i).unwrap();
            let byte_value = value.to_be_bytes();
            for b in byte_value{
                self.writer.write(&[
                    b
                ]).unwrap();
            }
        }
    }

    pub fn read_vector(&mut self, index: usize) -> Vector{
        const BYTE_SIZE: usize = size_of::<f64>();
        let forward: u64 = ((BYTE_SIZE * self.vector_len) * index ) as u64;
        let _ = self.writer.seek(std::io::SeekFrom::Start(1 + forward));

        let mut vector:Vec<f64> = Vec::new();
        for _ in 0..self.vector_len {
            let mut buffer = [0; (BYTE_SIZE)];
            // Does this also seek ?
            let _ = self.writer.read_exact(&mut buffer);
            vector.push(f64::from_be_bytes(buffer));
        }
        return Vector::new(vector);
    }

    pub fn len(&mut self) -> u64 {
        let size = self.writer.seek(SeekFrom::End(0));
        return size.unwrap();
    }
}
