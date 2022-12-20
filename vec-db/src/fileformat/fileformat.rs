use std::io::prelude::*;
use std::io::Cursor;
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

pub struct FileFormat<'a> {
    // n vector szie ? 
    writer:&'a mut Cursor<Vec<u8>>,
}

const VECTOR_LEN: usize = 3;

impl FileFormat<'_> {
    pub fn new(
        writer: &mut Cursor<Vec<u8>>,
    ) -> FileFormat {
        return FileFormat {
            writer: writer,
        };
    }

    pub fn read_header(self){
        // pass
    }

    pub fn get_dimensions(&mut self) -> u8{
        self.writer.seek(std::io::SeekFrom::Start(0));
        let mut buffer = [0; 1];
        self.writer.read_exact(&mut buffer);
        return buffer[0];
    }

    pub fn get_centroids(&mut self) -> u8{
        self.writer.seek(std::io::SeekFrom::Start(1));
        let mut buffer = [0; 1];
        self.writer.read_exact(&mut buffer);
        return buffer[0];
    }

    pub fn create_or_update_header(&mut self, centroid: &Vector) {
        if self.writer.get_ref().len() == 0 {
            self.writer.write(&[(centroid.len() as u8)]);
            self.writer.write(&[(1)]);
        } else {
            let centroids = self.get_centroids() + 1;
            self.writer.seek(std::io::SeekFrom::Start(1));
            self.writer.write(&[centroids]);
        }
    }

    pub fn add_vector(&mut self, vector: &Vector){
        assert_eq!(vector.len(), VECTOR_LEN);
        self.create_or_update_header(&vector);
        // TODO: Should probably also 
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
        let forward: u64 = ((BYTE_SIZE * VECTOR_LEN) * index ) as u64;
        println!("{}", forward);
        self.writer.seek(std::io::SeekFrom::Start(1 + forward));

        let mut vector:Vec<f64> = Vec::new();
        for i in 0..VECTOR_LEN {
            let mut buffer = [0; (BYTE_SIZE)];
            // Does this also seek ?
            self.writer.read_exact(&mut buffer);
            vector.push(f64::from_be_bytes(buffer));
        }
        return Vector::new(vector);
    }

    pub fn len(&mut self) -> usize{
        return self.writer.get_ref().len();
    }
}
