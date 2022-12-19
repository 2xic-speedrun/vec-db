use std::io::prelude::*;
use std::io::Cursor;
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

 use crate::vector::vector::Vector;

pub struct FileFormat<'a> {
    // n vector szie ? 
    writer:&'a mut Cursor<Vec<u8>>,
}

impl FileFormat<'_> {
    pub fn new(
        writer: &mut Cursor<Vec<u8>>
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

    pub fn write_header(&mut self, centroid: Vector) {
        // let len = self.writer.len();
        if self.writer.get_ref().len() == 0 {
            /*
            Header format
            - Centroid count
            - Centroid dim (vector dim)
            */
            self.writer.write(&[(centroid.len() as u8)]);
            self.writer.write(&[(1)]);
        } else {
            let centroids = self.get_centroids() + 1;
            self.writer.seek(std::io::SeekFrom::Start(1));
            self.writer.write(&[centroids]);
        }
    }

    pub fn add_centroid(&mut self, vector: Vector){
        self.write_header(vector);
        // Write random data.
        /*
        for i in 0..10 {
            self.writer.write(&[i]);
        }*/
    }

    pub fn len(&mut self) -> usize{
        return self.writer.get_ref().len();
    }
}

