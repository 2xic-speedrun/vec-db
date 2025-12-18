use criterion::{criterion_group, criterion_main, Criterion};
use vec_db::backends::lsh_hash::{LshDB, RocksDbBucket};
use vec_db::math::vector::VectorU8;

fn bench_lsh_rocksdb_vector_insert_f64(c: &mut Criterion) {
    c.bench_function("lsh_rocksdb_f64_insert_100", |b| {
        b.iter(|| {
            let mut db: LshDB<RocksDbBucket> =
                LshDB::new(64, 16, 128, 0.55).expect("Failed to init");
            for i in 0..100 {
                let vector: Vec<f64> = (0..128).map(|j| (i + j) as f64).collect();
                db.insert(vector).expect("Insert failed");
            }
        });
    });
}

fn bench_lsh_rocksdb_vector_insert_u8(c: &mut Criterion) {
    c.bench_function("lsh_rocksdb_u8_insert_100", |b| {
        b.iter(|| {
            let mut db: LshDB<RocksDbBucket> =
                LshDB::new(64, 16, 128, 0.55).expect("Failed to init");
            for i in 0..100 {
                let vector_u8: Vec<u8> = (0..128).map(|j| ((i + j) % 256) as u8).collect();
                let vector_f64: Vec<f64> = vector_u8.iter().map(|&x| x as f64 / 255.0).collect();
                db.insert(vector_f64).expect("Insert failed");
            }
        });
    });
}

fn bench_vector_quantization(c: &mut Criterion) {
    c.bench_function("f64_to_u8_quantization", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let f64_vector: Vec<f64> = (0..128).map(|j| (i + j) as f64 / 1000.0).collect();
                let _u8_vector: Vec<u8> = f64_vector
                    .iter()
                    .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect();
            }
        });
    });
}

fn bench_vector_u8_operations(c: &mut Criterion) {
    c.bench_function("u8_vector_distance", |b| {
        let vec1 = VectorU8::new((0..128).map(|i| (i % 256) as u8).collect());
        let vec2 = VectorU8::new((0..128).map(|i| ((i + 64) % 256) as u8).collect());

        b.iter(|| {
            let _ = vec1.l2_distance(&vec2).expect("Distance failed");
        });
    });
}

criterion_group!(
    benches,
    bench_lsh_rocksdb_vector_insert_f64,
    bench_lsh_rocksdb_vector_insert_u8,
    bench_vector_quantization,
    bench_vector_u8_operations
);
criterion_main!(benches);
