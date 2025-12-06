use criterion::{criterion_group, criterion_main, Criterion};
use vec_db::backends::lsh_hash::{LshDB, RocksDbBucket};

fn bench_lsh_rocksdb_vector_insert(c: &mut Criterion) {
    c.bench_function("lsh_rocksdb_vector_insert_1000", |b| {
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

criterion_group!(benches, bench_lsh_rocksdb_vector_insert);
criterion_main!(benches);
