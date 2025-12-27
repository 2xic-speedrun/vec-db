use rand::Rng;
use rocksdb::{BlockBasedOptions, DBCompressionType, Options, WriteBatch, DB};

pub struct RocksDB {
    db: DB,
}

impl RocksDB {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_write_buffer_size(64 * 1024 * 1024);
        opts.set_disable_auto_compactions(true);
        opts.set_level_zero_file_num_compaction_trigger(1000);
        opts.set_compression_type(DBCompressionType::None);

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_block_cache(&rocksdb::Cache::new_lru_cache(256 * 1024 * 1024));
        opts.set_block_based_table_factory(&block_opts);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));

        let db = DB::open(&opts, path)?;
        Ok(RocksDB { db })
    }

    pub fn compact(&self) {
        self.db.compact_range::<&[u8], &[u8]>(None, None);
    }

    pub fn multi_get(&self, keys: &[impl AsRef<[u8]>]) -> Vec<Option<Vec<u8>>> {
        self.db
            .multi_get(keys)
            .into_iter()
            .map(|r| r.ok().flatten())
            .collect()
    }

    pub fn put<K, V>(&self, key: K, value: V) -> anyhow::Result<()>
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        self.db.put(key, value).map_err(|x| anyhow::anyhow!(x))?;
        Ok(())
    }

    pub fn get(&self, key: impl AsRef<[u8]>) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.db.get(key)?)
    }

    pub fn prefix_iterator<'a>(
        &'a self,
        prefix: &str,
    ) -> impl Iterator<Item = (Box<[u8]>, Box<[u8]>)> + 'a {
        let prefix_bytes = prefix.as_bytes().to_vec();
        self.db
            .iterator(rocksdb::IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ))
            .take_while(move |result| {
                result
                    .as_ref()
                    .map(|(k, _)| k.starts_with(&prefix_bytes))
                    .unwrap_or(false)
            })
            .filter_map(|result| result.ok())
    }

    pub fn write_batch<F>(&self, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(&mut WriteBatch),
    {
        let mut batch = WriteBatch::default();
        f(&mut batch);
        self.db.write(batch).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn delete(&self, key: impl AsRef<[u8]>) -> anyhow::Result<()> {
        self.db.delete(key).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn sample_keys(&self, prefix: &str, n: usize) -> Vec<String> {
        let mut rng = rand::rng();
        self.db
            .iterator(rocksdb::IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ))
            .filter_map(|r| r.ok())
            .take_while(|(k, _)| k.starts_with(prefix.as_bytes()))
            .filter(|_| rng.random::<f32>() < 0.01)
            .take(n)
            .map(|(k, _)| String::from_utf8_lossy(&k).to_string())
            .collect()
    }
}
