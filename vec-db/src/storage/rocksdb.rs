use rocksdb::{Options, WriteBatch, DB};

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

        let db = DB::open(&opts, path)?;
        Ok(RocksDB { db })
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
}
