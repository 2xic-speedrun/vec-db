use rocksdb::{Options, DB};
use std::collections::HashSet;

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

    pub fn get<T: std::hash::Hash + std::cmp::Eq + for<'a> serde::Deserialize<'a>>(
        &self,
        key: &str,
    ) -> anyhow::Result<HashSet<T>> {
        let prefix = format!("{key}:");
        let mut vectors = HashSet::new();

        let iter = self.db.prefix_iterator(&prefix);
        for item in iter {
            let (db_key, value) = item?;
            let key_str = String::from_utf8(db_key.to_vec()).unwrap();
            if key_str.starts_with(&prefix) {
                let vector: T = bincode::deserialize(&value).unwrap();
                vectors.insert(vector);
            }
        }

        Ok(vectors)
    }
}
