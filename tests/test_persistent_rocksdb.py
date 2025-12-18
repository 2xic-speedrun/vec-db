import os
import tempfile
from libvec_db import PyDatabase


def test_persistent_rocksdb():
    """Test persistent RocksDB with database reopening"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "contracts_db")
        
        contract1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        contract2 = [1.1, 2.1, 3.0, 4.0, 5.0]
        contract3 = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        # Create and populate database
        db = PyDatabase.with_persistent_lsh_rocksdb(
            num_hashes=16,
            num_bands=4,
            vector_size=5,
            similarity_threshold=0.8,
            db_path=db_path
        )
        
        db.insert(contract1)
        db.insert(contract2)
        db.insert(contract3)
        
        duplicates = db.query(contract1, 5)
        assert len(duplicates) >= 1
        
        del db
        
        # Reopen database to test persistence
        db_reopened = PyDatabase.with_persistent_lsh_rocksdb(
            num_hashes=16,
            num_bands=4,
            vector_size=5,
            similarity_threshold=0.8,
            db_path=db_path
        )
        
        duplicates_after_reopen = db_reopened.query(contract1, 5)
        assert len(duplicates_after_reopen) >= 1


if __name__ == "__main__":
    test_persistent_rocksdb()
    print("✓ Persistent RocksDB test passed")