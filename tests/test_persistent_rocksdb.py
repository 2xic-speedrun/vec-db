import os
import tempfile
from libvec_db import PyDatabase


def test_persistent_rocksdb():
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
            db_path=db_path,
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
            db_path=db_path,
        )

        duplicates_after_reopen = db_reopened.query(contract1, 5)
        assert len(duplicates_after_reopen) >= 1


def test_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "contracts_with_metadata_db")

        # Simulated contract bytecode vectors with metadata
        contract1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        contract1_metadata = {
            "name": "UniswapV2Router",
            "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "chain": "ethereum",
        }

        # Similar to contract1
        contract2 = [1.1, 2.1, 3.0, 4.0, 5.0]
        contract2_metadata = {
            "name": "SushiSwapRouter",
            "address": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "chain": "ethereum",
        }

        # Different
        contract3 = [10.0, 20.0, 30.0, 40.0, 50.0]
        contract3_metadata = {
            "name": "WETH",
            "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "chain": "ethereum",
        }

        db = PyDatabase.with_persistent_lsh_rocksdb(
            num_hashes=16,
            num_bands=4,
            vector_size=5,
            similarity_threshold=0.8,
            db_path=db_path,
        )

        # Insert with metadata
        db.insert_with_metadata(contract1, contract1_metadata)
        db.insert_with_metadata(contract2, contract2_metadata)
        db.insert_with_metadata(contract3, contract3_metadata)

        # Query with metadata - should find similar contracts
        results = db.query_with_metadata(contract1, 5)
        assert len(results) >= 1, "Should find at least one similar contract"

        # Results are tuples of (vector, metadata, similarity)
        for _, metadata, similarity in results:
            print(
                f"Found similar contract: {metadata.get('name')} (similarity: {similarity:.4f})"
            )
            assert "name" in metadata
            assert "address" in metadata
            assert similarity >= 0.8

        del db

        # Test persistence - metadata should persist
        db_reopened = PyDatabase.with_persistent_lsh_rocksdb(
            num_hashes=16,
            num_bands=4,
            vector_size=5,
            similarity_threshold=0.8,
            db_path=db_path,
        )

        results_after_reopen = db_reopened.query_with_metadata(contract1, 5)
        assert len(results_after_reopen) >= 1, "Metadata should persist after reopen"

        for _, metadata, similarity in results_after_reopen:
            print(
                f"After reopen - Found: {metadata.get('name')} (similarity: {similarity:.4f})"
            )
            assert "name" in metadata


def test_minhash_rocksdb():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "minhash_db")

        contract1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        contract2 = [1.1, 2.1, 3.0, 4.0, 5.0]
        contract3 = [10.0, 20.0, 30.0, 40.0, 50.0]

        db = PyDatabase.with_persistent_min_hash_rocksdb(
            num_hashes=64,
            num_bands=16,
            similarity_threshold=0.5,
            db_path=db_path,
        )

        db.insert_with_metadata(contract1, {"name": "Contract1"})
        db.insert_with_metadata(contract2, {"name": "Contract2"})
        db.insert_with_metadata(contract3, {"name": "Contract3"})

        results = db.query_with_metadata(contract1, 5)
        print(f"MinHash results: {len(results)}")
        for vec, metadata, similarity in results:
            print(f"  {metadata.get('name')} (sim: {similarity:.4f})")

        assert len(results) >= 1, f"Should find at least one similar contract, got {len(results)}"


if __name__ == "__main__":
    test_persistent_rocksdb()
    test_metadata()
    test_minhash_rocksdb()
