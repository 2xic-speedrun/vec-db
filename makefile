.PHONY: python_test

build:
	cd vec-db && cargo test && cargo build --release
	cp ./vec-db/target/release/libvec_db.so ./python-examples/kmeans/libvec_db.so
	cp ./vec-db/target/release/libvec_db.so ./python-examples/vector-stream/libvec_db.so

python_test_kmeans: build
	cd python-examples/kmeans && python3 kmeans_simple_classification.py && python3 kmeans_sklearn_classification.py

python_test_stream: build
	cd python-examples/vector-stream && python3 VectorStream.py


tests: python_test_kmeans python_test_stream
	echo "Done"
