.PHONY: python_test

build:
	cd vec-db && cargo test && cargo build --release

python_test: build
	cp ./vec-db/target/release/libvec_db.so ./python-examples/kmeans/libvec_db.so
	cd python-examples/kmeans && python3 kmeans_simple_classification.py && python3 kmeans_sklearn_classification.py
