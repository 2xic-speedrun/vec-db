.PHONY: test lint

build: lint
	cd vec-db && cargo test && cargo build --release
	cp ./vec-db/target/release/libvec_db.so ./python-examples/kmeans/libvec_db.so
	cp ./vec-db/target/release/libvec_db.so ./python-examples/vector-db/libvec_db.so
	cp ./vec-db/target/release/libvec_db.so ./tests/libvec_db.so

python_test_kmeans: build
	cd python-examples/kmeans && python3 kmeans_simple_classification.py && python3 kmeans_sklearn_classification.py

python_test_stream: build
	cd python-examples/vector-db && python3 test_query.py

test: build
	pytest tests/
	cd vec-db && cargo test 

lint:
	cd vec-db && cargo fmt
	cd vec-db && cargo clippy
	
lint-fix:
	cd vec-db && cargo clippy --fix  --allow-dirty --all-targets --all-features -- -D warnings -W unused-crate-dependencies

check:
	cd vec-db && cargo check --all-targets --profile=test

install:
	cd vec-db && maturin build --release && pip3 install target/wheels/libvec_db-*.whl --force-reinstall  --break-system-packages
