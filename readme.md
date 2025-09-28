$$# Simple vector database
**This is a simple in-memory vector database. Not for use in production.**

This is just a basic vector database that uses k-means to reduce the number of vector lookups. See links below for more advanced indexing methods that will help with speedups. 

See [examples](./python-examples/) for some code examples. 

## Thanks
[Pinecone](https://www.pinecone.io/) has some [great articles](https://www.pinecone.io/learn/series/faiss/) (["vector indexes"](https://www.pinecone.io/learn/series/faiss/vector-indexes/)) and especially some of the videos from [James Briggs](https://www.youtube.com/@jamesbriggs) on the subject (["Faiss - Introduction to Similarity Search"](https://youtu.be/sKyvsdEv6rk)) made things easier to grasp.

These are also good resources
- [Algorithms Powering our Vector Database](https://thebook.devrev.ai/blog/2024-03-04-vector-db-1/)
- [Vector Database: The Secret Behind Large Language Models Capabilities](https://youssefh.substack.com/p/vector-database-the-secret-behind)
- [Everything You Need to Know about Vector Index Basics](https://zilliz.com/learn/vector-index)
- [Nearest Neighbor Indexes: What Are IVFFlat Indexes in Pgvector and How Do They Work](https://www.timescale.com/blog/nearest-neighbor-indexes-what-are-ivfflat-indexes-in-pgvector-and-how-do-they-work/) 
- [Vector Search Explained](https://weaviate.io/blog/vector-search-explained)

## Other good blog posts
- [How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db)
- [Vector Database Basics: HNSW](https://www.tigerdata.com/blog/vector-database-basics-hnsw)
- [Building a high recall vector database serving 1 billion embeddings from a single machine](https://blog.wilsonl.in/corenn/)

## Other algorithms
There are many other algorithms, check the [ann-benchmark](https://ann-benchmarks.com/index.html)
- [faiss-ivf](https://medium.com/@Jawabreh0/inverted-file-indexing-ivf-in-faiss-a-comprehensive-guide-c183fe979d20)
- [annoy](https://github.com/spotify/annoy)
- [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf)
- [ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)
- 