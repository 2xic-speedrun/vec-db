[https://www.pinecone.io/learn/faiss-tutorial/](https://www.pinecone.io/learn/faiss-tutorial/)
- [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
  - Should take a loot at this
- [https://youtu.be/sKyvsdEv6rk](https://youtu.be/sKyvsdEv6rk)
  - Pretty good video
- Index is split into Voronoi cells
  - Basically k-means ? 
  - You have centroids, and check the query vector against the centroid vector. 
    Then we check against all vectors closest to the closest centroid against the query vector.
- Index IVF
  - Allow you to search `n` additional centroids that are close
- Product Quantization
  - Splitting vectors into subvectors
    - Then you cluster all of them nodes into smaller nodes
      - 

[https://www.pinecone.io/learn/vector-indexes/](https://www.pinecone.io/learn/vector-indexes/)
- Maybe, but first version is IVF I think.

