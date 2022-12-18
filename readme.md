# Basic vector database in Rust 

Just for fun, and not for production!

Idea is that [https://github.com/2xic-speedrun/latios](https://github.com/2xic-speedrun/latios) might use this as a way to quickly find similar latent embeddings.

### TODO
- Filesystem format
  - Header  
    - metadata   
    - offsets
      - centroid header
      - data header   
  - Centroids header
    - num of centroids
    - ....
  - Data header
    - How to store this efficiently ? 
      - Linked list ? 
    - 
- K-means indexer
  - Partial fit
    -> How it is done in sklearn https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/cluster/_kmeans.py#L1515

- Python api (?)
  -> http://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/

# Api (todo)
`db.save(vec)`
- Save the vector to database

`db.query(vec, n=5)`
- Gives you n (5) most similar vectors to the query vector in the database 
