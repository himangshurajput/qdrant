# Comparison

## Setup

Qdrant 1.20.0 started with `docker compose up -d`. Before any collection was created, `http://localhost:6333/dashboard` returned HTTP 200 and `GET /collections` returned an empty list.

Dataset: 6,000 posts from 20 Newsgroups (headers, footers and quotes stripped, posts shorter than 100 characters dropped, text capped at 4,000 characters), covering all 20 categories.

Embeddings: `all-MiniLM-L6-v2`, 384 dimensions, with the model's final `Normalize` module removed. The stock pipeline L2-normalises every output, and on unit vectors cosine, dot product and Euclidean distance produce the same ranking, which would make Part 2 empty. The raw pooled vectors have norms between 1.01 and 5.58 (mean 2.25); the five query vectors have norms between 5.4 and 7.5. Short posts tend to have larger norms than long ones.

Each collection ended up with 5 segments and all 6,000 vectors HNSW-indexed (`indexing_threshold` set to 100 KB so the small dataset is indexed at all). Latency is the median of 5 repeats per query, then averaged over the 5 queries. Qdrant numbers are measured through the REST client and include the HTTP round trip; the IVF numbers are in-process NumPy.

## Part 2: distance metrics

Query 5, "encryption, government surveillance and the clipper chip", top result per metric (full top-5 in `results/distance_metrics.json`):

| metric | top-1 id | score | doc norm | cosine | dot | length (chars) |
|---|---|---|---|---|---|---|
| Cosine | 1831 | 0.7545 | 1.79 | 0.7545 | 9.22 | 4000 |
| Euclidean | 2978 | 5.5746 | 2.18 | 0.6837 | 10.20 | 746 |
| Dot | 5787 | 14.1848 | 3.88 | 0.5348 | 14.18 | 191 |

The three metrics pick three different top documents from identical vectors and the identical index.

- Cosine only measures the angle. Document 1831 is a long summary of the Clipper Chip and points in almost the same direction as the query, so it wins with 0.75 even though its vector is short (norm 1.79).
- Dot product is magnitude sensitive: `q · d = |q| |d| cos θ`. Document 5787 is a 191-character post whose vector is more than twice as long as 1831's. Its angle is much worse (cosine 0.53) but `3.88 × 0.53` beats `1.79 × 0.75`, so it ranks first. Dot product rewards documents with big norms, which here means short posts, regardless of how well they match.
- Euclidean measures straight-line distance: `|q − d|² = |q|² + |d|² − 2 q·d`. With the query norm fixed, it rewards a large dot product but penalises a large document norm, so it lands between the other two. Document 2978 has a reasonable angle (0.68) and a moderate norm (2.18); 1831 drops to 4th because its small norm leaves it far from the query in absolute terms, and 5787 falls out of the top 5 because its large norm pushes it away.

The same pattern shows on query 1 ("a question about a graphics card driver"): cosine ranks the 1,772-character post 2548 first (norm 1.68, cosine 0.439), dot product ranks the 164-character post 2208 first (norm 4.18, cosine 0.429, dot 12.58), and dot's 2nd result (428, cosine 0.319) is not in the cosine or Euclidean top 5 at all.

## Part 3: HNSW vs exact

Ground truth is `exact: true` on `news_cosine`. `news_cosine` uses Qdrant defaults (`m=16`, `ef_construct=100`); `news_cosine_hnsw_low` has `m=4`, `ef_construct=16`. Overlap is top-5 overlap with exact, averaged over the 5 queries.

| collection | ef | overlap | latency (ms) |
|---|---|---|---|
| exact | - | 1.00 | 3.47 |
| hnsw default | default | 1.00 | 3.42 |
| hnsw default | 16 | 1.00 | 3.49 |
| hnsw default | 64 | 1.00 | 3.35 |
| hnsw default | 128 | 1.00 | 3.48 |
| hnsw low | default | 1.00 | 3.59 |
| hnsw low | 16 | 1.00 | 3.48 |
| hnsw low | 64 | 1.00 | 3.50 |
| hnsw low | 128 | 1.00 | 3.65 |

Both HNSW collections returned the exact top 5 for every query at every `ef`, and latency is flat at about 3.4 ms because the HTTP round trip dominates; the graph walk over 1,200 vectors per segment is a small fraction of that. The under-tuned graph did not lose recall at this scale: with only 1,200 points per segment and 5 segments whose results are merged, even a sparse `m=4` graph reaches the true neighbours, and `ef=16` is already more than three times the result limit. The gap between the two configurations would appear with more vectors per segment or a lower `ef`, where a sparse graph gets trapped in a local optimum.

## Part 4: IVF

`ivf.py` clusters the 6,000 L2-normalised vectors into 40 KMeans clusters (sizes 75 to 264, mean 150), keeps each centroid and its member ids, and `search_ivf` scores only the members of the `nprobe` closest centroids with an exact dot product.

| nprobe | overlap | latency (ms) | query that missed |
|---|---|---|---|
| 1 | 0.96 | 0.06 | "debate about religion and the existence of god": 4/5 |
| 8 | 1.00 | 0.41 | none |

At `nprobe=1` the religion query lost document 5766 (`soc.religion.christian`, exact rank 4, score 0.5073) and replaced it with 2755 (score 0.4908). 5766 sits in the second-closest cluster to the query, not the closest one, so with `nprobe=1` it is never scored; 2755 lives in the closest cluster and fills the slot. At `nprobe=8` the second cluster is included and the top 5 matches exact search.

## Part 5: all methods

| method | top-5 overlap with exact | mean latency (ms) |
|---|---|---|
| exact brute force | 1.00 | 3.47 |
| hnsw default, ef=16 | 1.00 | 3.49 |
| hnsw default, ef=64 | 1.00 | 3.35 |
| hnsw default, ef=128 | 1.00 | 3.48 |
| hnsw low (m=4, ef_construct=16), ef=16 | 1.00 | 3.48 |
| hnsw low (m=4, ef_construct=16), ef=64 | 1.00 | 3.50 |
| hnsw low (m=4, ef_construct=16), ef=128 | 1.00 | 3.65 |
| ivf nprobe=1 | 0.96 | 0.06 |
| ivf nprobe=8 | 1.00 | 0.41 |

### Speed versus accuracy

The method that visibly moved towards exact search as it was given more work was the IVF index. At `nprobe=1` it scored roughly 150 of the 6,000 vectors per query and got 24 of the 25 ground-truth results; at `nprobe=8` it scored about 1,200 vectors, took seven times longer, and matched exact search on every query. That is the tradeoff from the indexing unit in its purest form: an approximate index buys speed by refusing to look at most of the data, and the knob (`nprobe` for IVF, `ef` for HNSW) decides how much of the data it is allowed to look at. Raising the knob makes the candidate set larger, which costs time and recovers the neighbours that were pruned away.

HNSW showed the same shape in principle but not in these numbers. On 1,200 vectors per segment the graph reaches the exact top 5 even at `m=4`, `ef_construct=16`, `ef=16`, and the ef sweep changes nothing but noise. The reason is scale: HNSW's search cost grows roughly logarithmically with collection size, so on a tiny collection the walk from the entry point to the true neighbours is a handful of hops and the recall loss that a sparse graph causes on millions of vectors never materialises. The latency column is also flat because every Qdrant call pays a few milliseconds of HTTP overhead that swamps the index work, while IVF ran in-process. The comparison that matters is within a method: IVF went from 0.06 ms and 0.96 overlap to 0.41 ms and 1.00 overlap, which is exactly "more work, closer to exact".

### Metric choice and index choice are separate decisions

For query 5 the index, the dataset and the vectors never changed, only the distance function. Cosine put document 1831 first, dot product put 5787 first, and Euclidean put 2978 first. Meanwhile every index configuration on the cosine collection, exact, both HNSW graphs at every `ef`, and IVF at `nprobe=8`, agreed on the same top 5. The index decides how fast you find the nearest neighbours; the metric decides what "nearest" means. Picking HNSW over IVF does not change which document is closest, and switching from cosine to dot product does not change how expensive the search is.
