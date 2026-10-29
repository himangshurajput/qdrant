from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

N_CLUSTERS = 40
RANDOM_STATE = 42


@dataclass
class IVFIndex:
    centroids: np.ndarray
    clusters: list
    vectors: np.ndarray


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def build_ivf_index(vectors, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
    normalized = normalize(np.asarray(vectors, dtype=np.float32))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(normalized)
    clusters = [np.flatnonzero(labels == c) for c in range(n_clusters)]
    return IVFIndex(normalize(kmeans.cluster_centers_), clusters, normalized)


def search_ivf(index, query_vector, nprobe, k):
    query = normalize(np.asarray(query_vector, dtype=np.float32))
    centroid_scores = index.centroids @ query
    probed = np.argsort(-centroid_scores)[:nprobe]
    candidate_ids = np.concatenate([index.clusters[c] for c in probed])
    scores = index.vectors[candidate_ids] @ query
    best = np.argsort(-scores)[:k]
    return [(int(candidate_ids[i]), float(scores[i])) for i in best]


if __name__ == "__main__":
    from embed import load_embeddings

    vectors, metadata, queries, query_vectors = load_embeddings()
    index = build_ivf_index(vectors)
    sizes = [len(c) for c in index.clusters]
    print(f"{len(index.clusters)} clusters, sizes min={min(sizes)} mean={np.mean(sizes):.0f} max={max(sizes)}")
    for query, query_vector in zip(queries, query_vectors):
        print(f"\nquery: {query}")
        for nprobe in (1, 8):
            hits = search_ivf(index, query_vector, nprobe, 5)
            print(f"  nprobe={nprobe}")
            for doc_id, score in hits:
                print(f"    {doc_id:>5}  {score:.4f}  {metadata[doc_id]['category']}")
