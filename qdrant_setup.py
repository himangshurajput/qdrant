import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from embed import load_embeddings

QDRANT_URL = "http://localhost:6333"
METRIC_COLLECTIONS = {
    "news_cosine": Distance.COSINE,
    "news_euclid": Distance.EUCLID,
    "news_dot": Distance.DOT,
}
HNSW_DEFAULT_COLLECTION = "news_cosine"
HNSW_LOW_COLLECTION = "news_cosine_hnsw_low"
HNSW_LOW_CONFIG = HnswConfigDiff(m=4, ef_construct=16)
INDEXING_THRESHOLD_KB = 100
UPLOAD_BATCH = 500
INDEX_TIMEOUT_SECONDS = 300


def create_collection(client, name, distance, dim, hnsw_config=None):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=distance),
        hnsw_config=hnsw_config,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD_KB),
    )


def upload_points(client, name, vectors, metadata):
    for start in range(0, len(metadata), UPLOAD_BATCH):
        batch = [
            PointStruct(
                id=doc["id"],
                vector=vectors[doc["id"]].tolist(),
                payload={"category": doc["category"], "text": doc["text"]},
            )
            for doc in metadata[start : start + UPLOAD_BATCH]
        ]
        client.upsert(collection_name=name, points=batch, wait=True)


def wait_for_index(client, name, expected):
    deadline = time.time() + INDEX_TIMEOUT_SECONDS
    while True:
        info = client.get_collection(name)
        indexed = info.indexed_vectors_count or 0
        if info.status == CollectionStatus.GREEN and indexed >= expected:
            return info
        if time.time() > deadline:
            return info
        time.sleep(1)


def main():
    vectors, metadata, _, _ = load_embeddings()
    client = QdrantClient(url=QDRANT_URL)
    dim = vectors.shape[1]
    collections = [(name, distance, None) for name, distance in METRIC_COLLECTIONS.items()]
    collections.append((HNSW_LOW_COLLECTION, Distance.COSINE, HNSW_LOW_CONFIG))
    for name, distance, hnsw_config in collections:
        create_collection(client, name, distance, dim, hnsw_config)
        upload_points(client, name, vectors, metadata)
        info = wait_for_index(client, name, len(vectors))
        print(
            f"{name}: distance={distance.value} points={info.points_count} "
            f"indexed={info.indexed_vectors_count} status={info.status.value}"
        )


if __name__ == "__main__":
    main()
