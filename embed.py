import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Normalize

from data import load_documents

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = Path("embeddings")
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
QUERY_VECTORS_PATH = EMBEDDINGS_DIR / "query_vectors.npy"
QUERIES_PATH = Path("queries.md")
BATCH_SIZE = 64


def load_model():
    model = SentenceTransformer(MODEL_NAME)
    last_index = len(model) - 1
    if isinstance(model[last_index], Normalize):
        del model[last_index]
    return model


def load_queries(path=QUERIES_PATH):
    queries = []
    for line in path.read_text().splitlines():
        match = re.match(r"^\d+\.\s+(.*\S)\s*$", line)
        if match:
            queries.append(match.group(1))
    return queries


def embed_texts(model, texts):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def load_embeddings():
    vectors = np.load(VECTORS_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    query_vectors = np.load(QUERY_VECTORS_PATH)
    return vectors, metadata, load_queries(), query_vectors


def main():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    documents = load_documents()
    queries = load_queries()
    model = load_model()
    vectors = embed_texts(model, [d["text"] for d in documents])
    query_vectors = embed_texts(model, queries)
    np.save(VECTORS_PATH, vectors)
    np.save(QUERY_VECTORS_PATH, query_vectors)
    METADATA_PATH.write_text(json.dumps(documents))
    norms = np.linalg.norm(vectors, axis=1)
    print(f"embedded {len(vectors)} documents with {vectors.shape[1]} dimensions")
    print(f"embedded {len(query_vectors)} queries")
    print(f"vector norms: min={norms.min():.3f} mean={norms.mean():.3f} max={norms.max():.3f}")


if __name__ == "__main__":
    main()
