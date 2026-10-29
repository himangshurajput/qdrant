import argparse
import json
import statistics
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from embed import load_embeddings
from ivf import build_ivf_index, search_ivf
from qdrant_setup import (
    HNSW_DEFAULT_COLLECTION,
    HNSW_LOW_COLLECTION,
    METRIC_COLLECTIONS,
    QDRANT_URL,
)

TOP_K = 5
EF_VALUES = [16, 64, 128]
NPROBE_VALUES = [1, 8]
REPEATS = 5
TEXT_PREVIEW = 70
RESULTS_DIR = Path("results")
HNSW_COLLECTIONS = {"hnsw_default": HNSW_DEFAULT_COLLECTION, "hnsw_low": HNSW_LOW_COLLECTION}


def timed(search):
    latencies = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        hits = search()
        latencies.append((time.perf_counter() - start) * 1000)
    return hits, statistics.median(latencies)


def qdrant_search(client, collection, query_vector, params=None):
    response = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=TOP_K,
        search_params=params,
        with_payload=False,
    )
    return [(point.id, point.score) for point in response.points]


def overlap(hits, exact_hits):
    return len({h[0] for h in hits} & {h[0] for h in exact_hits}) / TOP_K


def describe(hits, metadata):
    return [
        {
            "id": doc_id,
            "score": round(score, 4),
            "category": metadata[doc_id]["category"],
            "text": metadata[doc_id]["text"][:TEXT_PREVIEW].replace("\n", " "),
        }
        for doc_id, score in hits
    ]


def show(label, hits, latency_ms, metadata, overlap_value=None):
    suffix = "" if overlap_value is None else f"  overlap={overlap_value:.2f}"
    print(f"  {label:<24} {latency_ms:7.2f} ms{suffix}")
    for row in describe(hits, metadata):
        print(f"      {row['id']:>5} {row['score']:9.4f}  {row['category']:<24} {row['text']}")


def record(hits, latency_ms, metadata, exact_hits=None):
    entry = {"latency_ms": round(latency_ms, 3), "top": describe(hits, metadata)}
    if exact_hits is not None:
        entry["overlap"] = overlap(hits, exact_hits)
    return entry


def run_query(client, ivf_index, query, query_vector, metadata):
    print(f"\nquery: {query}")
    print("distance metrics")
    metrics = {}
    for collection in METRIC_COLLECTIONS:
        hits, latency = timed(lambda: qdrant_search(client, collection, query_vector))
        show(collection, hits, latency, metadata)
        metrics[collection] = record(hits, latency, metadata)

    print("hnsw vs exact")
    exact_hits, exact_latency = timed(
        lambda: qdrant_search(client, HNSW_DEFAULT_COLLECTION, query_vector, SearchParams(exact=True))
    )
    show("exact", exact_hits, exact_latency, metadata, 1.0)
    hnsw = {"exact": record(exact_hits, exact_latency, metadata, exact_hits)}
    for label, collection in HNSW_COLLECTIONS.items():
        hits, latency = timed(lambda: qdrant_search(client, collection, query_vector))
        show(f"{label} ef=default", hits, latency, metadata, overlap(hits, exact_hits))
        hnsw[f"{label}_ef_default"] = record(hits, latency, metadata, exact_hits)
        for ef in EF_VALUES:
            params = SearchParams(hnsw_ef=ef)
            hits, latency = timed(lambda: qdrant_search(client, collection, query_vector, params))
            show(f"{label} ef={ef}", hits, latency, metadata, overlap(hits, exact_hits))
            hnsw[f"{label}_ef_{ef}"] = record(hits, latency, metadata, exact_hits)

    print("ivf")
    ivf = {}
    for nprobe in NPROBE_VALUES:
        hits, latency = timed(lambda: search_ivf(ivf_index, query_vector, nprobe, TOP_K))
        show(f"ivf nprobe={nprobe}", hits, latency, metadata, overlap(hits, exact_hits))
        ivf[f"ivf_nprobe_{nprobe}"] = record(hits, latency, metadata, exact_hits)
    return metrics, hnsw, ivf


def summarize(hnsw_runs, ivf_runs):
    summary = {}
    for method in list(hnsw_runs[0]) + list(ivf_runs[0]):
        runs = [r[method] for r in hnsw_runs if method in r] + [r[method] for r in ivf_runs if method in r]
        summary[method] = {
            "mean_overlap": round(statistics.mean(r["overlap"] for r in runs), 3),
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in runs), 3),
        }
    return summary


def print_summary(summary):
    print(f"\n{'method':<24} {'overlap':>8} {'latency ms':>11}")
    for method, row in summary.items():
        print(f"{method:<24} {row['mean_overlap']:>8.2f} {row['mean_latency_ms']:>11.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, help="1-based index into queries.md; runs only that query")
    args = parser.parse_args()

    vectors, metadata, queries, query_vectors = load_embeddings()
    client = QdrantClient(url=QDRANT_URL)
    start = time.perf_counter()
    ivf_index = build_ivf_index(vectors)
    print(f"built ivf index with {len(ivf_index.clusters)} clusters in {time.perf_counter() - start:.1f}s")

    selected = range(len(queries)) if args.query is None else [args.query - 1]
    metric_runs, hnsw_runs, ivf_runs = [], [], []
    for i in selected:
        metrics, hnsw, ivf = run_query(client, ivf_index, queries[i], query_vectors[i], metadata)
        metric_runs.append({"query": queries[i], "metrics": metrics})
        hnsw_runs.append({"query": queries[i], **hnsw})
        ivf_runs.append({"query": queries[i], **ivf})

    summary = summarize([{k: v for k, v in r.items() if k != "query"} for r in hnsw_runs],
                        [{k: v for k, v in r.items() if k != "query"} for r in ivf_runs])
    print_summary(summary)

    if args.query is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "distance_metrics.json").write_text(json.dumps(metric_runs, indent=2))
        (RESULTS_DIR / "hnsw.json").write_text(json.dumps(hnsw_runs, indent=2))
        (RESULTS_DIR / "ivf.json").write_text(json.dumps(ivf_runs, indent=2))
        (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
