"""
Evaluation harness for the Employee Handbook RAG system.

Benchmarks three retrieval configurations against a labeled eval dataset:
  1. Semantic-only (dense embeddings via ChromaDB)
  2. Hybrid (semantic + BM25 fusion, no reranking)
  3. Hybrid + CrossEncoder rerank (the full production pipeline)

Metrics computed per configuration:
  - Retrieval Accuracy (hit rate@K): did we retrieve ANY relevant chunk?
  - Precision@K: of the K retrieved chunks, what fraction were relevant?
  - Recall@K: of all relevant chunks, what fraction did we retrieve?
  - MRR (Mean Reciprocal Rank): how high up was the FIRST relevant chunk?
  - Average latency per configuration
"""

import sys
import os
import json
import time
import statistics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectordb.database import VectorDatabase


# Evaluation K is set higher than the app's production top_k (3), because several
# ground-truth queries have 2 relevant chunks. Evaluating at K=3 would make Recall@K
# structurally capped below 1.0 even for a perfect retriever whenever the 2nd relevant
# chunk lands at rank 4+. K=5 gives a fairer picture of retrieval QUALITY, separate
# from the UI's current display limit. We report both later if needed.
TOP_K = 5


def load_eval_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_to_chunk_id_map(db: VectorDatabase):
    """
    The retrievers don't all carry chunk_id through fusion/reranking (only the raw
    semantic retriever does, after our fix). To score hybrid and reranked results
    against our chunk_id-based ground truth, we build a text -> chunk_id lookup
    once from ChromaDB directly, since chunk text is preserved unmodified through
    fusion and reranking.
    """
    all_data = db.collection.get(include=["documents"])
    return dict(zip(all_data["documents"], all_data["ids"]))


def get_chunk_ids(results, text_to_id):
    """
    Normalizes retrieval results (which may or may not already carry chunk_id)
    into an ordered list of chunk_ids for scoring.
    """
    ids = []
    for r in results:
        if "chunk_id" in r:
            ids.append(r["chunk_id"])
        else:
            ids.append(text_to_id.get(r["text"]))
    return ids


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def hit_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in relevant_ids for cid in top_k) else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_config(name, retrieved_ids_fn, dataset, text_to_id):
    """
    Runs one retrieval configuration across the full eval dataset and
    aggregates metrics + latency.
    """
    precisions, recalls, hits, rrs, latencies = [], [], [], [], []

    for item in dataset:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        start = time.perf_counter()
        retrieved_ids = retrieved_ids_fn(query, text_to_id)
        elapsed = time.perf_counter() - start

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, TOP_K))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, TOP_K))
        hits.append(hit_at_k(retrieved_ids, relevant_ids, TOP_K))
        rrs.append(reciprocal_rank(retrieved_ids, relevant_ids, TOP_K))
        latencies.append(elapsed)

    return {
        "name": name,
        "retrieval_accuracy": statistics.mean(hits),
        "precision_at_k": statistics.mean(precisions),
        "recall_at_k": statistics.mean(recalls),
        "mrr": statistics.mean(rrs),
        "avg_latency_sec": statistics.mean(latencies),
    }


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    dataset = load_eval_dataset(dataset_path)

    print(f"Loaded {len(dataset)} evaluation queries.\n")
    print("Initializing retrieval pipeline (this loads embedding model, "
          "BM25 index, and CrossEncoder)...\n")

    hybrid = HybridRetriever()
    text_to_id = build_text_to_chunk_id_map(hybrid.semantic.db)

    # --- Configuration 1: Semantic-only ---
    def semantic_fn(query, _):
        results = hybrid.semantic.search(query, top_k=TOP_K)
        return get_chunk_ids(results, text_to_id)

    # --- Configuration 2: Hybrid (fusion, no rerank) ---
    def hybrid_no_rerank_fn(query, mapping):
        full = hybrid.search(query, top_k=TOP_K)
        fusion_candidates = full["debug"]["fusion"][:TOP_K]
        return get_chunk_ids(fusion_candidates, mapping)

    # --- Configuration 3: Hybrid + CrossEncoder rerank (production pipeline) ---
    def hybrid_rerank_fn(query, mapping):
        full = hybrid.search(query, top_k=TOP_K)
        return get_chunk_ids(full["results"], mapping)

    configs = [
        ("Semantic Only", semantic_fn),
        ("Hybrid (Semantic + BM25)", hybrid_no_rerank_fn),
        ("Hybrid + CrossEncoder Rerank", hybrid_rerank_fn),
    ]

    results = []
    for name, fn in configs:
        print(f"Running evaluation: {name}...")
        result = evaluate_config(name, fn, dataset, text_to_id)
        results.append(result)

    # --- Print comparison table ---
    print("\n" + "=" * 90)
    print(f"{'Configuration':<32}{'Accuracy':<12}{'Precision@K':<14}{'Recall@K':<12}{'MRR':<10}{'Latency(s)':<10}")
    print("=" * 90)
    for r in results:
        print(f"{r['name']:<32}{r['retrieval_accuracy']:<12.3f}{r['precision_at_k']:<14.3f}"
              f"{r['recall_at_k']:<12.3f}{r['mrr']:<10.3f}{r['avg_latency_sec']:<10.3f}")
    print("=" * 90)
    print(f"\n(Evaluated at K={TOP_K} on {len(dataset)} queries)\n")

    # Save raw results for README generation later
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()