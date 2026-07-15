"""
Per-query diagnostic: compares semantic-only vs hybrid-fusion rankings
to identify which queries regress under fusion, and why.
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.retrieval.hybrid_retriever import HybridRetriever

TOP_K = 5


def load_eval_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_to_chunk_id_map(db):
    all_data = db.collection.get(include=["documents"])
    return dict(zip(all_data["documents"], all_data["ids"]))


def get_ids(results, text_to_id):
    return [text_to_id.get(r["text"]) for r in results]


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    dataset = load_eval_dataset(dataset_path)

    hybrid = HybridRetriever()
    text_to_id = build_text_to_chunk_id_map(hybrid.semantic.db)

    regressions = []

    for item in dataset:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        semantic_results = hybrid.semantic.search(query, top_k=TOP_K)
        semantic_ids = get_ids(semantic_results, text_to_id)

        full = hybrid.search(query, top_k=TOP_K)
        fusion_ids = get_ids(full["debug"]["fusion"][:TOP_K], text_to_id)

        semantic_hits = sum(1 for cid in semantic_ids if cid in relevant_ids)
        fusion_hits = sum(1 for cid in fusion_ids if cid in relevant_ids)

        if fusion_hits < semantic_hits:
            # Find which BM25 chunk(s) got pulled into top-K that displaced a relevant one
            keyword_results = hybrid.keyword.search(query, top_k=TOP_K * 2)
            top_bm25_chunk_id = None
            if keyword_results:
                top_bm25_chunk_id = text_to_id.get(keyword_results[0]["text"])

            regressions.append({
                "query": query,
                "relevant_ids": list(relevant_ids),
                "semantic_top5": semantic_ids,
                "fusion_top5": fusion_ids,
                "semantic_hits": semantic_hits,
                "fusion_hits": fusion_hits,
                "top_bm25_match": top_bm25_chunk_id,
                "top_bm25_was_relevant": top_bm25_chunk_id in relevant_ids
            })

    print(f"\n{len(regressions)} out of {len(dataset)} queries regressed under fusion.\n")
    print("=" * 90)

    for r in regressions:
        print(f"Query: {r['query']}")
        print(f"  Relevant IDs:      {r['relevant_ids']}")
        print(f"  Semantic top-5:    {r['semantic_top5']}  (hits: {r['semantic_hits']})")
        print(f"  Fusion top-5:      {r['fusion_top5']}  (hits: {r['fusion_hits']})")
        print(f"  Top BM25 match:    {r['top_bm25_match']}  (relevant? {r['top_bm25_was_relevant']})")
        print("-" * 90)


if __name__ == "__main__":
    main()