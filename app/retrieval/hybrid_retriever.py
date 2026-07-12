import time

from app.retrieval.retriever import Retriever
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.reranker import Reranker

from app.ingestion.loader import PDFLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker


class HybridRetriever:

    def __init__(self):

        print("Loading Hybrid Retriever...")

        self.semantic = Retriever()

        loader = PDFLoader()
        pages = loader.load("data/raw/employee_handbook.pdf")

        cleaner = TextCleaner()

        cleaned_pages = [
            {
                "page": page["page"],
                "text": cleaner.clean(page["text"])
            }
            for page in pages
        ]

        chunker = TextChunker()
        chunks = chunker.chunk(cleaned_pages)

        self.keyword = KeywordSearch()
        self.keyword.build_index(chunks)

        self.reranker = Reranker()

    def search(self, query, top_k=3):

        semantic_results = self.semantic.search(query, top_k * 2)
        keyword_results = self.keyword.search(query, top_k * 2)

        fused = {}

        # ----------------------------
        # Semantic contributes 70%
        # ----------------------------
        for chunk in semantic_results:

            key = (chunk["page"], chunk["text"])

            fused[key] = {
                "text": chunk["text"],
                "page": chunk["page"],
                "fusion_score": 0.7 * chunk.get("semantic_score", 0)
            }

        # ----------------------------
        # BM25 contributes 30%
        # ----------------------------
        if keyword_results:

            max_score = max(
                chunk["bm25_score"]
                for chunk in keyword_results
            )

            if max_score == 0:
                max_score = 1

            for chunk in keyword_results:

                key = (chunk["page"], chunk["text"])

                normalized = chunk["bm25_score"] / max_score

                if key not in fused:

                    fused[key] = {
                        "text": chunk["text"],
                        "page": chunk["page"],
                        "fusion_score": 0
                    }

                fused[key]["fusion_score"] += 0.3 * normalized

        combined = sorted(
            fused.values(),
            key=lambda x: x["fusion_score"],
            reverse=True
        )

        candidates = combined[: top_k * 2]

        start = time.perf_counter()

        reranked = self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=top_k
        )

        print(f"Reranking: {time.perf_counter() - start:.3f}s")

        return {
            "results": reranked,
            "debug": {
                "semantic": semantic_results,
                "keyword": keyword_results,
                "fusion": candidates,
                "reranked": reranked
            }
        }