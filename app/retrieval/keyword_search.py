from rank_bm25 import BM25Okapi


class KeywordSearch:

    def __init__(self):
        self.bm25 = None
        self.documents = []

    def build_index(self, documents):

        self.documents = documents

        tokenized_documents = []

        for doc in documents:
            tokens = doc["text"].lower().split()
            tokenized_documents.append(tokens)

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=3):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:

            doc = self.documents[index]

            results.append(
                {
                    "text": doc["text"],
                    "page": doc["page"],
                    "bm25_score": round(float(scores[index]), 4)
                }
            )

        return results
    def compute_reference_score(self, sample_size=20,seed=42):
        """
        Estimates a stable, corpus-wide BM25 score scale by sampling a handful
        of representative queries (using distinctive terms from the corpus itself)
        and taking the average top score across them. Used as a fixed denominator
        for normalization instead of any single query's own max score.
        """
        import random

        if not self.documents:
            return 1.0
        rng=random.Random(seed)

        sample_docs = rng.sample(self.documents, min(sample_size, len(self.documents)))
        top_scores = []

        for doc in sample_docs:
            pseudo_query = " ".join(doc["text"].split()[:8])  # first few words as a proxy query
            scores = self.bm25.get_scores(pseudo_query.lower().split())
            top_scores.append(max(scores))

        reference = sum(top_scores) / len(top_scores)
        return reference if reference > 0 else 1.0