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
        )

        results = []
        seen_pages = set()

        for index in ranked_indices:

            doc = self.documents[index]

            if doc["page"] not in seen_pages:

                results.append(
                    {
                        "text": doc["text"],
                        "page": doc["page"],
                        "bm25_score": round(float(scores[index]), 4)
                    }
                )

                seen_pages.add(doc["page"])

            if len(results) >= top_k:
                break

        return results