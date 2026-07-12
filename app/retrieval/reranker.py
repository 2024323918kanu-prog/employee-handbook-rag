from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        print("Loading Cross Encoder...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Cross Encoder Loaded!")

    def rerank(self, query, chunks, top_k=3):

        if not chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        return [chunk for score, chunk in ranked[:top_k]]