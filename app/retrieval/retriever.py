from sentence_transformers import SentenceTransformer
from app.vectordb.database import VectorDatabase


class Retriever:

    def __init__(self):

        print("Loading Retriever...")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = VectorDatabase()

    def search(self, query, top_k=3):

        query_embedding = self.embedder.encode(query).tolist()

        results = self.db.search(query_embedding, top_k)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Chroma returns distances (smaller = better)
        distances = results["distances"][0]

        retrieved_chunks = []
        seen_pages = set()

        for doc, metadata, distance in zip(documents, metadatas, distances):

            if metadata["page"] not in seen_pages:

                retrieved_chunks.append(
                    {
                        "text": doc,
                        "page": metadata["page"],
                        "semantic_score": round(1 - distance, 4)
                    }
                )

                seen_pages.add(metadata["page"])

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks