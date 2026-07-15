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
        ids = results["ids"][0]
        distances = results["distances"][0]

        retrieved_chunks = []

        for doc, metadata, chunk_id, distance in zip(documents, metadatas, ids, distances):

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "page": metadata["page"],
                    "semantic_score": round(1 - distance, 4)
                }
            )

        return retrieved_chunks