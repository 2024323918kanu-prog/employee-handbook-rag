import chromadb
import uuid


class VectorDatabase:

    def __init__(self, db_path="chroma_db", collection_name="employee_handbook"):

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_embeddings(self, embeddings):

        for item in embeddings:

            self.collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[item["embedding"].tolist()],
                documents=[item["text"]],
                metadatas=[
                    {
                        "page": item["page"]
                    }
                ]
            )

    def count(self):
        return self.collection.count()

    def reset(self):

        self.client.delete_collection("employee_handbook")

        self.collection = self.client.get_or_create_collection(
            name="employee_handbook"
        )

    def search(self, query_embedding, top_k=3):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )